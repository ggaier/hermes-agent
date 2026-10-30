"""#39 N2: live Discord ingress must be fenced by the durable ledger, not only the TTL dedup.

A RESUMED event replay of a message the ledger already settled as answered must not run a
second turn; a concurrent duplicate must still run exactly once. Anything less settled than
"answered" (processing/crashed, failed, unadmitted, discovered) stays runnable, and a ledger
fault fails open to the in-memory dedup instead of dropping a user message.
"""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from tests.gateway.test_discord_missed_message_backfill import adapter, make_message


def _answered(message, adapter):
    """Drive the real ledger writers to a terminal, confirmed answer."""
    event = MessageEvent(
        text=message.content, message_type=MessageType.TEXT,
        raw_message=message, message_id=str(message.id),
    )
    assert adapter._record_recovery_attempt(message, status="queued", owner="scan:1") is True
    adapter._record_discord_processing_start(event, emoji_ack=True)
    adapter._record_discord_processing_complete(event, ProcessingOutcome.SUCCESS)
    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status, replied FROM discord_messages WHERE message_id=?",
        (str(message.id),)).fetchone())
    assert row == ("responded", 1)


@pytest.mark.asyncio
async def test_resumed_replay_of_answered_message_does_not_run_a_second_turn(adapter):
    message = make_message(message_id=9101, content="<@999> restart done", mentions=[adapter._client.user])
    _answered(message, adapter)

    admitted = await adapter._dispatch_discord_message(message)

    assert admitted is False, "an already-answered input must not be executed again"
    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_answered_drop_is_observable(adapter, caplog):
    message = make_message(message_id=9102, content="<@999> dup", mentions=[adapter._client.user])
    _answered(message, adapter)
    with caplog.at_level(logging.INFO, logger="plugins.platforms.discord.adapter"):
        await adapter._dispatch_discord_message(message)
    hits = [r for r in caplog.records if "9102" in r.getMessage() and "responded" in r.getMessage()]
    assert hits, [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_concurrent_duplicates_still_execute_exactly_once(adapter):
    message = make_message(message_id=9103, content="<@999> race", mentions=[adapter._client.user])
    results = await asyncio.gather(
        adapter._dispatch_discord_message(message),
        adapter._dispatch_discord_message(message),
    )
    assert adapter._handle_message.await_count == 1
    assert sorted(results) == [False, True]


@pytest.mark.asyncio
async def test_processing_row_is_not_dropped(adapter):
    """A processing row may be a crashed turn's leftover; it stays runnable (unchanged policy)."""
    message = make_message(message_id=9104, content="<@999> crashed", mentions=[adapter._client.user])
    event = MessageEvent(text=message.content, message_type=MessageType.TEXT,
                         raw_message=message, message_id=str(message.id))
    adapter._record_discord_processing_start(event, emoji_ack=False)

    assert await adapter._dispatch_discord_message(message) is True
    adapter._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_backfill_never_queries_the_ledger(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "false")
    adapter.config.extra["missed_message_backfill"] = {"enabled": False}
    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(adapter, "_with_discord_recovery_db_async", probe)

    message = make_message(message_id=9105, content="<@999> plain", mentions=[adapter._client.user])
    assert await adapter._dispatch_discord_message(message) is True
    probe.assert_not_awaited()
    adapter._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_ledger_read_failure_fails_open_and_warns_once(adapter, caplog):
    """A broken ledger must not swallow user messages; the failure is reported once."""
    def _boom(_fn, _default=None):
        raise RuntimeError("ledger exploded")
    adapter._discord_recovery_store.call = _boom

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
        for message_id in (9106, 9107):
            message = make_message(
                message_id=message_id, content="<@999> hi", mentions=[adapter._client.user])
            assert await adapter._dispatch_discord_message(message) is True

    assert adapter._handle_message.await_count == 2
    warnings = [r for r in caplog.records if "ledger" in r.getMessage().lower()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]

@pytest.mark.asyncio
async def test_ledger_fault_is_re_reported_after_recovery(adapter, caplog):
    """M1: a fault that clears and returns must warn again — a once-per-process flag hides it."""
    calls = {"fail": True}

    async def _flaky(_fn, default=None):
        if calls["fail"]:
            raise RuntimeError("ledger exploded")
        return False  # a real "no such row": a healthy read, not the unread sentinel

    adapter._with_discord_recovery_db_async = _flaky

    with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
        for message_id in (9110, 9111):
            message = make_message(
                message_id=message_id, content="<@999> hi", mentions=[adapter._client.user])
            assert await adapter._dispatch_discord_message(message) is True
        assert len([r for r in caplog.records if "ledger" in r.getMessage().lower()]) == 1

        # Ledger recovers for one message...
        calls["fail"] = False
        message = make_message(message_id=9112, content="<@999> ok", mentions=[adapter._client.user])
        assert await adapter._dispatch_discord_message(message) is True

        # ...then fails again: the outage must be reported again, not stay silent.
        calls["fail"] = True
        message = make_message(message_id=9113, content="<@999> hi", mentions=[adapter._client.user])
        assert await adapter._dispatch_discord_message(message) is True

    warnings = [r for r in caplog.records
                if "ledger" in r.getMessage().lower() and r.levelno >= logging.WARNING]
    assert len(warnings) == 2, [r.getMessage() for r in caplog.records]
