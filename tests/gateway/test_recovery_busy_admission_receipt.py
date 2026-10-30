"""#39 N3: every busy-path *consumption* must leave an admission receipt.

Production (2026-10-30): while a turn was already running, the reconnect backfill re-dispatched a
message the outage had swallowed. The busy path consumed it — interrupt mode redirected the text
into the running turn and answered with `↪ Redirected current run` anchored to it — but nothing
stamped ``event._gateway_accepted``, so ``_dispatch_recovered_message`` read "not delivered" and
the scan wrote a *retryable* ``unadmitted`` row for input that had already executed. A second
message that landed in the busy TEXT debounce buffer was consumed the same way: left retryable and
silent, because nothing drains a buffer for an event no turn owns.

Two contracts are pinned here:

* a consumed event carries ``_gateway_accepted``; a *provable refusal* carries
  ``_gateway_refused`` instead and never both;
* recovery settles a consumed event as ``processing`` (never retryable) even when it cannot prove
  the receipt, and the scan records the retry debt it leaves behind.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from tests.gateway.test_active_session_text_merge import _make_adapter, _make_event
from tests.gateway.test_discord_missed_message_backfill import adapter, make_message


class _Receiver:
    """A running turn's agent: reports success from redirect()/steer()."""

    _supports_active_turn_redirect = True

    def __init__(self):
        self.redirected: list[str] = []
        self.steered: list[str] = []

    def redirect(self, text):
        self.redirected.append(text)
        return True

    def steer(self, text):
        self.steered.append(text)
        return True


def _event(text: str = "已恢复", message_id: str = "m-1") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm",
    )
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, message_id=message_id,
    )


def _seed_running_turn(runner, key: str, receiver: _Receiver) -> MessageEvent:
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    opening = MessageEvent(text="earlier question", source=source, message_id="opening")
    turn = runner._session_state(key).turn
    turn.agent, turn.event = receiver, opening
    return opening


# --------------------------------------------------------------------------- R1 / R2
# A steer or a redirect puts the text INSIDE the running turn: the event is consumed and must be
# receipted as such, or a caller that rebinds the event replays an instruction that already ran.


@pytest.mark.asyncio
async def test_busy_redirect_stamps_an_admission_receipt():
    runner, _ = make_restart_runner()
    receiver = _Receiver()
    _seed_running_turn(runner, "key", receiver)
    event = _event()

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", receiver)

    assert outcome.redirected is True
    assert receiver.redirected, "the redirect must actually reach the running turn"
    assert event._gateway_accepted is True, (
        "the redirect injected this text into the running turn; a caller holding the event must "
        "not read it as undelivered and run it again")


@pytest.mark.asyncio
async def test_busy_steer_stamps_an_admission_receipt():
    runner, _ = make_restart_runner()
    receiver = _Receiver()
    _seed_running_turn(runner, "key", receiver)
    event = _event()

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", receiver)

    assert outcome.steered is True
    assert event._gateway_accepted is True, (
        "a steered follow-up already rode inside the running turn")


# --------------------------------------------------------------------------- debounce
# queue-mode busy text is buffered and flushed as the next turn; the event itself is absorbed, so
# the buffer — not the event — is what a later drain runs.


@pytest.mark.asyncio
async def test_busy_text_debounce_stamps_an_admission_receipt():
    adapter_ = _make_adapter()
    adapter_._busy_text_mode = "queue"
    event = _make_event("已恢复")
    session_key = adapter_._event_session_key(event)
    adapter_._active_sessions[session_key] = asyncio.Event()  # a turn is running

    await adapter_.handle_message(event)

    assert session_key in adapter_._text_debounce, "the follow-up was buffered for the next turn"
    assert event._gateway_accepted is True, (
        "the debounce buffer owns this text; leaving the event unaccepted strands it as retryable")


# --------------------------------------------------------------------------- R3
# Provable refusals keep their semantics: no receipt, and an explicit refusal marker so recovery
# can tell "never consumed" (retryable) from "consumed without a receipt" (never retryable).


@pytest.mark.asyncio
async def test_unauthorized_busy_message_is_refused_without_a_receipt():
    runner, _ = make_restart_runner()
    runner._is_user_authorized = lambda _source: False
    event = _event()

    assert await runner._handle_active_session_busy_message(event, "key") is True
    assert event._gateway_accepted is False
    assert getattr(event, "_gateway_refused", False) is True


@pytest.mark.asyncio
async def test_bot_loop_budget_refusal_has_no_receipt():
    runner, _ = make_restart_runner()
    runner._admit_bot_message_for_source = lambda _source: False
    event = _event()

    assert await runner._handle_active_session_busy_message(event, "key") is True
    assert event._gateway_accepted is False
    assert getattr(event, "_gateway_refused", False) is True


@pytest.mark.asyncio
async def test_draining_busy_rejection_has_no_receipt():
    runner, platform_adapter = make_restart_runner()
    runner._draining = True  # restarting/stopping: the follow-up is refused, not consumed
    event = _event()

    assert await runner._handle_active_session_busy_message(event, "key") is True
    assert event._gateway_accepted is False
    assert getattr(event, "_gateway_refused", False) is True
    assert platform_adapter.sent, "the user is told the gateway is draining"


@pytest.mark.asyncio
async def test_no_consumer_is_refused_without_a_receipt():
    adapter_ = _make_adapter()
    adapter_._message_handler = None  # nothing can ever take the event
    event = _make_event("已恢复")

    await adapter_.handle_message(event)

    assert event._gateway_accepted is False
    assert getattr(event, "_gateway_refused", False) is True


# --------------------------------------------------------------------------- R4
# The recovery scan must never leave a consumed input retryable — and the reproduction of the
# production defect is exactly that: consumed by the busy path, no receipt, retryable row.


def _recovery_adapter(adapter_, runner) -> None:
    """Point a Discord adapter at *runner*'s live ingress (real ``_handle_message``)."""
    adapter_._message_handler = runner._handle_message
    adapter_._handle_message = DiscordAdapter._handle_message.__get__(adapter_)


async def _run_scan(monkeypatch, adapter_, message) -> None:
    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter_, "_iter_missed_message_backfill_candidates", candidates)
    monkeypatch.setattr(adapter_, "_missed_message_backfill_channels", lambda: {"123"})
    await adapter_._run_missed_message_backfill()


def _ledger_row(adapter_, message_id) -> tuple:
    return adapter_._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status, claim_owner FROM discord_messages WHERE message_id=?",
        (str(message_id),)).fetchone())


@pytest.mark.asyncio
async def test_recovery_redirected_into_a_running_turn_settles_as_processing(adapter, monkeypatch):
    """The production shape: a busy turn redirects the recovered text and answers it."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter.config.extra.update(auto_thread=False, reactions=False)
    monkeypatch.setattr(adapter, "_run_processing_hook", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_start_typing_refresh", lambda *a, **k: None)

    runner, _ = make_restart_runner()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._busy_input_mode = "interrupt"
    receiver = _Receiver()
    key = "recovery-busy"
    _seed_running_turn(runner, key, receiver)
    _recovery_adapter(adapter, runner)

    # The session is busy: the recovered event takes the busy path, not a fresh turn.
    adapter._active_sessions[key] = asyncio.Event()
    monkeypatch.setattr(adapter, "_event_session_key", lambda _event: key)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    message = make_message(message_id=9301, content="<@999> 已恢复", mentions=[adapter._client.user])
    await _run_scan(monkeypatch, adapter, message)

    assert receiver.redirected, "the recovered text was injected into the running turn"
    assert _ledger_row(adapter, 9301) == ("processing", None), (
        "an input the running turn consumed must not be left retryable")
    counts = adapter._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status, dispatched, pending_retry FROM discord_recovery_scans "
        "ORDER BY started_at DESC LIMIT 1").fetchone())
    assert counts == ("success", 1, 0), counts


@pytest.mark.asyncio
async def test_recovery_consumed_without_a_receipt_is_not_left_retryable(adapter, monkeypatch):
    """Fail closed: a consumer that took the event but left no receipt is not retryable.

    Stands in for any busy-path consumer that forgets the receipt — the production bug. Recovery
    cannot prove the input did not run, so it must not hand it back to the retry pool.
    """
    adapter.config.extra.update(auto_thread=False, reactions=False)
    monkeypatch.setattr(adapter, "_run_processing_hook", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_start_typing_refresh", lambda *a, **k: None)
    adapter._handle_message = DiscordAdapter._handle_message.__get__(adapter)
    adapter._message_handler = lambda _event: None  # a consumer exists; it just leaves no receipt

    async def _consuming_handler(event, session_key):
        return True  # consumed the event, forgot the receipt

    adapter.set_busy_session_handler(_consuming_handler)
    monkeypatch.setattr(adapter, "_event_session_key", lambda _event: "busy-key")
    adapter._active_sessions["busy-key"] = asyncio.Event()

    message = make_message(message_id=9302, content="<@999> 已恢复", mentions=[adapter._client.user])
    await _run_scan(monkeypatch, adapter, message)

    assert _ledger_row(adapter, 9302) == ("processing", None), (
        "unknown outcome is fail-closed: settle it, never retry it")
    assert adapter._dedup.contains("9302"), (
        "a possibly-consumed id keeps its claim so a replay cannot run it twice")


@pytest.mark.asyncio
async def test_recovery_without_a_consumer_stays_retryable(adapter, monkeypatch):
    """A provable refusal is still retried: nothing ever saw this message."""
    adapter.config.extra.update(auto_thread=False, reactions=False)
    adapter._handle_message = DiscordAdapter._handle_message.__get__(adapter)
    assert adapter._message_handler is None

    message = make_message(message_id=9303, content="<@999> lost", mentions=[adapter._client.user])
    await _run_scan(monkeypatch, adapter, message)

    assert _ledger_row(adapter, 9303) == ("unadmitted", None), (
        "a message no consumer could reach must stay retryable")


# --------------------------------------------------------------------------- R5
# A scan that leaves retryable rows must say so; "success" alone reads as a clean sweep.


@pytest.mark.asyncio
async def test_scan_completion_records_the_retry_debt(adapter, monkeypatch, caplog):
    adapter.config.extra.update(auto_thread=False, reactions=False)
    adapter._handle_message = DiscordAdapter._handle_message.__get__(adapter)
    assert adapter._message_handler is None  # every candidate stays retryable

    messages = [
        make_message(message_id=9401, content="<@999> one", mentions=[adapter._client.user]),
        make_message(message_id=9402, content="<@999> two", mentions=[adapter._client.user]),
    ]

    async def candidates(_channels):
        for message in messages:
            yield message

    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", candidates)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})

    with caplog.at_level(logging.INFO):
        await adapter._run_missed_message_backfill()

    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status, scanned, missed, dispatched, pending_retry FROM discord_recovery_scans "
        "ORDER BY started_at DESC LIMIT 1").fetchone())
    assert row == ("partial", 2, 2, 0, 2), (
        "the scan left two retryable rows: the record must carry the debt, not report a clean run")
    assert "pending_retry=2" in caplog.text, caplog.text
