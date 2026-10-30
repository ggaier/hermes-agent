"""#39 N1: a recovery dispatch the startup restore gate takes over must settle as owned.

The gate used to absorb the event (`_hm_admit_event` -> `_startup_restore_queue`, `return None`)
without any signal, so the backfill read it as "no consumer" and wrote a retryable
`unadmitted` row while the turn was in fact about to run. The event is now marked as taken
over where it is queued, and the scan settles the row as `processing` instead.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.run import GatewayRunner
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from tests.gateway.test_discord_missed_message_backfill import adapter, make_message


@pytest.mark.asyncio
async def test_startup_gate_marks_the_event_as_taken_over():
    runner, _ = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    event = MessageEvent(
        text="hello", message_type=MessageType.TEXT, source=make_restart_source(chat_id="restore-chat"),
    )

    assert await runner._handle_message(event) is None
    assert runner._startup_restore_queue == [event]
    assert event._gateway_accepted is True, (
        "an event the restore gate absorbed is owned by a consumer, not dropped")


@pytest.mark.asyncio
async def test_recovery_absorbed_by_restore_gate_settles_as_processing(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_handle_message", DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    # The turn lifecycle is not under test here: keep the background task from re-settling the row.
    monkeypatch.setattr(adapter, "_run_processing_hook", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_start_typing_refresh", lambda *a, **k: None)

    runner, _ = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = []
    adapter._message_handler = GatewayRunner._handle_message.__get__(runner, GatewayRunner)

    message = make_message(message_id=9201, content="<@999> please ingest", mentions=[adapter._client.user])

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", candidates)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})

    await adapter._run_missed_message_backfill()

    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status, claim_owner FROM discord_messages WHERE message_id=?", ("9201",)).fetchone())
    assert row == ("processing", None), row
    assert runner._startup_restore_queue, "the turn is owned by the restore queue"

    # The turn finishing still lands the row on its terminal state.
    adapter._record_discord_processing_complete(
        MessageEvent(text=message.content, message_type=MessageType.TEXT,
                     raw_message=message, message_id="9201"),
        ProcessingOutcome.SUCCESS,
    )
    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status, replied FROM discord_messages WHERE message_id=?", ("9201",)).fetchone())
    assert row == ("responded", 1)


@pytest.mark.asyncio
async def test_recovery_with_no_consumer_still_writes_retryable_unadmitted(adapter, monkeypatch):
    """The N1 fix must not turn a genuinely consumer-less dispatch into a settled row."""
    monkeypatch.setattr(adapter, "_handle_message", DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    assert adapter._message_handler is None

    message = make_message(message_id=9202, content="<@999> lost", mentions=[adapter._client.user])

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, "_iter_missed_message_backfill_candidates", candidates)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"123"})

    await adapter._run_missed_message_backfill()

    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        "SELECT status FROM discord_messages WHERE message_id=?", ("9202",)).fetchone())
    assert row == ("unadmitted",), row
