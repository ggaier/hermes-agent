"""Recovery safety contracts through the real adapter and durable ledger."""
import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from tests.gateway.test_discord_missed_message_backfill import adapter, make_message


@pytest.mark.asyncio
async def test_cancelled_turn_is_not_replayed_by_reconnect_scan(adapter, monkeypatch):
    message = make_message(message_id=801, content='<@999> stop this', mentions=[adapter._client.user])
    event = MessageEvent(text=message.content, message_type=MessageType.TEXT,
                         raw_message=message, message_id=str(message.id))
    adapter._record_discord_processing_start(event, emoji_ack=False)
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    adapter._handle_message.assert_not_awaited()
    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status, attempts FROM discord_messages WHERE message_id=?', ('801',)).fetchone())
    assert row == ('cancelled', 0)


@pytest.mark.asyncio
async def test_resumed_callback_reconciles_missed_history(adapter, monkeypatch):
    import asyncio
    from tests.gateway.test_discord_liveness import _LiveBot, _connect
    from tests.gateway.test_discord_missed_message_backfill import FakeChannel

    message = make_message(message_id=802, content='<@999> missed', mentions=[adapter._client.user])
    channel = FakeChannel(history_messages=[message])
    message.channel = channel
    adapter.config.extra['missed_message_backfill'] = {'enabled': False, 'channels': ['123']}
    monkeypatch.setattr(adapter, '_run_post_connect_initialization', AsyncMock())

    def factory(**kwargs):
        bot = _LiveBot(**kwargs)
        bot.get_channel = lambda _id: channel
        return bot

    await _connect(adapter, monkeypatch, factory)
    try:
        adapter.config.extra['missed_message_backfill']['enabled'] = True
        callback = adapter._client._events.get('on_resumed')
        if callback:
            await callback()
        task = adapter._missed_message_backfill_task
        if task:
            await asyncio.wait_for(task, 2)
        assert adapter._handle_message.await_count == 1, 'RESUMED must reconcile configured history'
        adapter._handle_message.assert_awaited_once_with(message, role_authorized=False, recovered=True)
        row = adapter._with_discord_recovery_db(lambda c: c.execute(
            'SELECT attempts FROM discord_messages WHERE message_id=?', ('802',)).fetchone())
        assert row == (1,)
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_recovery_requires_base_adapter_admission_receipt(adapter, monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter
    # Restore the actual Discord normalizer and base-adapter admission path.
    monkeypatch.setattr(adapter, '_handle_message', DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    message = make_message(message_id=803, content='<@999> receipt', mentions=[adapter._client.user])
    assert adapter._message_handler is None
    receipt = await adapter._dispatch_recovered_message(message)
    assert receipt.accepted is False, 'normalization is not an admission receipt when no consumer is installed'
    assert receipt.refused is True, 'no handler is a provable refusal: the row stays retryable'
    assert receipt.unknown is False


@pytest.mark.asyncio
async def test_unadmitted_recovery_dispatch_is_visible_and_not_counted(adapter, monkeypatch):
    """A dispatch the base adapter dropped must not be counted as delivered, and the row must say so."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(adapter, '_handle_message', DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    message = make_message(message_id=804, content='<@999> lost', mentions=[adapter._client.user])

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    scan = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT dispatched, status FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1').fetchone())
    # A scan that leaves a retryable row reports the debt instead of a clean 'success' (#39 N3).
    assert scan == (0, 'partial')
    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status FROM discord_messages WHERE message_id=?', ('804',)).fetchone())
    assert row == ('unadmitted',)


@pytest.mark.asyncio
async def test_history_read_failure_is_visible_in_scan_ledger(adapter, monkeypatch):
    """A channel whose history cannot be read leaves the scan incomplete — never 'success'."""
    from tests.gateway.test_discord_missed_message_backfill import FakeChannel

    class BrokenChannel(FakeChannel):
        def history(self, **_kwargs):
            async def _gen():
                raise RuntimeError('403 Missing Access')
                yield  # pragma: no cover
            return _gen()

    adapter._client.get_channel = lambda _id: BrokenChannel(channel_id=123)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    status, error = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status, error FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1').fetchone())
    assert status == 'partial'
    assert 'Missing Access' in (error or '')


@pytest.mark.asyncio
async def test_readable_scan_is_recorded_success(adapter, monkeypatch):
    """Positive control for the partial-scan status: a clean channel stays 'success'."""
    from tests.gateway.test_discord_missed_message_backfill import FakeChannel

    message = make_message(message_id=805, content='<@999> hi', mentions=[adapter._client.user])
    adapter._client.get_channel = lambda _id: FakeChannel(channel_id=123, history_messages=[message])
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status, error, dispatched FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1').fetchone())
    assert row == ('success', None, 1)


@pytest.mark.asyncio
async def test_recovered_dispatch_claims_dedup_so_replay_cannot_double_run(adapter):
    """RESUMED replays live events while the scan may dispatch the same id; the first admission
    must own the dedup claim so the second one is refused."""
    message = make_message(message_id=806, content='<@999> once', mentions=[adapter._client.user])

    assert (await adapter._dispatch_recovered_message(message)).accepted is True
    assert (await adapter._dispatch_recovered_message(message)).accepted is False
    assert adapter._handle_message.await_count == 1


@pytest.mark.asyncio
async def test_unknown_execution_is_not_replayed(adapter):
    """'unknown' means the outcome was never settled; replay could duplicate a real turn."""
    message = make_message(message_id=807, content='<@999> unsure', mentions=[adapter._client.user])
    adapter._record_recovery_attempt(message, status='unknown')

    assert await adapter._should_backfill_discord_message(message) is False


@pytest.mark.asyncio
async def test_ledger_unavailable_aborts_scan_without_dispatch(adapter, monkeypatch):
    """No durable ledger means no dedupe/claim memory: recover nothing and say so."""
    message = make_message(message_id=808, content='<@999> no db', mentions=[adapter._client.user])
    adapter._client.get_channel = lambda _id: message.channel

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    monkeypatch.setattr(adapter, '_with_discord_recovery_db_async', AsyncMock(return_value=False))

    await adapter._run_missed_message_backfill()

    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_scan_only_reads_configured_channels(adapter, monkeypatch):
    """Allowlist negative and positive control through the real candidate iterator."""
    from tests.gateway.test_discord_missed_message_backfill import FakeChannel

    configured = FakeChannel(channel_id=123, history_messages=[
        make_message(message_id=809, content='<@999> in', mentions=[adapter._client.user])])
    other = FakeChannel(channel_id=777, history_messages=[
        make_message(message_id=810, content='<@999> out', mentions=[adapter._client.user])])
    adapter._client.get_channel = lambda channel_id: {123: configured, 777: other}.get(channel_id)

    got = [m.id async for m in adapter._iter_missed_message_backfill_candidates({'123'})]
    assert got == [809]


@pytest.mark.asyncio
async def test_scan_window_bounds_candidates(adapter, monkeypatch):
    """Messages older than the configured window are never candidates; newer ones are."""
    import datetime as dt

    from tests.gateway.test_discord_missed_message_backfill import FakeChannel

    now = dt.datetime.now(dt.UTC)
    old = make_message(message_id=811, content='<@999> old', mentions=[adapter._client.user])
    old.created_at = now - dt.timedelta(hours=7)
    fresh = make_message(message_id=812, content='<@999> fresh', mentions=[adapter._client.user])
    fresh.created_at = now - dt.timedelta(minutes=1)

    class WindowChannel(FakeChannel):
        def history(self, **kwargs):
            floor = kwargs['after']
            async def _gen():
                for message in self._history_messages:
                    if floor is not None and message.created_at <= floor:
                        continue
                    yield message
            return _gen()

    channel = WindowChannel(channel_id=123, history_messages=[old, fresh])
    adapter._client.get_channel = lambda _id: channel

    got = [m.id async for m in adapter._iter_missed_message_backfill_candidates({'123'})]
    assert got == [812]


def _cold_adapter():
    """A restarted adapter sharing the same profile ledger (same HERMES_HOME)."""
    from types import SimpleNamespace
    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    fresh = DiscordAdapter(PlatformConfig(
        enabled=True, token='t', extra={'missed_message_backfill': {'enabled': True}, 'auto_thread': False}))
    fresh._client = SimpleNamespace(
        user=SimpleNamespace(id=999, bot=True, display_name='Hermes', name='hermes'),
        get_channel=lambda _id: None, guilds=[])
    fresh._ready_event.set()
    fresh._handle_message = AsyncMock(return_value=True)
    return fresh


def _age_row(adapter, message_id, minutes=60):
    import datetime as dt
    old = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=minutes)).isoformat()
    adapter._with_discord_recovery_db(lambda c: c.execute(
        'UPDATE discord_messages SET updated_at=? WHERE message_id=?', (old, str(message_id))))


@pytest.mark.asyncio
async def test_processing_claim_is_not_replayed_after_ttl(adapter, monkeypatch):
    """processing means side effects may already exist; a TTL is not proof they did not."""
    message = make_message(message_id=820, content='<@999> ran', mentions=[adapter._client.user])
    event = MessageEvent(text=message.content, message_type=MessageType.TEXT,
                         raw_message=message, message_id='820')
    adapter._record_discord_processing_start(event, emoji_ack=False)
    _age_row(adapter, 820)

    fresh = _cold_adapter()
    assert await fresh._should_backfill_discord_message(message) is False

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(fresh, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(fresh, '_missed_message_backfill_channels', lambda: {'123'})
    await fresh._run_missed_message_backfill()
    fresh._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_execution_is_not_blindly_replayed(adapter, monkeypatch):
    """FAILED may also have run; only never-admitted rows get a bounded retry."""
    message = make_message(message_id=821, content='<@999> crashed', mentions=[adapter._client.user])
    adapter._record_recovery_attempt(message, status='failed')
    _age_row(adapter, 821)

    fresh = _cold_adapter()
    assert await fresh._should_backfill_discord_message(message) is False


@pytest.mark.asyncio
async def test_unadmitted_row_keeps_bounded_retry(adapter):
    """Positive control: an explicitly never-admitted dispatch may be retried."""
    message = make_message(message_id=822, content='<@999> retry', mentions=[adapter._client.user])
    adapter._record_recovery_attempt(message, status='unadmitted')
    _age_row(adapter, 822)

    fresh = _cold_adapter()
    assert await fresh._should_backfill_discord_message(message) is True


@pytest.mark.asyncio
async def test_completed_cursor_cannot_hide_earlier_incomplete_messages(adapter):
    """A high-id completion must not advance the scan floor past older unprocessed input."""
    import datetime as dt
    from tests.gateway.test_discord_missed_message_backfill import FakeChannel

    old_pending = make_message(message_id=900, content='<@999> first', mentions=[adapter._client.user])
    newer_pending = make_message(message_id=901, content='<@999> second', mentions=[adapter._client.user])
    completed = make_message(message_id=950, content='<@999> done', mentions=[adapter._client.user])

    class SnowflakeChannel(FakeChannel):
        def history(self, **kwargs):
            floor = kwargs['after']
            floor_id = getattr(floor, 'id', None)
            async def _gen():
                for message in sorted(self._history_messages, key=lambda m: m.id):
                    if isinstance(floor, dt.datetime) and message.created_at <= floor:
                        continue
                    if floor_id is not None and message.id <= floor_id:
                        continue
                    yield message
            return _gen()

    channel = SnowflakeChannel(channel_id=123, history_messages=[old_pending, newer_pending, completed])
    adapter._client.get_channel = lambda _id: channel
    adapter._advance_discord_recovery_cursor('123', '950')

    got = [m.id async for m in adapter._iter_missed_message_backfill_candidates({'123'})]
    # Candidates are bounded by the window, not by the completion cursor: the completed high id is
    # still yielded (newest-first) and excluded per message by the durable ledger, never by id order.
    assert sorted(got) == [900, 901, 950]
    adapter._with_discord_recovery_db(lambda c: c.execute(
        "INSERT OR REPLACE INTO discord_messages (message_id, status, replied, updated_at) "
        "VALUES ('950', 'responded', 1, ?)", (__import__('datetime').datetime.now(__import__('datetime').UTC).isoformat(),)))
    assert await adapter._should_backfill_discord_message(completed) is False


@pytest.mark.asyncio
async def test_channel_fetch_failure_is_visible_in_scan_ledger(adapter, monkeypatch):
    """A configured channel that cannot even be fetched is a coverage gap, not a clean scan."""

    class Unfetchable:
        def get_channel(self, _id):
            return None

        async def fetch_channel(self, _id):
            raise RuntimeError('500 Internal Server Error')

    adapter._client.get_channel = lambda _id: None
    adapter._client.fetch_channel = Unfetchable().fetch_channel
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    status, error = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status, error FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1').fetchone())
    assert status == 'partial'
    assert 'channel 123' in (error or '')


@pytest.mark.asyncio
async def test_scan_error_buffer_is_cleared_after_the_scan(adapter, monkeypatch):
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: set())
    await adapter._run_missed_message_backfill()
    assert getattr(adapter, '_recovery_scan_errors', None) is None


@pytest.mark.asyncio
async def test_unadmitted_recovery_releases_dedup_claim(adapter, monkeypatch):
    """An explicit admission refusal must not leave the id claimed forever; a later scan retries."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(adapter, '_handle_message', DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    message = make_message(message_id=823, content='<@999> retry me', mentions=[adapter._client.user])

    assert (await adapter._dispatch_recovered_message(message)).refused is True
    assert adapter._dedup.contains('823') is False

    # A real consumer now exists: the same id must be dispatchable again.
    adapter._message_handler = AsyncMock(return_value=None)
    assert (await adapter._dispatch_recovered_message(message)).accepted is True


# ---------------------------------------------------------------------------
# F1: a scan that lost the admission race must not rewrite the winner's row.
# ---------------------------------------------------------------------------


def _row_status(adapter, message_id):
    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status FROM discord_messages WHERE message_id=?', (str(message_id),)).fetchone())
    return row[0] if row else None


def _seed_row(adapter, message_id, status):
    adapter._with_discord_recovery_db(lambda c: c.execute(
        "INSERT INTO discord_messages (message_id, status, replied, updated_at) VALUES (?, ?, 0, ?)",
        (str(message_id), status, adapter._utc_now_iso())))


async def _scan_after_live_claim(adapter, monkeypatch, message):
    """Scan sees the message as a candidate, then a live consumer claims it before recovery does."""
    original_dispatch = adapter._dispatch_recovered_message

    async def live_wins_first(_message):
        admitted, _ = adapter._discord_message_admission(_message, claim=True)
        assert admitted, 'the live consumer holds the dedup claim'
        event = MessageEvent(text=_message.content, message_type=MessageType.TEXT,
                             raw_message=_message, message_id=str(_message.id))
        adapter._record_discord_processing_start(event, emoji_ack=False)
        return await original_dispatch(_message)

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    monkeypatch.setattr(adapter, '_dispatch_recovered_message', live_wins_first)
    await adapter._run_missed_message_backfill()


@pytest.mark.asyncio
async def test_scan_that_lost_the_admission_race_keeps_the_live_claim(adapter, monkeypatch):
    """A claim lost inside the scan window must not turn a started turn into a replayable row."""
    message = make_message(message_id=830, content='<@999> race', mentions=[adapter._client.user])

    await _scan_after_live_claim(adapter, monkeypatch, message)

    assert _row_status(adapter, 830) == 'processing'
    assert adapter._dedup.contains('830') is True, 'the live claim still owns the id'


@pytest.mark.asyncio
async def test_lost_admission_race_fence_survives_cold_adapter(adapter, monkeypatch):
    """The winner's row is the fence: a restarted adapter (reopened ledger) cannot replay it."""
    message = make_message(message_id=831, content='<@999> race everywhere',
                           mentions=[adapter._client.user])

    await _scan_after_live_claim(adapter, monkeypatch, message)
    _age_row(adapter, 831)

    fresh = _cold_adapter()
    assert await fresh._should_backfill_discord_message(message) is False

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(fresh, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(fresh, '_missed_message_backfill_channels', lambda: {'123'})
    await fresh._run_missed_message_backfill()
    fresh._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_scan_writes_never_move_a_started_or_settled_row(adapter):
    """The guard sits in the write itself, so no recovery write can move a live or terminal row."""
    for message_id, prior in ((832, 'processing'), (833, 'responded'), (834, 'failed'),
                              (835, 'cancelled'), (836, 'unknown')):
        message = make_message(message_id=message_id, content=f'<@999> {prior}',
                               mentions=[adapter._client.user])
        _seed_row(adapter, message_id, prior)

        adapter._record_discord_message_seen(message, status='discovered')
        assert _row_status(adapter, message_id) == prior, f'discovered moved a {prior} row'
        adapter._record_recovery_attempt(message, status='queued')
        assert _row_status(adapter, message_id) == prior, f'queued moved a {prior} row'
        adapter._record_recovery_attempt(message, status='unadmitted')
        assert _row_status(adapter, message_id) == prior, f'unadmitted moved a {prior} row'


async def _scan_queued_race(adapter, monkeypatch, message):
    """A live consumer admits the id and records its own queue claim inside the scan window."""
    original_dispatch = adapter._dispatch_recovered_message

    async def live_queues_first(_message):
        admitted, _ = adapter._discord_message_admission(_message, claim=True)
        assert admitted, 'the live consumer holds the dedup claim'
        adapter._record_discord_message_seen(_message, status='queued')
        return await original_dispatch(_message)

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    monkeypatch.setattr(adapter, '_dispatch_recovered_message', live_queues_first)
    await adapter._run_missed_message_backfill()


@pytest.mark.asyncio
async def test_scan_that_lost_the_queue_race_keeps_the_live_queued_claim(adapter, monkeypatch):
    """'queued' is an admitted claim, not a scan reservation: equal status is not a shared owner."""
    message = make_message(message_id=850, content='<@999> queued race',
                           mentions=[adapter._client.user])

    await _scan_queued_race(adapter, monkeypatch, message)

    assert _row_status(adapter, 850) == 'queued', "a losing scan must not settle another owner's claim"
    assert adapter._dedup.contains('850') is True, 'the live claim still owns the id'
    scan = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT dispatched FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1').fetchone())
    assert scan == (0,), 'a lost race is not a dispatch'


@pytest.mark.asyncio
async def test_live_queued_claim_fence_survives_cold_adapter(adapter, monkeypatch):
    """A live queue claim the scan failed to settle must not be replayed by a restarted adapter."""
    message = make_message(message_id=851, content='<@999> queued everywhere',
                           mentions=[adapter._client.user])

    await _scan_queued_race(adapter, monkeypatch, message)
    _age_row(adapter, 851)

    fresh = _cold_adapter()
    assert await fresh._should_backfill_discord_message(message) is False

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(fresh, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(fresh, '_missed_message_backfill_channels', lambda: {'123'})
    await fresh._run_missed_message_backfill()
    fresh._handle_message.assert_not_awaited()


class _WriteFailingLedgerConn:
    """Ledger connection whose message-row INSERTs fail (read-only/full disk); reads still work."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, *params):
        if sql.lstrip().upper().startswith('INSERT INTO DISCORD_MESSAGES'):
            raise sqlite3.OperationalError('attempt to write a readonly database')
        return self._conn.execute(sql, *params)

    def commit(self):
        self._conn.commit()


@pytest.mark.asyncio
async def test_scan_dispatches_nothing_when_the_durable_claim_write_fails(adapter, monkeypatch):
    """A working SELECT 1 is not a claim: no durable 'queued' row means no business dispatch."""
    message = make_message(message_id=852, content='<@999> no claim',
                           mentions=[adapter._client.user])
    store = adapter._discord_recovery_store
    real_call = store.call

    def failing_call(fn, default=None):
        return real_call(lambda conn: fn(_WriteFailingLedgerConn(conn)), default)

    monkeypatch.setattr(store, 'call', failing_call)

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    adapter._handle_message.assert_not_awaited()
    assert adapter._dedup.contains('852') is False, 'nothing may be claimed for a turn that never ran'
    row = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT status FROM discord_messages WHERE message_id=?', ('852',)).fetchone())
    assert row is None


# ---------------------------------------------------------------------------
# F2: a recovered active-session bypass command executes inline and is admitted.
# ---------------------------------------------------------------------------


def _dense_session_key(adapter, message_id):
    """The key the real ingress derives for the fixture's group channel."""
    return adapter._source_session_key(adapter.build_source(
        chat_id='123', chat_name='emo / #wiki-inbox', chat_type='group', user_id='42',
        user_name='Emo', guild_id='777', message_id=str(message_id)))


@pytest.mark.asyncio
async def test_recovered_bypass_command_runs_inline_once_and_is_fenced(adapter, monkeypatch):
    """A /approve recovered while the session is busy runs inline (once) and counts as admitted."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(adapter, '_handle_message', DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    message = make_message(message_id=840, content='/approve', mentions=[adapter._client.user])

    handled = []

    async def handler(event):
        handled.append(event.text)
        return 'approved'

    sent = []

    async def send_with_retry(chat_id, content, **kwargs):
        sent.append(content)
        return SimpleNamespace(success=True, message_id='1')

    adapter._message_handler = handler
    monkeypatch.setattr(adapter, '_send_with_retry', send_with_retry)
    adapter._active_sessions[_dense_session_key(adapter, 840)] = asyncio.Event()

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    assert handled == ['/approve'], 'the inline side effect runs exactly once'
    assert sent == ['approved'], 'the command reply reached the user'
    assert not adapter._session_tasks, 'the bypass path must not spawn a session turn'
    scan = adapter._with_discord_recovery_db(lambda c: c.execute(
        'SELECT dispatched FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1').fetchone())
    assert scan == (1,), 'an inline-executed command is an admitted dispatch'

    # Durable fence: a restarted adapter with a reopened ledger must not re-run the command.
    fresh = _cold_adapter()
    _age_row(adapter, 840)
    monkeypatch.setattr(fresh, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(fresh, '_missed_message_backfill_channels', lambda: {'123'})
    await fresh._run_missed_message_backfill()
    fresh._handle_message.assert_not_awaited()
    assert handled == ['/approve']


@pytest.mark.asyncio
async def test_control_command_refused_by_ingress_is_still_retryable(adapter, monkeypatch):
    """Positive control for the bypass fix: a command nobody admitted keeps a bounded retry."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(adapter, '_handle_message', DiscordAdapter._handle_message.__get__(adapter))
    adapter.config.extra.update(auto_thread=False, reactions=False)
    message = make_message(message_id=841, content='/approve', mentions=[adapter._client.user])

    async def candidates(_channels):
        yield message

    monkeypatch.setattr(adapter, '_iter_missed_message_backfill_candidates', candidates)
    monkeypatch.setattr(adapter, '_missed_message_backfill_channels', lambda: {'123'})
    await adapter._run_missed_message_backfill()

    assert _row_status(adapter, 841) == 'unadmitted'
    fresh = _cold_adapter()
    assert await fresh._should_backfill_discord_message(message) is True

