"""Producer completions, async delegations and liveness ticks borrow a route's committed envelope
through the real session authority: auto-thread hints on the persisted origin must be admitted
exactly once, with a retry landing on the same admission row instead of a second one."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_automation import completion_admission
from gateway.session_cron import unbind_owner
from gateway.session_identity import replace_source
from gateway.session_ingress_context import native_callback, register_transport_home
from hermes_state import SessionDB
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter


class ProducerAdmissionRealAuthority(unittest.IsolatedAsyncioTestCase):
    async def _admitted_once(self, producer, extra):
        """Admit one producer event twice into a private SQLite ledger and return its only row."""
        with tempfile.TemporaryDirectory(prefix='admission-fixture-') as home_dir:
            home = Path(home_dir)
            # No scheduler, so nothing claims the row and no model turn is started.
            with patch.dict('os.environ', {'HERMES_HOME': home_dir, 'DISCORD_ALLOWED_USERS': '42'}), \
                    patch.object(SessionAuthority, '_schedule', lambda self, ref: None), \
                    patch('gateway.session_bot._watch_reply', lambda *args: None):
                runner = GatewayRunner(GatewayConfig(sessions_dir=home / 'sessions'))
                db = SessionDB(home / 'state.db')
                authority = None
                try:
                    runner.session_store._db = db
                    runner._session_db = db
                    authority = await initialize_session_authority(
                        runner, profile_id='default', instance_id='offline-fixture', db=db)
                    adapter = DiscordAdapter(PlatformConfig(
                        enabled=True, token='fixture-token', typing_indicator=False))
                    runner.adapters = {Platform.DISCORD: adapter}
                    runner._wire_adapter_handlers(adapter)
                    register_transport_home(runner, None, home)
                    source = adapter.build_source(chat_id='42', chat_type='thread', thread_id='42',
                                                  user_id='42', scope_id='fixture-guild')
                    human = MessageEvent(text='human', source=source, message_id='human-1')
                    with native_callback(runner, human, home):
                        await authority.admit_native(human)
                    route = runner.session_store._generate_session_key(source)
                    entry = runner.session_store.lookup_by_session_key(route)
                    # The persisted origin keeps the auto-thread hints; the committed envelope does not.
                    entry.origin = replace_source(
                        source, auto_thread_created=True,
                        auto_thread_initial_name='initial title', message_id='stale-trigger')
                    event = dict(type=producer, session_id='proc_fixture', started_at=1.0,
                                 session_key=route, parent_session_id=entry.session_id, **extra)

                    self.assertFalse(adapter.is_connected)  # no transport, no delivery attempted
                    self.assertTrue(await runner._inject_watch_notification(
                        'review complete', event, raise_not_accepted=True))
                    first = completion_admission(runner, event)
                    self.assertIsNotNone(first)
                    self.assertTrue(await runner._inject_watch_notification(
                        'review complete', event, raise_not_accepted=True))
                    second = completion_admission(runner, event)
                    self.assertEqual(first['admission_id'], second['admission_id'])

                    rows = [row for row in list_session_admissions(
                        db, session_id=authority.logical_owner(entry.session_id), pending_only=False)
                        if row['principal_id'].startswith('automation:')]
                    self.assertEqual([row['admission_id'] for row in rows], [first['admission_id']])
                    self.assertEqual(rows[0]['status'], 'queued')
                    self.assertEqual(rows[0]['payload']['native_text_v1']['automation']['identity'],
                                     rows[0]['request_id'])
                    self.assertEqual(json.loads(rows[0]['request_id'])[0], producer)
                    for hint in ('auto_thread_created', 'auto_thread_initial_name'):
                        self.assertNotIn(hint, rows[0]['payload']['native_text_v1']['source'])
                    return rows[0]
                finally:
                    if authority is not None:
                        # initialize_session_authority bound this (authority, loop) pair by home;
                        # left bound it outlives the test holding a closed DB and a dead loop.
                        unbind_owner(authority)
                    db.close()

    async def test_completion_admitted_once_with_auto_thread_origin(self):
        row = await self._admitted_once('completion', {})
        self.assertEqual(json.loads(row['request_id']), ['completion', 'proc_fixture', 1.0])

    async def test_async_delegation_admitted_once_with_auto_thread_origin(self):
        row = await self._admitted_once('async_delegation', {'delegation_id': 'deleg_fixture'})
        self.assertEqual(json.loads(row['request_id']), ['async_delegation', 'deleg_fixture', ''])

    async def test_heartbeat_tick_admitted_once_reads_back_its_tick(self):
        """A liveness tick commits like any other producer and carries its raw output on the stored
        row (``automation.tick``), the queued state a later tick of the same process folds from (the
        join itself is pinned in the route-hints module). One tick's redelivery replays onto this
        row rather than queueing a second one."""
        row = await self._admitted_once('heartbeat', {
            'event_id': 'tick-1', 'output': 'half the suite is green', 'omitted': 2})
        self.assertEqual(json.loads(row['request_id']), ['heartbeat', 'proc_fixture', 'tick-1'])
        self.assertEqual(row['payload']['native_text_v1']['automation']['tick'],
                         {'output': 'half the suite is green', 'omitted': 2})
