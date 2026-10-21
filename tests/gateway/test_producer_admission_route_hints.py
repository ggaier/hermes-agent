"""A producer wake whose persisted route origin carries Discord auto-thread hints is still admitted,
without widening routing comparability: every other origin, owner or adapter difference stays a
refusal. The pre-fix comparator is swapped back in-process for the negative control, so the admitted
cases are proven to depend on the shipped comparison rather than on the harness.
"""
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import gateway.session_automation as session_automation
from gateway.config import Platform
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session import SessionSource
from gateway.session_identity import replace_source
from gateway.wake import WakeNotAccepted
from tools.process_registry_notifications import join_heartbeat_outputs

ROUTE = 'agent:main:discord:thread:42:42'
SESSION_ID = 'fixture-parent'
TEXT = 'review complete'
HINTS = ('auto_thread_created', 'auto_thread_initial_name')
# Every source field that says where (and as whom) a message is routed; none may become comparable.
ROUTING_KEYS = ('platform', 'chat_id', 'chat_type', 'thread_id', 'user_id', 'user_name', 'chat_name',
                'scope_id', 'parent_chat_id', 'profile', 'prospective_thread_id')
# Origin routing keys the admitted source must still carry the origin's own value for.
ADMITTED_ROUTING_KEYS = ('platform', 'chat_id', 'chat_type', 'thread_id', 'user_id', 'user_name',
                         'scope_id', 'parent_chat_id', 'profile', 'prospective_thread_id')


def base_route_identity(source):
    """The pre-fix comparator: drop the platform reply anchor and nothing else."""
    return {key: value for key, value in source.to_dict().items() if key != 'message_id'}


class _ProducerRunner(GatewayNotificationsMixin):
    """The only inputs a producer wake resolves: the persisted origin, the adapter, the route row."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.entry = None
        self.session_store = NS(
            lookup_by_session_key=lambda key: self.entry if key == ROUTE else None,
            _generate_session_key=lambda source: ROUTE if source.thread_id == '42' else 'wrong-thread',
        )

    def _build_process_event_source(self, evt):
        return self.entry.origin

    def _resolve_injection_adapter(self, platform_name, source=None):
        return self.adapter

    def _adapter_for_source(self, source):
        return self.adapter


class ProducerAdmissionRouteHints(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The committed human source a producer turn borrows: routing keys, no auto-thread hints.
        self.committed = SessionSource(
            platform=Platform.DISCORD, chat_id='42', chat_type='thread', thread_id='42',
            user_id='7', user_name='human', scope_id='guild', parent_chat_id='channel',
        )
        self.adapter = NS()
        self.runner = _ProducerRunner(self.adapter)
        self.entry = NS(session_key=ROUTE, session_id=SESSION_ID, suspended=False)
        self.runner.entry = self.entry
        self.snapshot_calls = []
        self.admissions = []
        authority = NS(runner=self.runner, logical_owner=lambda sid: sid,
                       db=NS(get_compression_tip=lambda sid: sid))

        async def admit(adapter, event, identity, **kwargs):
            # The real snapshot runs here; a refusal (RuntimeStoreError) leaves no admission behind.
            self.snapshot_calls.append(identity)
            payload, entry = session_automation.snapshot_automation(authority, adapter, event, identity)
            self.admissions.append({'identity': identity, 'payload': payload, 'route': entry.session_key})
            return NS()  # the receipt is not asserted here; the ledger identity is a real-DB concern

        authority.admit_automation = admit
        envelope = {'source': self.committed.to_dict(), 'provenance': {}, 'route': ROUTE}

        def restore(payload, runner=None):
            return NS(source=SessionSource.from_dict(payload['native_text_v1']['source']))

        for patcher in (
            patch('gateway.session_authorities.active_authority', return_value=authority),
            patch.object(session_automation, 'committed_route_envelope', return_value=envelope),
            patch('gateway.session_envelope.restore_native', side_effect=restore),
            patch.object(session_automation, 'withdraw_process_ticks', new=self._noop),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self._reset_subject()

    async def _noop(self, *args, **kwargs):
        pass

    def _reset_subject(self):
        """The failing state: a persisted origin carrying both hints and a stale reply anchor."""
        self.entry.origin = replace_source(
            self.committed, auto_thread_created=True,
            auto_thread_initial_name='initial thread title', message_id='stale-trigger')
        self.evt = dict(type='completion', session_id='proc_fixture', started_at=1.0,
                        session_key=ROUTE, parent_session_id=SESSION_ID)
        self.runner.__dict__.pop('_adapter_for_source', None)  # back to the class default
        self.snapshot_calls.clear()
        self.admissions.clear()

    async def _inject(self):
        return await self.runner._inject_watch_notification(TEXT, self.evt, raise_not_accepted=True)

    def _assert_persisted_route(self, payload):
        """The admitted payload borrows the committed routing identity, never the origin's hints."""
        source = payload['native_text_v1']['source']
        self.assertEqual(source, {key: value for key, value in self.committed.to_dict().items()
                                  if key != 'message_id'})
        # Anchored to the origin itself, not only to the mocked committed envelope: borrowing the
        # committed source must never move a routing key off the route the wake came from.
        origin = self.entry.origin.to_dict()
        for key in ADMITTED_ROUTING_KEYS:
            self.assertEqual(source.get(key), origin.get(key), key)
        for hint in HINTS:
            self.assertNotIn(hint, source)
        self.assertEqual(payload['native_text_v1']['route'], ROUTE)

    async def test_both_hints_admitted_with_one_identity_and_payload_on_retry(self):
        # The producer identity of the event, computed here from its own fields.
        expected_identity = json.dumps(['completion', 'proc_fixture', 1.0], separators=(',', ':'))
        self.assertTrue(await self._inject())
        self.assertEqual(len(self.snapshot_calls), 1)
        self.assertEqual(len(self.admissions), 1)
        first = self.admissions[0]
        self._assert_persisted_route(first['payload'])
        # The persisted dedupe key of the admission is the producer identity both times; the row a
        # retry lands on is the same one (real request_id replay, in the real-authority module).
        self.assertEqual(first['payload']['native_text_v1']['automation']['identity'], expected_identity)

        self.assertTrue(await self._inject())  # the identical wake, retried
        self.assertEqual(len(self.snapshot_calls), 2)
        self.assertEqual([row['identity'] for row in self.admissions], [expected_identity] * 2)
        self.assertEqual(self.admissions[1]['payload'], first['payload'])

    async def test_auto_thread_created_hint_alone_admitted(self):
        self.entry.origin = replace_source(self.committed, auto_thread_created=True)
        self.assertTrue(await self._inject())
        self.assertEqual(len(self.snapshot_calls), 1)
        self.assertEqual(len(self.admissions), 1)
        self._assert_persisted_route(self.admissions[0]['payload'])

    async def test_auto_thread_initial_name_hint_alone_admitted(self):
        self.entry.origin = replace_source(self.committed, auto_thread_initial_name='initial thread title')
        self.assertTrue(await self._inject())
        self.assertEqual(len(self.snapshot_calls), 1)
        self.assertEqual(len(self.admissions), 1)
        self._assert_persisted_route(self.admissions[0]['payload'])

    async def test_unchanged_origin_admitted(self):
        """No-regression control: an origin with no hints and the stale anchor is admitted before the
        patch too, so this case is green either way and is not evidence for the patch."""
        self.entry.origin = replace_source(self.committed, message_id='stale-trigger')
        self.assertTrue(await self._inject())
        self.assertEqual(len(self.snapshot_calls), 1)
        self.assertEqual(len(self.admissions), 1)
        self._assert_persisted_route(self.admissions[0]['payload'])

    async def test_heartbeat_tick_admitted_with_its_own_output(self):
        self.evt = dict(type='heartbeat', session_id='proc_tick', event_id='tick-1',
                        output='half the suite is green', omitted=2,
                        session_key=ROUTE, parent_session_id=SESSION_ID)
        self.assertTrue(await self._inject())
        self.assertEqual(len(self.snapshot_calls), 1)
        self.assertEqual(len(self.admissions), 1)
        stored = self.admissions[0]
        self._assert_persisted_route(stored['payload'])
        self.assertEqual(stored['payload']['native_text_v1']['automation']['tick'],
                         {'output': 'half the suite is green', 'omitted': 2})
        self.assertEqual(json.loads(stored['identity']), ['heartbeat', 'proc_tick', 'tick-1'])

    def test_heartbeat_outputs_fold_into_one_tick(self):
        """The join ``prepare_heartbeat`` applies when a later tick of the same process replaces one
        still waiting: the waiting tick's text and cut count survive, counted once. Driven directly
        here because the end-to-end fold reads the waiting tick back out of a real admission ledger
        (``list_session_admissions`` / ``_queued_tick``, which only folds a row the scheduler left
        ``queued``), and this file's synthetic authority has no ledger; the case above pins what one
        admitted tick stores."""
        earlier = {'output': 'half the suite is green\n', 'omitted': 2}
        self.assertEqual(join_heartbeat_outputs(earlier, {'output': 'the other half too\n', 'omitted': 0}),
                         {'output': 'half the suite is green\nthe other half too\n', 'omitted': 2})
        # A later tick that was itself cut leaves none of the earlier output, whose characters
        # still count as omitted.
        self.assertEqual(join_heartbeat_outputs(earlier, {'output': 'tail', 'omitted': 5}),
                         {'output': 'tail', 'omitted': 5 + earlier['omitted'] + len(earlier['output'])})

    def test_route_identity_separates_every_routing_key(self):
        """The comparator itself, unshadowed by the upstream session-key/owner guards that already
        refuse some of these rows: widening the exclusion past the two presentation hints must fail
        here for every routing key, whichever guard the wake would hit first."""
        base = session_automation._route_identity(self.committed)
        drift = {
            'platform': Platform.SLACK, 'chat_id': '43', 'chat_type': 'channel', 'thread_id': '99',
            'user_id': '8', 'user_name': 'stranger', 'chat_name': 'other thread',
            'scope_id': 'other-guild', 'parent_chat_id': 'other-channel',
            'profile': 'other-profile', 'prospective_thread_id': 'other-thread',
        }
        self.assertEqual(sorted(drift), sorted(ROUTING_KEYS))
        for key, value in drift.items():
            with self.subTest(key=key):
                self.assertNotEqual(base, session_automation._route_identity(
                    replace_source(self.committed, **{key: value})))

    def test_presentation_hint_exclusion_set_is_exactly_the_auto_thread_hints(self):
        """Data contract: the exclusion set is closed. A third key added here silently widens what a
        producer wake may ignore, which is the bug the two admitted cases exist to catch."""
        self.assertEqual(session_automation._ROUTE_PRESENTATION_HINTS,
                         frozenset({'auto_thread_created', 'auto_thread_initial_name'}))

    async def test_base_comparator_refuses_the_wake_the_shipped_one_admits(self):
        """Negative control: with only the pre-fix comparator restored, the same wake is refused."""
        with patch.object(session_automation, '_route_identity', base_route_identity):
            with self.assertRaisesRegex(WakeNotAccepted, 'admission_conflict'):
                await self._inject()
            self.assertEqual(len(self.snapshot_calls), 1)
            self.assertEqual(self.admissions, [])
        # Same event, shipped comparator: admitted. The comparator is the only moving part.
        self.assertTrue(await self._inject())
        self.assertEqual(len(self.admissions), 1)

    async def test_routing_owner_and_adapter_drift_still_refused(self):
        def origin(**changes):
            return lambda: setattr(self.entry, 'origin', replace_source(self.entry.origin, **changes))

        drifts = {
            'chat_id': origin(chat_id='43'),
            'chat_type': origin(chat_type='channel'),
            'user_id': origin(user_id='stranger'),
            'user_name': origin(user_name='stranger'),
            'thread_id': origin(thread_id='99'),
            'chat_name': origin(chat_name='other thread'),
            'scope_id': origin(scope_id='other-guild'),
            'parent_chat_id': origin(parent_chat_id='other-channel'),
            'profile': origin(profile='other-profile'),
            'prospective_thread_id': origin(prospective_thread_id='other-thread'),
            'owner': lambda: self.evt.__setitem__('parent_session_id', 'previous-conversation'),
            'adapter': lambda: setattr(self.runner, '_adapter_for_source', lambda source: object()),
        }
        for label, drift in drifts.items():
            with self.subTest(drift=label):
                self._reset_subject()
                drift()
                with self.assertRaisesRegex(WakeNotAccepted, 'admission_conflict'):
                    await self._inject()
                # Refused inside the snapshot: reached it once, admitted nothing.
                self.assertEqual(len(self.snapshot_calls), 1)
                self.assertEqual(self.admissions, [])
