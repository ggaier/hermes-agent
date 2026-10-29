"""Local real admission → TurnRunner → Discord reply after backend clock expiry.

The shared FakeView has no automatic timer. This verifies the backend monotonic
deadline, not discord.py ViewStore scheduling or production Discord delivery.
"""
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session_authority import initialize_session_authority
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter
from tests.waiting import async_wait_until


class _Channel:
    def __init__(self, channel_id, sent, edits):
        self.id, self.sent, self.edits = channel_id, sent, edits
        self.guild = None
        self.messages = {}

    async def send(self, content=None, **kwargs):
        message_id = 1000 + len(self.sent)

        async def edit(**changes):
            self.edits.append((self.id, changes))
            return message

        message = SimpleNamespace(id=message_id, edit=edit, embeds=[])
        self.messages[message_id] = message
        self.sent.append((self.id, content, kwargs))
        return message

    async def fetch_message(self, message_id):
        return self.messages[int(message_id)]

    def get_partial_message(self, message_id):
        return self.messages[int(message_id)]


def _rows(authority):
    return {row['request_id']: row for sid in authority.sessions
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)}


@pytest.mark.asyncio
@pytest.mark.parametrize('arrival', ['pending', 'after_terminal'])
async def test_backend_expiry_executes_ordinary_followon_in_same_thread(tmp_path, monkeypatch, arrival):
    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home
    from run_agent import AIAgent
    from tools import approval
    from tools.approval_gateway_wait import _await_gateway_decision

    home = Path(get_hermes_home())
    (home / 'config.yaml').write_text(
        'model:\n  default: fixture-model\n  provider: custom\n'
        '  base_url: http://127.0.0.1:9/v1\n  api_key: inert-test-key\n  context_length: 200000\n'
        'platform_toolsets:\n  discord: []\n'
        'gateway:\n  platforms:\n    discord:\n      skip_context_files: true\n'
        'display:\n  busy_input_mode: queue\n  tool_progress: false\n  thinking: false\n'
        'session_reset:\n  mode: none\n', encoding='utf-8')
    monkeypatch.setattr('gateway.run._hermes_home', home)
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', home / 'gateway_voice_mode.json')
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setenv('DISCORD_AUTO_THREAD', 'false')
    monkeypatch.setenv('DISCORD_REQUIRE_MENTION', 'false')
    monkeypatch.setattr('tools.approval_context._get_approval_timeout', lambda: 3)
    # Same external client/metadata seams as existing deterministic AIAgent fixtures.
    monkeypatch.setattr('agent.process_bootstrap.OpenAI', MagicMock())
    monkeypatch.setattr('agent.model_metadata.fetch_model_metadata', lambda *a, **k: {})
    calls, decisions, sent, edits = [], [], [], []

    def conversation(agent, message, **kwargs):
        calls.append(message)
        if len(calls) == 1:
            key = approval.get_current_session_key()
            started = time.monotonic()
            decision = _await_gateway_decision(key, approval._gateway_notify_cb(key), {
                'command': 'A2_INERT_GATE_PAYLOAD', 'description': 'test payload only',
                'pattern_key': 'a2-inert', 'pattern_keys': ['a2-inert'], 'session_key': key,
                'surface': 'discord'})
            decisions.append((decision, time.monotonic() - started))
            response = 'A2_FIRST_EXPIRED'
        else:
            response = 'A2_FOLLOWON_OK'
        messages = [*kwargs.get('conversation_history', []), {'role': 'user', 'content': message},
                    {'role': 'assistant', 'content': response}]
        return {'final_response': response, 'messages': messages, 'completed': True}

    monkeypatch.setattr(AIAgent, 'run_conversation', conversation)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='inert', extra={'reactions': False}))
    adapter.gateway_runner = runner
    adapter._text_batch_delay_seconds = 0
    channel = _Channel(555, sent, edits)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999), get_channel=lambda cid: channel,
                                      fetch_channel=AsyncMock(return_value=channel))
    # Discord typing is a network operation; all sends/edits still use the real adapter.
    monkeypatch.setattr(adapter, 'send_typing', AsyncMock())
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='a2-chain')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='444', chat_type='thread', thread_id='555', user_id='42')
    key = runner._session_key_for_source(source)
    first = None
    try:
        first = asyncio.create_task(adapter.handle_message(MessageEvent(text='first ordinary input', source=source,
                                                                        message_id='101')))
        await async_wait_until(lambda: authority.sessions and any(
            live.controls.pending for live in authority.sessions.values()), timeout=20,
            what='real TurnRunner registered approval')
        assert approval.list_gateway_approvals(key)
        assert len(calls) == 1
        followon = MessageEvent(text='report the follow-on marker please', source=source, message_id='102')
        if arrival == 'pending':
            await asyncio.wait_for(adapter.handle_message(followon), 10)
            assert _rows(authority)['101']['status'] == 'started'
            assert _rows(authority)['102']['status'] == 'queued'
            assert len(calls) == 1
        else:
            await asyncio.wait_for(first, 20)
            await async_wait_until(lambda: _rows(authority)['101']['status'] == 'terminal'
                                   and any(content == 'A2_FIRST_EXPIRED' for _, content, _ in sent)
                                   and not adapter._background_tasks, timeout=20,
                                   what='actual first admission terminal and reply delivered')
            assert _rows(authority)['101']['status'] == 'terminal'
            assert decisions and not approval.list_gateway_approvals(key)
            await asyncio.wait_for(adapter.handle_message(followon), 20)
        await asyncio.wait_for(first, 20)
        await async_wait_until(lambda: any(content == 'A2_FOLLOWON_OK' for _, content, _ in sent)
                               and not adapter._background_tasks, timeout=20, what='follow-on reply and guard release')
        assert [(cid, content) for cid, content, _ in sent if content == 'A2_FOLLOWON_OK'] == [(555, 'A2_FOLLOWON_OK')]
        assert len(calls) == 2
        assert decisions[0][0]['resolved'] is False and decisions[0][0]['choice'] is None
        assert decisions[0][1] >= 3, 'real monotonic deadline must elapse'
        assert {rid: row['status'] for rid, row in _rows(authority).items()} == {'101': 'terminal', '102': 'terminal'}
        assert not approval.list_gateway_approvals(key)
        for sid, live in authority.sessions.items():
            assert live.controls.snapshot(sid, None) == ()
        assert not adapter._active_sessions
        assert not runner._is_session_running(key)
        assert not authority.marker_adopters
        assert runner.session_store._entries[key].active_turn_token is None
        assert not authority.waiters
        assert edits, 'real timeout settlement should update the card'
    finally:
        if first is not None and not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        for live in authority.sessions.values():
            if live.task is not None and not live.task.done():
                live.task.cancel()
                await asyncio.gather(live.task, return_exceptions=True)
        authority.db.close()
