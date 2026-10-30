"""B1: the real ``connect()`` entry distinguishes a refused proxy port and a network timeout from a
generic connect error, without ever reading exception text or echoing proxy credentials.

Positive/negative controls run through ``DiscordAdapter.connect`` (not the classifier in
isolation), so the code the gateway actually stamps into ``gateway_state.json`` is what is asserted.
"""
import errno
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp.client_reqrep import ConnectionKey

from gateway.config import PlatformConfig
from tests.gateway.test_discord_connect import FakeBot, _ensure_discord_mock

_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform
from plugins.platforms.discord.adapter import DiscordAdapter


def _connection_key():
    return ConnectionKey('proxy.invalid', 1080, False, False, None, None, None)


def _proxy_refused():
    return aiohttp.ClientProxyConnectionError(
        _connection_key(), ConnectionRefusedError(errno.ECONNREFUSED, 'Connection refused'))


def _adapter(monkeypatch):
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token='test-token', extra={'slash_commands': False}))
    monkeypatch.setattr('gateway.status.acquire_scoped_lock',
                        lambda scope, identity, metadata=None: (True, None))
    monkeypatch.setattr('gateway.status_scoped_locks.release_scoped_lock',
                        lambda scope, identity: None)
    monkeypatch.setattr(discord_platform.Intents, 'default', lambda: SimpleNamespace(
        message_content=False, dm_messages=False, guild_messages=False,
        members=False, voice_states=False))
    return adapter


def _fail_connect(monkeypatch, error):
    class BoomBot(FakeBot):
        async def start(self, token):
            raise error

    monkeypatch.setattr(discord_platform.commands, 'Bot', lambda **kwargs: BoomBot(
        intents=kwargs['intents'], proxy=kwargs.get('proxy'),
        allowed_mentions=kwargs.get('allowed_mentions')))


@pytest.mark.asyncio
async def test_proxy_refusal_is_its_own_retryable_code_without_credentials(monkeypatch):
    adapter = _adapter(monkeypatch)
    _fail_connect(monkeypatch, _proxy_refused())

    assert await adapter.connect() is False

    assert adapter.fatal_error_code == 'discord_proxy_refused'
    assert adapter.fatal_error_retryable is True
    # The diagnostic must not echo the proxy endpoint or credentials from the exception.
    assert 'proxy.invalid' not in adapter.fatal_error_message
    assert '1080' not in adapter.fatal_error_message


@pytest.mark.asyncio
async def test_network_timeout_is_not_a_local_ready_timeout(monkeypatch):
    adapter = _adapter(monkeypatch)
    _fail_connect(monkeypatch, aiohttp.ConnectionTimeoutError())

    assert await adapter.connect() is False

    assert adapter.fatal_error_code == 'discord_network_timeout'
    assert adapter.fatal_error_retryable is True


@pytest.mark.asyncio
async def test_local_ready_timeout_keeps_the_existing_code(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(discord_platform.commands, "Bot",
                        lambda **kwargs: FakeBot(intents=kwargs["intents"]))

    async def _timed_out(_ready_event, _bot_task, timeout):
        raise TimeoutError()

    monkeypatch.setattr(discord_platform, '_wait_for_ready_or_bot_exit', _timed_out)

    assert await adapter.connect() is False

    assert adapter.fatal_error_code == 'discord_connect_timeout'
    assert adapter.fatal_error_retryable is True


@pytest.mark.asyncio
async def test_generic_connection_refused_is_not_called_a_proxy(monkeypatch):
    adapter = _adapter(monkeypatch)
    _fail_connect(monkeypatch, ConnectionRefusedError(errno.ECONNREFUSED, 'Connection refused'))

    assert await adapter.connect() is False

    assert adapter.fatal_error_code == 'discord_connect_error'
    assert adapter.fatal_error_retryable is True


@pytest.mark.asyncio
async def test_auth_failure_stays_terminal(monkeypatch):
    class LoginFailure(Exception):
        pass

    adapter = _adapter(monkeypatch)
    _fail_connect(monkeypatch, LoginFailure('bad token'))

    assert await adapter.connect() is False

    assert adapter.fatal_error_code == 'discord_auth_error'
    assert adapter.fatal_error_retryable is False


@pytest.mark.asyncio
async def test_unknown_exception_stays_generic_and_retryable(monkeypatch):
    adapter = _adapter(monkeypatch)
    _fail_connect(monkeypatch, RuntimeError('something unexpected'))

    assert await adapter.connect() is False

    assert adapter.fatal_error_code == 'discord_connect_error'
    assert adapter.fatal_error_retryable is True
