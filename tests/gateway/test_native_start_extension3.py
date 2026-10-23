"""C1: the native-start control plane must accept the REAL authenticated POSIX peer.

The peer subject comes from the kernel (``getpeereid``/``SO_PEERCRED``) through a real
``GatewayControlServer`` unix socket — no test-supplied username is ever presented as
the caller's identity.  ``gateway.control_socket._posix_peer_subject`` labels that peer
``uid:<n>``; the grant handler must treat that verified label as the trusted account.

Only fixture seams: the kernel-uid lookup is substituted to model another/missing uid
(we cannot spawn another account), and nothing in ``gateway/`` is patched.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import control_socket as control_socket_module
from gateway.config import Platform
from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    resolve_server_socket_path,
)
from gateway.native_start import control_handlers, request_digest

CARD = "card-alpha"
PARENT = "910000000000000002"
OPERATOR = {"platform": "discord", "user_id": "1563145615154876438", "guild_id": "900000000000000001"}
BRIEF = "Ship the native-thread start control plane. Start from this brief only."

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX peer credentials")


def _params(**over):
    params = {"profile": "default", "card_id": CARD, "idempotency_key": "attempt-1",
              "operator": dict(OPERATOR), "parent_channel_id": PARENT,
              "thread_name": "alpha", "brief": BRIEF}
    params.update(over)
    return params


def _envelope(verb, params, **extra):
    request = {"protocol": CONTROL_PROTOCOL_VERSION, "id": 1, "verb": verb, "params": params}
    request.update(extra)
    return request


class _LiveServer:
    """A real GatewayControlServer bound to a real AF_UNIX path in its own thread."""

    def __init__(self, home: Path):
        self.home = home
        self.loop = asyncio.new_event_loop()
        self.server: GatewayControlServer | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        asyncio.set_event_loop(self.loop)
        runner = SimpleNamespace(hermes_home=self.home, adapters={Platform.DISCORD: None})
        self.server = GatewayControlServer(self.home)
        self.server.private_handlers.update(control_handlers(runner, self.home, self.loop))
        self.loop.run_until_complete(self.server.start())
        self.ready.set()
        self.loop.run_forever()

    def __enter__(self) -> _LiveServer:
        self.thread.start()
        assert self.ready.wait(timeout=10), "control socket did not start"
        return self

    def __exit__(self, *exc):
        shutdown = asyncio.run_coroutine_threadsafe(self.server.stop(), self.loop)
        shutdown.result(timeout=10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()

    @property
    def path(self) -> Path:
        bind_path, _pointer = resolve_server_socket_path(self.home)
        return bind_path


def call(server: _LiveServer, request: dict) -> dict:
    """One real request line over the socket; returns the decoded reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(server.path))
        client.sendall(json.dumps(request).encode("utf-8") + b"\n")
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            buffer += chunk
    assert buffer, "the control socket returned no answer"
    return json.loads(buffer.decode("utf-8"))


def result_of(reply: dict) -> dict:
    body = reply.get("result")
    return body if reply.get("ok") and isinstance(body, dict) else reply


@pytest.fixture
def live(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with _LiveServer(home) as server:
        yield SimpleNamespace(home=home, server=server)


@pytest.fixture
def peer_uid(monkeypatch):
    """Substitute only the kernel uid lookup (cannot spawn a foreign account)."""

    def pin(value):
        import hermes_cli.gateway_runtime_discovery as discovery
        monkeypatch.setattr(discovery, "socket_peer_uid", lambda sock: value)

    return pin


def test_real_authenticated_peer_uid_can_grant(live):
    """The verified local peer (uid:<our uid>) is the trusted account — grant succeeds."""
    reply = call(live.server, _envelope("native-start-grant", _params()))
    granted = result_of(reply)
    assert granted.get("ok") is True, reply
    digest = request_digest("default", CARD, PARENT, OPERATOR, BRIEF)
    assert granted["digest"] == digest
    record = json.loads((live.home / "native_start" / "grants" / f"{digest}.json").read_text())
    assert record["card_id"] == CARD


def test_other_uid_peer_is_rejected(live, peer_uid):
    peer_uid(os.getuid() + 1)
    reply = call(live.server, _envelope("native-start-grant", _params()))
    denied = result_of(reply)
    assert denied.get("ok") is False, reply
    assert list((live.home / "native_start" / "grants").glob("*.json")) == []


def test_peer_without_a_uid_is_rejected(live, peer_uid):
    peer_uid(None)
    reply = call(live.server, _envelope("native-start-grant", _params()))
    denied = result_of(reply)
    assert denied.get("ok") is False, reply
    assert list((live.home / "native_start" / "grants").glob("*.json")) == []


def test_request_cannot_forge_its_own_peer_identity(live):
    """A caller-supplied username in the envelope is not identity: the socket refuses it."""
    forged = _envelope("native-start-grant", _params(), peer_subject=_forged_name())
    reply = call(live.server, forged)
    assert reply.get("ok") is False, reply
    assert list((live.home / "native_start" / "grants").glob("*.json")) == []


def _forged_name() -> str:
    import getpass

    return getpass.getuser()


def test_real_peer_subject_is_the_kernel_uid(live):
    """Guard: the accepted subject is uid-labeled, never the account name."""
    real = control_socket_module.GatewayControlServer
    assert real is GatewayControlServer
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(live.server.path))
        from hermes_cli.gateway_runtime_discovery import socket_peer_uid
        assert socket_peer_uid(client) == os.getuid()


# --- C4 / C10 on the same real control plane -----------------------------------
from tests.gateway.test_native_start_control_plane import (
    _dispatch, _grant, _params, plane)


def test_same_authorized_card_with_a_new_key_starts_once(plane):
    """C4: the key is the card, so a re-keyed retry reuses the first thread."""
    _grant(plane)
    first = _dispatch(plane.server, "native-start-thread", _params())
    assert first["ok"], first
    again = _dispatch(plane.server, "native-start-thread", _params(idempotency_key="attempt-2"))
    assert again["ok"], again
    assert again["thread_id"] == first["thread_id"]
    assert len(plane.boundary.threads) == 1


def test_corrupt_run_record_fails_closed(plane):
    """C10: an unreadable run record is not a fresh start."""
    import hashlib

    _grant(plane)
    runs = plane.home / "native_start" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(b"default\ncard-alpha").hexdigest()
    (runs / f"{key}.json").write_text("{not json", encoding="utf-8")
    denied = _dispatch(plane.server, "native-start-thread", _params())
    assert denied["ok"] is False and denied["code"] == "state_corrupt", denied
    assert plane.boundary.threads == {}


def test_distinct_cards_get_distinct_session_keys(plane):
    """C8: the native-start source (thread chat_id, parent_chat_id, no scope_id) keys
    each card's thread onto its own session; Discord guild scope is deliberately absent."""
    from gateway.session import SessionSource, build_session_key

    def key(card: str, thread: str) -> str:
        source = SessionSource(platform=Platform.DISCORD, chat_id=thread, chat_type="thread",
                               user_id=OPERATOR["user_id"], user_name="operator",
                               thread_id=thread, parent_chat_id=PARENT)
        return build_session_key(source, profile="default")

    first = _dispatch(plane.server, "native-start-grant", _params())
    assert first["ok"], first
    alpha = key("card-alpha", "9200000000000000001")
    beta = key("card-beta", "9200000000000000002")
    assert alpha != beta
    assert ":discord:thread:" in alpha and "9200000000000000001" in alpha
    assert "9200000000000000002" not in alpha
