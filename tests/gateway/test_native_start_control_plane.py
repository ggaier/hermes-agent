"""Native Discord-thread start control plane: real control socket, grant, store, admission.

Fake boundary: the Discord HTTP calls (thread creation, send, membership) and the
model turn.  Everything else is production: the private control-socket verb, the
scoped grant + immutable request digest, the durable idempotency record under the
isolated HERMES_HOME, and ``SessionAuthority.register/submit`` (canonical FIFO).
"""
from __future__ import annotations

import asyncio
import getpass
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.control_socket import GatewayControlServer, CONTROL_PROTOCOL_VERSION
from gateway.native_start import control_handlers, request_digest
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions

# One authoritative gateway loop per process: handlers marshal onto it (C9); a throwaway
# asyncio.run per dispatch would kill the FIFO drain task it schedules.
_MAIN_LOOP = asyncio.new_event_loop()
threading.Thread(target=_MAIN_LOOP.run_forever, daemon=True).start()

CARD = "card-alpha"
PARENT = "910000000000000002"
OPERATOR = {"platform": "discord", "user_id": "1563145615154876438", "guild_id": "900000000000000001"}
BRIEF = "Ship the native-thread start control plane. Start from this brief only."


class DiscordBoundary:
    """Fake Discord HTTP boundary (allowed); real adapter methods are exercised in prod."""

    def __init__(self):
        self.threads: dict[str, dict] = {}
        self.sent: list[dict] = []
        self.members: list[tuple[str, str]] = []
        self.participated: set[str] = set()
        self.fail_brief = False
        self._n = 0

    async def create_handoff_thread(self, parent_chat_id: str, name: str):
        self._n += 1
        tid = f"9200000000000{self._n:05d}"
        self.threads[tid] = {"parent": str(parent_chat_id), "name": name, "visibility": "public"}
        return tid

    async def add_thread_member(self, thread_id: str, user_id: str):
        self.members.append((thread_id, user_id))
        return True

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if self.fail_brief:
            return SimpleNamespace(success=False, error="injected")
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata or {}})
        return SimpleNamespace(success=True)

    @property
    def _threads(self):
        holder = self

        class Tracker:
            async def mark_async(self, thread_id):
                holder.participated.add(thread_id)
        return Tracker()


def _dispatch(server, verb, params, *, peer=getpass.getuser()):
    raw = json.dumps({"protocol": CONTROL_PROTOCOL_VERSION, "id": 1, "verb": verb,
                      "params": params}).encode()
    response = json.loads(server.handle_request_line(raw, peer_subject=peer).decode())
    result = response.get("result")
    return result if response.get("ok") and isinstance(result, dict) else response


@pytest.fixture
def plane(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    store = SessionStore(config=GatewayConfig(), sessions_dir=home / "sessions")
    store._db = db
    boundary = DiscordBoundary()

    async def turn(event):  # model/turn boundary only
        return "ok"

    runner = SimpleNamespace(
        _draining=False, session_store=store, profile_id="default",
        adapters={Platform.DISCORD: boundary}, hermes_home=home,
        _main_loop=None, _handle_message=turn, _cached_agent_for=lambda route: None,
    )
    authority = SessionAuthority(runner, profile_id="default", instance_id="test", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="test"))
    runner.session_authority = authority
    server = GatewayControlServer(home)
    server.private_handlers.update(control_handlers(runner, home, _MAIN_LOOP))
    yield SimpleNamespace(home=home, db=db, boundary=boundary, runner=runner,
                          authority=authority, server=server)
    db.close()


def _params(**over):
    params = {"profile": "default", "card_id": CARD, "idempotency_key": "attempt-1",
              "operator": dict(OPERATOR), "parent_channel_id": PARENT,
              "thread_name": "alpha", "brief": BRIEF}
    params.update(over)
    return params


def _grant(plane, **over):
    return _dispatch(plane.server, "native-start-grant", _params(**over))


def test_grant_is_scoped_to_the_immutable_request(plane):
    granted = _grant(plane)
    assert granted["ok"], granted
    digest = request_digest("default", CARD, PARENT, OPERATOR, BRIEF)
    assert granted["digest"] == digest
    assert (plane.home / "native_start" / "grants" / f"{digest}.json").is_file()
    # A different brief is a different authorization, not a reuse.
    changed = _grant(plane, brief=BRIEF + " altered")
    assert changed["digest"] != digest


def test_untrusted_peer_cannot_grant_or_start(plane):
    denied = _dispatch(plane.server, "native-start-grant", _params(), peer="someone-else")
    assert denied["ok"] is False and denied["code"] == "untrusted_peer"
    started = _dispatch(plane.server, "native-start-thread", _params(), peer="someone-else")
    assert started["ok"] is False and started["code"] == "untrusted_peer"
    assert plane.boundary.threads == {}


def test_start_creates_public_thread_brief_membership_and_admission(plane):
    _grant(plane)
    started = _dispatch(plane.server, "native-start-thread", _params())
    assert started["ok"], started
    tid = started["thread_id"]
    assert plane.boundary.threads[tid]["visibility"] == "public"
    assert plane.boundary.threads[tid]["parent"] == PARENT
    assert plane.boundary.members == [(tid, OPERATOR["user_id"])]
    assert plane.boundary.sent and BRIEF in plane.boundary.sent[0]["content"]
    assert tid in plane.boundary.participated
    rows = list_session_admissions(plane.db, session_id=started["session_id"])
    assert rows and rows[0]["request_id"] == "attempt-1"


def test_missing_grant_fails_closed_without_side_effects(plane):
    started = _dispatch(plane.server, "native-start-thread", _params())
    assert started["ok"] is False and started["code"] == "unauthorized"
    assert plane.boundary.threads == {}


def test_cross_card_and_cross_profile_fail_closed(plane):
    _grant(plane)
    other = _dispatch(plane.server, "native-start-thread", _params(card_id="card-beta"))
    assert other["code"] in {"unauthorized", "identity_mismatch"}
    cross = _dispatch(plane.server, "native-start-thread", _params(profile="other"))
    assert cross["code"] == "cross_profile"
    assert plane.boundary.threads == {}


def test_idempotent_retry_returns_same_thread_and_conflicts_on_changed_request(plane):
    _grant(plane)
    first = _dispatch(plane.server, "native-start-thread", _params())
    second = _dispatch(plane.server, "native-start-thread", _params())
    assert first["ok"] and second["ok"]
    assert first["thread_id"] == second["thread_id"]
    assert len(plane.boundary.threads) == 1
    _grant(plane, brief=BRIEF + " altered")
    conflict = _dispatch(plane.server, "native-start-thread", _params(brief=BRIEF + " altered"))
    assert conflict["ok"] is False and conflict["code"] == "idempotency_conflict"


def test_failed_attempt_keeps_its_thread_for_the_retry(plane):
    """The shipped adapter has no delete/archive primitive, so a failed attempt must
    persist its thread and the retry must adopt it instead of creating a second one."""
    _grant(plane)
    plane.boundary.fail_brief = True
    failed = _dispatch(plane.server, "native-start-thread", _params())
    assert failed["ok"] is False and failed["code"] == "transport_failed"
    records = list((plane.home / "native_start" / "runs").glob("*.json"))
    assert len(records) == 1
    pending = json.loads(records[0].read_text())
    assert pending["state"] == "pending" and pending["thread_id"]
    orphan = pending["thread_id"]
    plane.boundary.fail_brief = False
    recovered = _dispatch(plane.server, "native-start-thread", _params())
    assert recovered["ok"], recovered
    assert recovered["thread_id"] == orphan
    assert len(plane.boundary.threads) == 1


def test_identity_mismatch_is_not_authorized(plane):
    _grant(plane)
    other = dict(OPERATOR, user_id="999999999999999999")
    started = _dispatch(plane.server, "native-start-thread", _params(operator=other))
    assert started["ok"] is False and started["code"] == "unauthorized"
    assert plane.boundary.threads == {}


def test_concurrent_same_key_creates_exactly_one_thread(plane):
    from concurrent.futures import ThreadPoolExecutor

    _grant(plane)
    barrier = threading.Barrier(4)
    def call():
        barrier.wait(timeout=10)
        return _dispatch(plane.server, "native-start-thread", _params())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [f.result(timeout=60) for f in [pool.submit(call) for _ in range(4)]]
    assert all(r["ok"] for r in results), results
    assert len({r["thread_id"] for r in results}) == 1
    assert len(plane.boundary.threads) == 1


def test_restart_reopens_state_and_keeps_idempotency(plane, tmp_path):
    _grant(plane)
    first = _dispatch(plane.server, "native-start-thread", _params())
    assert first["ok"], first

    plane.db.close()
    db = SessionDB(plane.home / "state.db")
    store = SessionStore(config=GatewayConfig(), sessions_dir=plane.home / "sessions")
    store._db = db
    async def turn(event):
        return "ok"
    runner = SimpleNamespace(
        _draining=False, session_store=store, profile_id="default",
        adapters={Platform.DISCORD: plane.boundary}, hermes_home=plane.home,
        _main_loop=None, _handle_message=turn, _cached_agent_for=lambda route: None,
    )
    runner.session_authority = SessionAuthority(
        runner, profile_id="default", instance_id="restarted", db=db,
        epoch=begin_runtime_epoch(db, instance_id="restarted"))
    server = GatewayControlServer(plane.home)
    server.private_handlers.update(control_handlers(runner, plane.home, _MAIN_LOOP))
    try:
        again = _dispatch(server, "native-start-thread", _params())
        assert again["ok"], again
        assert again["thread_id"] == first["thread_id"]
        assert len(plane.boundary.threads) == 1
    finally:
        db.close()


# --- F3: required identity fields are validated, with no side effects -----------

_IDENTITY_FIELDS = ("profile", "card_id", "parent_channel_id",
                    "operator.user_id", "operator.guild_id", "brief")


def _without(params: dict, field: str, empty: bool) -> dict:
    """Copy of ``params`` with ``field`` removed (missing) or blanked (empty)."""
    import copy

    mutated = copy.deepcopy(params)
    if field.startswith("operator."):
        key = field.split(".", 1)[1]
        if empty:
            mutated["operator"][key] = ""
        else:
            mutated["operator"].pop(key)
    elif empty:
        mutated[field] = ""
    else:
        mutated.pop(field)
    return mutated


@pytest.mark.parametrize("empty", [False, True], ids=["missing", "empty"])
@pytest.mark.parametrize("field", _IDENTITY_FIELDS)
def test_identity_fields_are_required_on_both_verbs(plane, field, empty):
    """F3: every required identity field rejects missing and blank values as
    ``invalid_params`` — on both the grant and the start verb — before any durable
    record or Discord thread exists."""
    params = _without(_params(), field, empty)
    for verb in ("native-start-grant", "native-start-thread"):
        result = _dispatch(plane.server, verb, params)
        assert result["ok"] is False, (verb, result)
        assert result["code"] == "invalid_params", (verb, result)
    assert list((plane.home / "native_start" / "grants").glob("*.json")) == []
    assert list((plane.home / "native_start" / "runs").glob("*.json")) == []
    assert plane.boundary.threads == {}
