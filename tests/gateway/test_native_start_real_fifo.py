"""C3/C11 tracer: native-start brief actually executes and replies through the real chain.

Real components: GatewayRunner, the real DiscordAdapter (inbound handle_message and
outbound send/ledger), SessionAuthority (admission, FIFO drain, settlement delivery),
and the real GatewayRunner._handle_message turn chain.  Fixed at the two external
boundaries only: the Discord network client object (channels/threads/messages) and
the AIAgent model-generation class seam (``ctx.AIAgent`` / ``run_agent.AIAgent``).

Contracts asserted:
- C3: a granted native-start brief is executed by the real FIFO (queued -> terminal)
  and its reply lands in the created Discord thread.
- C11: a follow-up short sentence without any @mention enters through the real
  adapter inbound path, lands in the SAME session, executes and replies.
- second card: a second fixture card gets its own thread and its own session; the
  first card's history is not inherited.
"""
from __future__ import annotations

import asyncio
import json
import os

from types import SimpleNamespace

import pytest
import pytest_asyncio

from pathlib import Path

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.discord.adapter import DiscordAdapter

OPERATOR_ID = "1563145615154876438"
PARENT = "910000000000000002"
BRIEF = "Ship the native-thread start control plane. Start from this brief only."


class _FakeMessage:
    def __init__(self, n: int):
        self.id = n


class _FakeChannel:
    """Network object: one Discord channel or thread (send records, threads spawn)."""

    def __init__(self, client, cid: int):
        self.client = client
        self.id = cid
        self.sent: list[str] = []

    async def send(self, content=None, reference=None, **_kw):
        self.sent.append(str(content or ""))
        self.client.sent.append((str(self.id), str(content or "")))
        self.client._n += 1
        return _FakeMessage(self.client._n)

    async def create_thread(self, name=None, auto_archive_duration=None, reason=None):
        thread = _FakeChannel(self.client, self.client._next_thread_id())
        self.client.channels[thread.id] = thread
        self.client.created_threads.append((self.id, thread.id, str(name or "")))
        return thread

    async def add_user(self, user):
        """Real ``discord.Thread.add_user`` boundary: the HTTP call lives here."""
        self.client.members.append((str(self.id), str(getattr(user, "id", user))))


class _FakeDiscordClient:
    """The discord.py client stand-in: the adapter's whole HTTP surface, nothing else."""

    def __init__(self):
        self.channels: dict[int, _FakeChannel] = {}
        self.sent: list[tuple[str, str]] = []
        self.created_threads: list[tuple[int, int, str]] = []
        self.members: list[tuple[str, str]] = []
        self.user = type("BotUser", (), {"id": 998, "bot": True})()
        self._n = 0
        self._thread_n = 9200000000000000000

    def _next_thread_id(self) -> int:
        self._thread_n += 1
        return self._thread_n

    def get_channel(self, cid: int):
        return self.channels.get(int(cid))

    async def fetch_channel(self, cid: int):
        return self.channels.get(int(cid))


class _FixedOutputAIAgent:
    """The model-generation boundary only: accepts any construction the real turn
    performs, answers every conversation with a deterministic reply."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update({
            "max_iterations": kwargs.get("max_iterations") or 25,
            "session_prompt_tokens": 0, "session_completion_tokens": 0,
            "session_id": kwargs.get("session_id"), "model": kwargs.get("model"),
        })

    def __getattr__(self, name):
        # Config knobs the turn chain touches but the fixed boundary ignores.
        return None

    def run_conversation(self, user_message=None, task_id=None, **_kw):
        # agent_persisted=False: this fixed boundary persists nothing itself, so the real
        # gateway transcript writer records the user turn and the reply in the session
        # store (the same rows a non-persisting runtime produces).
        return {"final_response": f"reply to: {user_message}", "messages": [],
                "api_calls": 1, "completed": True, "failed": False,
                "agent_persisted": False}


@pytest_asyncio.fixture
async def gw(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", OPERATOR_ID)  # the real allowlist path
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", home / "gateway_voice_mode.json",
                        raising=False)  # class body pins the import-time home
    token = set_hermes_home_override(str(home))
    try:
        runner = GatewayRunner(GatewayConfig())
    finally:
        reset_hermes_home_override(token)
    # Align every later home read with the runner's OWN state.db location: profile
    # resolution inside GatewayRunner can relocate the db under a profile subdirectory,
    # and the transport-home/authority binding requires the two to agree.
    home = Path(runner.session_store._db.db_path).resolve().parent
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner.hermes_home = home

    client = _FakeDiscordClient()
    parent = _FakeChannel(client, int(PARENT))
    client.channels[parent.id] = parent
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token",
                                            typing_indicator=False))
    adapter._client = client  # network object only; every adapter method stays real
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}

    # Model config resolution is fixture config; the generation boundary is the class seam.
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime",
                        lambda **_k: ("fixture-model", {"api_key": "sk-fixture", "provider": "custom"}))
    import run_agent
    monkeypatch.setattr(run_agent, "AIAgent", _FixedOutputAIAgent)

    from gateway.session_authority import initialize_session_authority
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fifo-e2e")
    runner._wire_adapter_handlers(adapter, busy_text_mode="queue")

    from gateway.native_start import control_handlers
    loop = asyncio.get_running_loop()
    handlers = control_handlers(runner, home, loop)

    class _Gateway:
        pass
    g = _Gateway()
    g.home = home
    g.runner = runner
    g.authority = authority
    g.adapter = adapter
    g.client = client
    g.parent = parent
    g.handlers = handlers
    yield g
    authority.db.close()


def _grant_params(card: str, brief: str = BRIEF, **over) -> dict:
    params = {"profile": "default", "card_id": card, "idempotency_key": f"key-{card}",
              "operator": {"platform": "discord", "user_id": OPERATOR_ID,
                           "guild_id": "900000000000000001"},
              "parent_channel_id": PARENT, "thread_name": card, "brief": brief}
    params.update(over)
    return params


@pytest.mark.asyncio
async def test_native_brief_executes_replies_and_followup_continues(gw):
    from hermes_state_runtime import list_session_admissions
    from tests.waiting import async_wait_until

    # The socket executor dispatches handlers off the loop thread; mirror that here so
    # the marshal-onto-the-authoritative-loop path is the real one (not a same-thread deadlock).
    peer = f"uid:{os.getuid()}"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"],
                                    _grant_params("card-alpha"), peer)
    assert grant["ok"], grant
    started = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                      _grant_params("card-alpha"), peer)
    assert started["ok"], started
    thread_id = started["thread_id"]

    # C3: the brief executes on the real FIFO and the reply lands in the thread.
    # The brief itself is sent by native_start; the turn's REPLY is the fixed agent's
    # "reply to: ..." — wait for that specifically (receipt != execution).
    thread_channel = gw.adapter._client.channels[int(thread_id)]
    assert await async_wait_until(
        lambda: any("reply to:" in c for c in thread_channel.sent), timeout=30), gw.client.sent
    rows = list_session_admissions(gw.authority.db, session_id=started["session_id"],
                                   pending_only=False)
    mine = [r for r in rows if r["admission_id"] == started["admission_id"]]
    assert mine and mine[0]["status"] in {"completed", "terminal"}, mine

    # C11: a follow-up without any @mention enters through the real adapter inbound path.
    source = gw.adapter.build_source(chat_id=thread_id, chat_type="thread",
                                     user_id=OPERATOR_ID, user_name="operator",
                                     thread_id=thread_id, parent_chat_id=PARENT)
    event = MessageEvent(text="status update please", source=source, message_id="followup-1")
    await gw.adapter.handle_message(event)
    try:
        await async_wait_until(
            lambda: any(c.startswith("reply to:") and "status update please" in c
                        for c in thread_channel.sent),
            timeout=30)
    except AssertionError:
        pytest.fail(f"follow-up never executed; sent={gw.client.sent!r} "
                    f"rows={list_session_admissions(gw.authority.db, session_id=started['session_id'], pending_only=False)!r}")

    # Same session: the follow-up executed in the native-start session.
    followup_rows = list_session_admissions(gw.authority.db, session_id=started["session_id"],
                                            pending_only=False)
    assert followup_rows and followup_rows[-1]["status"] in {"completed", "terminal"}, followup_rows

    # Second card: its own thread and its own session, no inherited history.
    grant_b = await asyncio.to_thread(gw.handlers["native-start-grant"],
                                      _grant_params("card-beta"), peer)
    assert grant_b["ok"], grant_b
    started_b = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                        _grant_params("card-beta"), peer)
    assert started_b["ok"], started_b
    assert started_b["thread_id"] != thread_id
    assert started_b["session_id"] != started["session_id"]


class _GateThread:
    """The adapter-accepted stand-in for a discord.Thread (same seam the Discord
    admission tests use: a plain object the isinstance gate accepts)."""

    def __init__(self, cid: int, parent_id=None):
        self.id = cid
        self.name = f"thread-{cid}"
        self.parent_id = parent_id
        self.guild = None
        self.topic = None

    def history(self, *, limit, before, after=None, oldest_first=None):
        async def _iter():
            return
            yield
        return _iter()


def _gate_message(thread, content: str, n: int):
    from datetime import datetime, UTC
    import discord

    author = type("Author", (), {"id": int(OPERATOR_ID), "display_name": "operator",
                                 "name": "operator", "bot": False})()
    return SimpleNamespace(
        id=n, content=content, mentions=[], attachments=[], reference=None,
        created_at=datetime.now(UTC), channel=thread, author=author,
        type=discord.MessageType.default,
    )


@pytest.mark.asyncio
async def test_untracked_thread_requires_mention_then_native_start_admits(gw, monkeypatch):
    """C11/CP04 gate contract, on the real dispatch path (require_mention default true).

    CP04 is asserted as four separate facts: (a) the follow-up message carries no
    mention, (b) it is admitted, (c) it executes and replies in the SAME session, and
    (d) its admission row is terminal. Membership in the thread is a different fact and
    is not used as evidence here.

    F6: the allowlist is no longer fixture-assigned. It is resolved through the real
    adapter resolver (``_get_allowed_users()``, the same call ``connect()`` makes) from
    ``DISCORD_ALLOWED_USERS``; only the network client object is fixed. ``connect()``
    itself is not called because it would log in to Discord.
    """
    import json as _json

    import plugins.platforms.discord.adapter as discord_platform
    from hermes_state_runtime import list_session_admissions
    from tests.waiting import async_wait_until

    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", _GateThread, raising=False)
    gw.adapter._ready_event.set()
    gw.adapter._allowed_user_ids = gw.adapter._get_allowed_users()  # real resolver, env-driven
    assert gw.adapter._allowed_user_ids == {OPERATOR_ID}, gw.adapter._allowed_user_ids
    assert gw.adapter._discord_require_mention() is True  # default, not disabled here

    peer = f"uid:{os.getuid()}"
    thread = _GateThread(9200000000000000001, parent_id=int(PARENT))

    # CP04 (a): the follow-up is genuinely unmentioned.
    followup = _gate_message(thread, "status update please", 602)
    assert followup.mentions == [] and followup.reference is None
    assert gw.adapter._self_is_explicitly_mentioned(followup) is False

    # (a) the thread is not registered: a no-@ message is dropped by the real mention gate.
    before = len(list_session_admissions(gw.authority.db, session_id=gw.authority.sessions
                                         and next(iter(gw.authority.sessions)) or "none",
                                         pending_only=False)) if gw.authority.sessions else 0
    dropped = await gw.adapter._dispatch_discord_message(_gate_message(thread, "status update please", 601))
    assert dropped is False
    total = sum(len(list_session_admissions(gw.authority.db, session_id=sid, pending_only=False))
                for sid in list(gw.authority.sessions))
    assert total == before, "a mention-gated drop must not create any admission"

    # (b) native start creates + registers the thread; the same no-@ message is now admitted,
    # lands in the SAME session, executes and replies.
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"], _grant_params("card-alpha"), peer)
    assert grant["ok"], grant
    started = await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params("card-alpha"), peer)
    assert started["ok"], started
    assert started["thread_id"] == str(thread.id)

    thread_channel = gw.adapter._client.channels[int(thread.id)]
    # The brief turn must settle first: a follow-up while the session is running takes the
    # busy path, and the gate contract under test is the idle no-mention continuation.
    assert await async_wait_until(
        lambda: any(c.startswith("reply to:") for c in thread_channel.sent), timeout=30), gw.client.sent

    admitted = await gw.adapter._dispatch_discord_message(followup)
    assert admitted is True  # CP04 (b), no mention, on the real admission path
    assert await async_wait_until(
        lambda: any(c.startswith("reply to:") and "status update please" in c
                    for c in thread_channel.sent), timeout=30), gw.client.sent
    rows = list_session_admissions(gw.authority.db, session_id=started["session_id"], pending_only=False)
    assert [r for r in rows if r["request_id"] == "key-card-alpha"]  # brief admission
    followups = [r for r in rows if r["request_id"] not in {"key-card-alpha"}]
    assert followups, rows  # CP04 (c): the no-@ follow-up admitted in the SAME session
    assert all(r["status"] in {"completed", "terminal"} for r in followups), followups  # CP04 (d)


@pytest.mark.asyncio
async def test_drain_refused_start_keeps_pending_thread_and_retry_adopts(gw):
    """C7: a real admission refusal (runtime draining) must leave no success record,
    persist the thread as pending, and the retry must adopt it — no orphan, no second thread."""
    peer = f"uid:{os.getuid()}"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"], _grant_params("card-gamma"), peer)
    assert grant["ok"], grant

    gw.runner._draining = True  # the real SessionAuthority refuses admission while draining
    with pytest.raises(Exception):
        await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params("card-gamma"), peer)
    gw.runner._draining = False

    records = list((gw.home / "native_start" / "runs").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["state"] == "pending" and record["thread_id"]
    refused_thread = record["thread_id"]
    assert "result" not in record

    retry = await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params("card-gamma"), peer)
    assert retry["ok"], retry
    assert retry["thread_id"] == refused_thread
    assert len(gw.client.created_threads) == 1  # one thread, no orphan, no second creation


@pytest.mark.asyncio
async def test_digest_change_and_cross_profile_fail_closed(gw):
    """Negative controls on the real handlers: a changed brief under the same card is a
    conflict, and another profile is refused."""
    peer = f"uid:{os.getuid()}"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"], _grant_params("card-delta"), peer)
    assert grant["ok"], grant
    started = await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params("card-delta"), peer)
    assert started["ok"], started

    await asyncio.to_thread(gw.handlers["native-start-grant"],
                            _grant_params("card-delta", brief=BRIEF + " altered"), peer)
    conflict = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                       _grant_params("card-delta", brief=BRIEF + " altered"), peer)
    assert conflict["ok"] is False and conflict["code"] == "idempotency_conflict", conflict

    cross = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                    _grant_params("card-delta", profile="other"), peer)
    assert cross["ok"] is False and cross["code"] == "cross_profile", cross


@pytest.mark.asyncio
async def test_rekeyed_replay_returns_same_thread_and_session(gw):
    peer = f"uid:{os.getuid()}"
    await asyncio.to_thread(gw.handlers["native-start-grant"], _grant_params("card-eps"), peer)
    first = await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params("card-eps"), peer)
    assert first["ok"], first
    again = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                    _grant_params("card-eps", idempotency_key="key-eps-2"), peer)
    assert again["ok"], again
    assert again["thread_id"] == first["thread_id"] and again["session_id"] == first["session_id"]
    assert len(gw.client.created_threads) == 1


@pytest.mark.asyncio
async def test_c6_bootstrap_control_socket_falls_back_to_the_profile_home(gw, monkeypatch, tmp_path):
    """C6 runtime branch: with no runner.hermes_home the bootstrap must resolve the profile
    home through get_hermes_home() (the import that previously raised NameError)."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.run_bootstrap import _start_gateway_start_control_socket

    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", gw.home / "gateway_voice_mode.json",
                        raising=False)
    # Runtime state the real bootstrap installs before the control socket starts.
    gw.runner.session_runtime_descriptor = {
        "instance_id": "c6-probe", "runtime_protocol": 1, "state": "starting",
        "capabilities": [], "served_profiles": []}
    gw.runner.session_ticket_store = None  # pre-bootstrap value; no ticket verb is exercised
    monkeypatch.setattr(gw.runner, "hermes_home", None)  # the C6 condition: no runner home
    token = set_hermes_home_override(str(gw.home))
    # The control SERVER binds at the process home (env-empty -> platform default); in
    # production that directory exists, so the isolated sandbox provides it too. The
    # native-start HANDLERS are the fallback under test: they must use the profile home.
    import hermes_constants
    (tmp_path / hermes_constants._get_platform_default_hermes_home().name).mkdir(exist_ok=True)
    server = None
    try:
        server = await _start_gateway_start_control_socket(gw.runner)
        assert server is not None, "the control socket must start (no NameError on the fallback)"
        # The native-start handlers were wired with the PROFILE home, not a stale launch home.
        peer = f"uid:{os.getuid()}"
        granted = server.private_handlers["native-start-grant"](_grant_params("card-c6"), peer)
        assert granted["ok"], granted
        digest = granted["digest"]
        assert (gw.home / "native_start" / "grants" / f"{digest}.json").is_file()
    finally:
        reset_hermes_home_override(token)
        if server is not None:
            await server.stop()


@pytest.mark.asyncio
async def test_beta_thread_followup_lands_in_beta_session(gw, monkeypatch):
    """A no-@ follow-up in the SECOND card's thread must execute in the second card's
    session, never in the first card's.

    F5: isolation is asserted on the real stored transcript, not only on ids: the beta
    session's messages contain no alpha brief and no alpha reply, and (positive control)
    the alpha transcript contains both. Messages are read back with the store's own
    ``get_messages`` API — the same database the turn wrote to.

    F6: the allowlist comes from ``DISCORD_ALLOWED_USERS`` through the adapter's real
    ``_get_allowed_users()`` resolver (what ``connect()`` calls); no policy method is
    patched and ``connect()`` is not called (it would log in to Discord).
    """
    import plugins.platforms.discord.adapter as discord_platform
    from hermes_state_runtime import list_session_admissions
    from tests.waiting import async_wait_until

    monkeypatch.delenv("DISCORD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("DISCORD_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", _GateThread, raising=False)
    gw.adapter._ready_event.set()
    gw.adapter._allowed_user_ids = gw.adapter._get_allowed_users()  # real resolver, env-driven
    assert gw.adapter._allowed_user_ids == {OPERATOR_ID}

    peer = f"uid:{os.getuid()}"
    alpha_brief = "ALPHA-CARD: " + BRIEF
    beta_brief = "BETA-CARD: " + BRIEF
    starteds = {}
    for card, brief in (("card-one", alpha_brief), ("card-two", beta_brief)):
        grant = await asyncio.to_thread(gw.handlers["native-start-grant"],
                                        _grant_params(card, brief=brief), peer)
        assert grant["ok"], grant
        started = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                          _grant_params(card, brief=brief), peer)
        assert started["ok"], started
        starteds[card] = started

    alpha = starteds["card-one"]
    beta = starteds["card-two"]
    beta_thread = _GateThread(int(beta["thread_id"]), parent_id=int(PARENT))

    channel = gw.adapter._client.channels[int(beta_thread.id)]
    assert await async_wait_until(lambda: channel.sent, timeout=30)
    admitted = await gw.adapter._dispatch_discord_message(_gate_message(beta_thread, "mine please", 701))
    assert admitted is True
    assert await async_wait_until(
        lambda: any(c.startswith("reply to:") and "mine please" in c for c in channel.sent),
        timeout=30), gw.client.sent
    alpha_rows = list_session_admissions(gw.authority.db, session_id=alpha["session_id"],
                                         pending_only=False)
    assert all(r["request_id"] not in {"key-card-two", "701"} for r in alpha_rows), alpha_rows
    beta_rows = list_session_admissions(gw.authority.db, session_id=beta["session_id"],
                                        pending_only=False)
    followups = [r for r in beta_rows if r["request_id"] not in {"key-card-one", "key-card-two"}]
    assert followups, beta_rows  # the beta follow-up executed in the BETA session
    assert all(r["status"] in {"completed", "terminal"} for r in followups)

    # F5: transcript-level isolation from the real store the turn wrote to.
    def rows_for(session_id: str) -> list[dict]:
        return gw.authority.db.get_messages(session_id)

    def text_of(rows: list[dict]) -> str:
        return "\n".join(str(row.get("content") or "") for row in rows)

    alpha_text = text_of(rows_for(alpha["session_id"]))
    beta_text = text_of(rows_for(beta["session_id"]))
    # Positive control: the alpha transcript holds the brief and the alpha reply.
    assert alpha_brief in alpha_text, alpha_text
    assert any(row.get("role") == "assistant" and "reply to:" in text_of([row])
               and alpha_brief in text_of([row]) for row in rows_for(alpha["session_id"])), alpha_text
    assert beta_brief in beta_text, beta_text
    # The beta session inherited neither alpha's brief nor alpha's turn text.
    assert alpha_brief not in beta_text, beta_text
    assert not any("reply to:" in text_of([row]) and alpha_brief in text_of([row])
                   for row in rows_for(beta["session_id"])), beta_text


@pytest.mark.asyncio
async def test_child_process_reopen_reuses_thread_and_session(gw, tmp_path):
    """Restart recovery, truly cross-process: a NEW python process reopens the same
    HERMES_HOME and the same card's start returns the original thread and session."""
    import subprocess
    import sys

    peer = f"uid:{os.getuid()}"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"], _grant_params("card-restart"), peer)
    assert grant["ok"], grant
    first = await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params("card-restart"), peer)
    assert first["ok"], first

    script = tmp_path / "reopen_child.py"
    repo = Path(__file__).resolve().parents[2]
    script.write_text(f'''
import asyncio, json, os, sys
sys.path.insert(0, {str(repo)!r})
from pathlib import Path
from hermes_constants import set_hermes_home_override
set_hermes_home_override(os.environ["HERMES_HOME"])
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.native_start import control_handlers

async def main():
    runner = GatewayRunner(GatewayConfig())
    await initialize_session_authority(runner, profile_id="default", instance_id="child-restart")
    loop = asyncio.get_running_loop()
    handlers = control_handlers(runner, Path(os.environ["HERMES_HOME"]), loop)
    params = json.loads(sys.argv[1])
    result = await asyncio.to_thread(
        handlers["native-start-thread"], params, f"uid:{os.getuid()}")
    print(json.dumps(result))

asyncio.run(main())
''')
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path),
           "HERMES_HOME": str(gw.home), "TMPDIR": str(tmp_path)}
    child = await asyncio.to_thread(
        subprocess.run, [sys.executable, str(script), json.dumps(_grant_params("card-restart"))],
        check=False, env=env, cwd=str(repo), capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=240)
    assert child.returncode == 0, child.stderr[-2000:]
    reopened = json.loads(child.stdout.strip().splitlines()[-1])
    assert reopened["ok"] is True, reopened
    assert reopened["thread_id"] == first["thread_id"]
    assert reopened["session_id"] == first["session_id"]


# --- F1: a killed mid-start attempt is reclaimed, not wedged --------------------

_KILLED_THREAD = "9300000000000000001"


@pytest.mark.asyncio
async def test_sigkilled_start_is_reclaimed_and_adopts_the_thread(gw, tmp_path, monkeypatch):
    """F1: a real child process creates the thread and is SIGKILLed between create and
    submit. The next start must reclaim the dead ``starting`` reservation, adopt the
    thread the child persisted, and complete — not wait 60 s and report ``in_progress``.

    Evidence that the kill landed inside the window: the child prints RESERVED from
    ``send`` (after ``_reserve_thread`` wrote the thread id) and its exit code is -9.
    """
    import hashlib
    import json as _json
    import sys
    import time

    # The gateway test session mocks discord; give the adapter's Snowflake a real id so
    # the member-add boundary records the operator id it was actually given.
    import plugins.platforms.discord.adapter as discord_platform
    monkeypatch.setattr(discord_platform.discord, "Object",
                        lambda *, id: SimpleNamespace(id=id), raising=False)

    peer = f"uid:{os.getuid()}"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"],
                                    _grant_params("card-killed"), peer)
    assert grant["ok"], grant

    # The child's boundary always creates this deterministic id; give the parent's
    # adapter the same channel so the adopted thread is a real send target.
    gw.client.channels[int(_KILLED_THREAD)] = _FakeChannel(gw.client, int(_KILLED_THREAD))
    created_log = gw.home / "created-threads.json"

    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "killed_child.py"
    script.write_text(f'''
import asyncio, json, os, sys
sys.path.insert(0, {str(repo)!r})
from pathlib import Path
from types import SimpleNamespace
from gateway.config import Platform
from gateway.native_start import start_native_thread

THREAD_ID = {_KILLED_THREAD!r}
HOME = Path(os.environ["HERMES_HOME"])

class Tracker:
    async def mark_async(self, thread_id):
        pass

class Boundary:
    _threads = Tracker()
    async def create_handoff_thread(self, parent, name):
        (HOME / "created-threads.json").write_text(json.dumps([THREAD_ID]))
        return THREAD_ID
    async def add_thread_member(self, thread_id, user_id):
        return True
    async def send(self, *args, **kwargs):
        # The thread exists and its id is already persisted: the kill lands here,
        # after create and before register/submit.
        print("RESERVED", flush=True)
        while True:
            await asyncio.sleep(0.5)

class Authority:
    profile_id = "default"
    def register(self, source):
        raise RuntimeError("unreachable: killed before admission")
    async def submit(self, *args, **kwargs):
        raise RuntimeError("unreachable: killed before admission")

async def main():
    runner = SimpleNamespace(session_authority=Authority(), hermes_home=HOME,
                             adapters={{Platform.DISCORD: Boundary()}})
    print(json.dumps(await start_native_thread(
        runner, json.loads(sys.argv[1]), f"uid:{{os.getuid()}}")), flush=True)

asyncio.run(main())
''')
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path),
           "HERMES_HOME": str(gw.home), "TMPDIR": str(tmp_path)}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script), _json.dumps(_grant_params("card-killed")),
        env=env, cwd=str(repo), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    run_key = hashlib.sha256(b"default\ncard-killed").hexdigest()
    run_file = gw.home / "native_start" / "runs" / f"{run_key}.json"
    try:
        deadline = time.monotonic() + 90
        record = None
        while time.monotonic() < deadline:
            try:
                record = _json.loads(run_file.read_text())
            except (OSError, ValueError):
                record = None
            if (record or {}).get("state") == "starting" and \
                    record.get("thread_id") == _KILLED_THREAD:
                break
            await asyncio.sleep(0.05)
        assert record and record.get("state") == "starting" and \
            record.get("thread_id") == _KILLED_THREAD, (record, proc.returncode)
        assert record.get("pid") == proc.pid, record
        proc.kill()  # SIGKILL: no cleanup, no finally, no record rewrite
        await proc.wait()
        returncode = proc.returncode
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        child_log = ((await proc.stdout.read() if proc.stdout else b"")
                     + (await proc.stderr.read() if proc.stderr else b"")).decode()
    assert returncode == -9, (returncode, child_log)
    assert "RESERVED" in child_log, child_log

    # The dead attempt's reservation is reclaimed and its thread adopted.
    recovered = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                        _grant_params("card-killed"), peer)
    assert recovered["ok"], recovered
    assert recovered["thread_id"] == _KILLED_THREAD
    assert _json.loads(created_log.read_text()) == [_KILLED_THREAD]  # one thread, child's
    assert gw.client.created_threads == []  # parent created no second thread
    assert (str(_KILLED_THREAD), OPERATOR_ID) in gw.client.members  # adopted thread is visible

    # A later start returns the completed result, never ``in_progress``.
    again = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                    _grant_params("card-killed", idempotency_key="key-second"), peer)
    assert again["ok"] is True, again
    assert again["thread_id"] == _KILLED_THREAD
    assert again.get("session_id") == recovered["session_id"]
    final = _json.loads(run_file.read_text())
    assert final["state"] == "complete" and isinstance(final.get("result"), dict), final


# --- F2: the receipt.status gate refuses an unknown/terminal receipt -----------

class _ReceiptStatusSeam:
    """Wraps the real authority: the real admission still runs, then the receipt is
    replaced with one carrying an unknown/terminal status. Everything else (register,
    thread create, member add, send, _hold) is the production chain."""

    def __init__(self, authority, status):
        self._authority = authority
        self._status = status

    def __getattr__(self, name):
        return getattr(self._authority, name)

    async def submit(self, principal, submission):
        receipt = await self._authority.submit(principal, submission)
        return SimpleNamespace(status=self._status, admission_id=receipt.admission_id)


@pytest.mark.parametrize("status", ["terminal", "refused", None], ids=["terminal", "refused", "missing"])
@pytest.mark.asyncio
async def test_receipt_status_gate_refuses_and_retry_adopts(gw, status):
    """F2: a receipt whose status is not queued/started must be reported as
    ``admission_refused`` with a pending, resumable record — never a success record —
    and the retry must adopt the same thread instead of creating another one.

    Seam: only ``SessionAuthority.submit``'s returned receipt is substituted (after the
    real admission). Thread creation, member add, send and ``_hold`` are real.
    """
    import json as _json

    peer = f"uid:{os.getuid()}"
    card = "card-receipt"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"], _grant_params(card), peer)
    assert grant["ok"], grant

    real_authority = gw.runner.session_authority
    gw.runner.session_authority = _ReceiptStatusSeam(real_authority, status)
    try:
        refused = await asyncio.to_thread(gw.handlers["native-start-thread"],
                                          _grant_params(card), peer)
    finally:
        gw.runner.session_authority = real_authority
    assert refused["ok"] is False, refused
    assert refused["code"] == "admission_refused", refused

    records = list((gw.home / "native_start" / "runs").glob("*.json"))
    assert len(records) == 1
    record = _json.loads(records[0].read_text())
    assert record["state"] == "pending", record
    assert record["thread_id"], record
    assert "result" not in record, record

    retry = await asyncio.to_thread(gw.handlers["native-start-thread"], _grant_params(card), peer)
    assert retry["ok"], retry
    assert retry["thread_id"] == record["thread_id"]
    assert len(gw.client.created_threads) == 1  # adopted, never a second thread


# --- CP03: the real Discord adapter member primitive on a real private thread ---

class _ThreadHTTP:
    """The discord.py HTTP client boundary for one Thread object, nothing more."""

    def __init__(self):
        self.added: list[tuple[int, int]] = []

    async def add_user_to_thread(self, thread_id, user_id):
        self.added.append((int(thread_id), int(user_id)))

    async def get_thread_members(self, thread_id):
        return [
            {"id": str(thread_id), "user_id": str(uid),
             "join_timestamp": "2026-01-01T00:00:00+00:00", "flags": 0}
            for tid, uid in self.added if int(tid) == int(thread_id)
        ]


@pytest.mark.asyncio
async def test_add_thread_member_on_a_real_private_discord_thread(gw, monkeypatch):
    """CP03: ``DiscordAdapter.add_thread_member`` resolves the real thread and calls the
    real ``discord.Thread.add_user`` on a real ``discord.Thread`` built from a type=12
    private-thread payload; the member list is read back with ``fetch_members``.

    Only the HTTP client inside the Thread's connection state is faked. The gateway test
    session installs a discord mock in ``sys.modules`` (tests/gateway/conftest.py), so
    the real library is imported under a cleared module cache for this test and the
    session mock is restored afterwards; the adapter's module reference is pointed at
    the real library for the duration of the call.
    """
    import importlib
    import sys

    import plugins.platforms.discord.adapter as discord_platform

    saved = {k: v for k, v in list(sys.modules.items())
             if k == "discord" or k.startswith("discord.")}
    for key in saved:
        sys.modules.pop(key, None)
    try:
        real_discord = importlib.import_module("discord")
        monkeypatch.setattr(discord_platform, "discord", real_discord)

        http = _ThreadHTTP()
        data = {
            "id": _KILLED_THREAD, "type": 12, "guild_id": "900000000000000001",
            "parent_id": PARENT, "owner_id": OPERATOR_ID, "name": "native start",
            "thread_metadata": {"archived": False, "auto_archive_duration": 1440,
                                "archive_timestamp": "2026-01-01T00:00:00+00:00",
                                "locked": False, "invitable": True,
                                "create_timestamp": "2026-01-01T00:00:00+00:00"},
            "message_count": 0, "member_count": 1, "rate_limit_per_user": 0,
            "flags": 0, "last_message_id": None,
        }
        thread = real_discord.Thread(
            guild=SimpleNamespace(id=900000000000000001),
            state=SimpleNamespace(http=http), data=data,
        )
        assert thread.type is real_discord.ChannelType.private_thread
        gw.client.channels[int(thread.id)] = thread  # resolved through the real _resolve_channel

        assert await gw.adapter.add_thread_member(str(thread.id), OPERATOR_ID) is True
        assert http.added == [(int(thread.id), int(OPERATOR_ID))]  # add_user called with the operator
        members = await thread.fetch_members()
        assert int(OPERATOR_ID) in {m.id for m in members}  # visible to the operator now

        # Unresolvable target: an explicit False, never a silent success.
        assert await gw.adapter.add_thread_member("9300000000000000999", OPERATOR_ID) is False
    finally:
        for key in [k for k in sys.modules if k == "discord" or k.startswith("discord.")]:
            sys.modules.pop(key, None)
        sys.modules.update(saved)


# --- N1: a reclaimed thread survives a missing adapter --------------------------

_N1_THREAD = "9300000000000000123"


def _run_file(gw, card: str) -> Path:
    import hashlib

    key = hashlib.sha256(f"default\n{card}".encode()).hexdigest()
    return gw.home / "native_start" / "runs" / f"{key}.json"


@pytest.mark.asyncio
async def test_reclaimed_thread_survives_missing_adapter_and_is_adopted(gw):
    """N1 regression: a reclaimed ``starting`` record that already carries a ``thread_id``
    must not be deleted when the Discord adapter is gone.

    Baseline (6895e85f) deleted it with ``_release`` (the reclaim had just written this
    attempt's id, so the attempt guard matched), losing the only durable reference to a
    real Discord thread; the next start then created a SECOND thread for the same card.
    Here the stale record is a dead-pid ``starting`` record (the crash-recovery shape the
    F1 test produces), the adapter is absent for the first call, and the retry with the
    adapter back must adopt that same thread with zero creations by the parent.
    """
    import sys
    import time

    from gateway.native_start import _pid_alive, request_digest

    params = _grant_params("card-orphan")
    peer = f"uid:{os.getuid()}"
    grant = await asyncio.to_thread(gw.handlers["native-start-grant"], params, peer)
    assert grant["ok"], grant

    # A real process that has exited: the liveness probe is the production one.
    reaped = await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
    dead_pid = reaped.pid
    assert await reaped.wait() == 0
    assert _pid_alive(dead_pid) is False

    digest = request_digest("default", "card-orphan", PARENT, params["operator"], BRIEF)
    run_file = _run_file(gw, "card-orphan")
    run_file.parent.mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps({
        "state": "starting", "digest": digest, "attempt": "dead:attempt",
        "idempotency_key": "key-card-orphan", "pid": reaped.pid,
        "started_at": time.time() - 3600, "thread_id": _N1_THREAD,
    }))

    # The adapter is gone: no Discord connection for this attempt.
    gw.runner.adapters = {}
    missing = await asyncio.to_thread(gw.handlers["native-start-thread"], params, peer)
    assert missing["ok"] is False and missing["code"] == "unavailable", missing

    record = json.loads(run_file.read_text())
    assert record["state"] == "pending", record
    assert record["thread_id"] == _N1_THREAD, record
    assert "result" not in record, record

    # The adapter is back: the retry adopts the persisted thread, never creates another.
    gw.client.channels[int(_N1_THREAD)] = _FakeChannel(gw.client, int(_N1_THREAD))
    gw.runner.adapters = {Platform.DISCORD: gw.adapter}
    resumed = await asyncio.to_thread(gw.handlers["native-start-thread"], params, peer)
    assert resumed["ok"], resumed
    assert resumed["thread_id"] == _N1_THREAD, resumed
    assert gw.client.created_threads == [], gw.client.created_threads


# --- N2: a live owner is never displaced ---------------------------------------

def _write_starting(gw, card: str, digest: str, attempt: str, pid, started_at: float) -> Path:
    path = _run_file(gw, card)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "state": "starting", "digest": digest, "attempt": attempt,
        "idempotency_key": f"key-{card}", "pid": pid, "started_at": started_at,
    }))
    return path


@pytest.mark.asyncio
async def test_expired_lease_never_reclaims_a_live_owner(gw):
    """N2: an expired ``started_at`` with a LIVE pid is not reclaimable.

    The start coroutine is not bounded by the 60 s control-socket deadline, so an attempt
    stuck in adapter I/O can legitimately outlive any lease. A second caller must keep
    seeing ``busy`` (and the handler must keep waiting), never take the record over.
    """
    import time

    from gateway.native_start import _claim, _starting_owner_alive, request_digest, start_native_thread

    card = "card-live"
    params = _grant_params(card)
    peer = f"uid:{os.getuid()}"
    assert (await asyncio.to_thread(gw.handlers["native-start-grant"], params, peer))["ok"]

    digest = request_digest("default", card, PARENT, params["operator"], BRIEF)
    run_file = _write_starting(gw, card, digest, "owner:live", os.getpid(),
                               time.time() - 3600)
    record = json.loads(run_file.read_text())

    # Real liveness probe: our own pid is alive although the lease is long gone.
    assert _starting_owner_alive(record) is True

    paths = {"run": run_file,
             "grant": gw.home / "native_start" / "grants" / f"{digest}.json",
             "card_grant": gw.home / "native_start" / "grants" / "unused.json"}
    state, _ = _claim(gw.home, paths, digest, "second:attempt", "key-card-live")
    assert state == "busy", state
    assert json.loads(run_file.read_text())["attempt"] == "owner:live"

    # The caller itself: still waiting, no thread created, record untouched.
    task = asyncio.create_task(start_native_thread(gw.runner, params, peer))
    await asyncio.sleep(0.3)
    assert not task.done(), "the second caller must not take over a live owner"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads(run_file.read_text())["attempt"] == "owner:live"
    assert gw.client.created_threads == []


def test_displaced_attempt_cannot_overwrite_the_new_owner(gw):
    """N2 defence in depth: ``_reserve_thread``/``_hold``/``_complete`` are attempt-guarded.

    Both attempt ids are real values; the record is the new owner's. Every write from the
    displaced attempt must be refused (``False``) and leave the record byte-identical.
    """
    import time

    from gateway.native_start import _complete, _hold, _reserve_thread, request_digest

    card = "card-displaced"
    params = _grant_params(card)
    digest = request_digest("default", card, PARENT, params["operator"], BRIEF)
    run_file = _write_starting(gw, card, digest, "new-owner", os.getpid(), time.time())
    before = run_file.read_bytes()
    paths = {"run": run_file,
             "grant": gw.home / "native_start" / "grants" / f"{digest}.json",
             "card_grant": gw.home / "native_start" / "grants" / "unused.json"}

    assert _reserve_thread(paths, digest, "displaced", "key-card-displaced",
                           "9300000000000000999", time.time()) is False
    assert run_file.read_bytes() == before
    assert _hold(paths, digest, "9300000000000000999", "key-card-displaced", "displaced") is False
    assert run_file.read_bytes() == before
    assert _complete(paths, digest, "key-card-displaced", {"thread_id": "x"}, "displaced") is False
    assert run_file.read_bytes() == before


@pytest.mark.asyncio
async def test_displaced_start_fails_closed_instead_of_submitting(gw, monkeypatch):
    """N2: the real call path of an attempt displaced mid-flight fails closed.

    ``create_handoff_thread`` is wrapped so that, right after the thread is created, the
    record is rewritten to name a second owner — exactly what a takeover would leave on
    disk. The displaced attempt must then report ``superseded``, persist nothing, add no
    member, send no brief and admit no turn.
    """
    from gateway.native_start import request_digest
    from hermes_state_runtime import list_session_admissions

    card = "card-superseded"
    params = _grant_params(card)
    peer = f"uid:{os.getuid()}"
    assert (await asyncio.to_thread(gw.handlers["native-start-grant"], params, peer))["ok"]
    run_file = _run_file(gw, card)

    real_create = gw.adapter.create_handoff_thread

    async def takeover_create(parent, name):
        thread_id = await real_create(parent, name)
        record = json.loads(run_file.read_text())
        record["attempt"] = "usurper"
        run_file.write_text(json.dumps(record))
        return thread_id

    monkeypatch.setattr(gw.adapter, "create_handoff_thread", takeover_create)

    result = await asyncio.to_thread(gw.handlers["native-start-thread"], params, peer)
    assert result["ok"] is False and result["code"] == "superseded", result
    assert json.loads(run_file.read_text())["attempt"] == "usurper"
    assert "thread_id" not in json.loads(run_file.read_text())
    assert gw.client.members == [] and gw.client.sent == []
    assert [r for sid in list(gw.authority.sessions)
            for r in list_session_admissions(gw.authority.db, session_id=sid, pending_only=False)] == []
