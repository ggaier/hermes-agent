"""Regression tests for Card24: Background process completion admission in native-start sessions.

Proves:
1. Tracer bullet: A native-start session (whose initial admission is local_operator_v1)
   fails to admit process completion events with RuntimeStoreError('not_found') / WakeNotAccepted('not_found')
   because snapshot_automation currently requires a committed native_text_v1 envelope.
2. Second failure mode: Once a human message is admitted (native_text_v1 with Discord
   scope_id/guild_id/chat_name metadata), subsequent process completions fail with
   RuntimeStoreError('admission_conflict') / WakeNotAccepted('admission_conflict') because
   _route_identity rejects the difference between the stored origin and the enriched human source.
3. Negative controls: Foreign owners, cross-profile/generation mismatches, and unauthorized
   adapters remain strictly refused.
"""
import asyncio
import os
import pytest
from types import SimpleNamespace

from gateway.config import Platform, GatewayConfig
from gateway.run import SessionStore
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session import SessionSource
from gateway.session_authority import SessionAuthority
from gateway.session_contract import Principal, Submission
from gateway.session_envelope import restore_native
from gateway.wake import WakeNotAccepted
from gateway import session_automation
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions, admit_session_input

THREAD_ID = "1564094484428623913"
CHANNEL_ID = "1542188817987936267"
GUILD_ID = "1541785164227809334"
USER_ID = "1060030119420383312"
USER_NAME = "ggaier"
ROUTE = f"agent:main:discord:thread:{THREAD_ID}:{THREAD_ID}"


class MockDiscordAdapter:
    platform = Platform.DISCORD

    def __init__(self):
        self.sent = []

    def authorize_source(self, source):
        return getattr(source, "platform", None) == Platform.DISCORD

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata or {}})
        return SimpleNamespace(success=True)


class GatewayRunnerStub(GatewayNotificationsMixin):
    """Minimal real-interface Gateway runner for notification injection."""

    def __init__(self, home, store, adapter):
        self.hermes_home = home
        self.session_store = store
        self.adapters = {Platform.DISCORD: adapter}
        self.config = GatewayConfig()
        self.profile_id = "default"
        self._draining = False
        self.session_authority = None
        self._cached_agents = {}

    def _resolve_injection_adapter(self, platform_name, source=None):
        try:
            p = Platform(platform_name) if isinstance(platform_name, str) else platform_name
        except Exception:
            return None
        return self.adapters.get(p)

    def _adapter_for_source(self, source):
        return self.adapters.get(getattr(source, "platform", None))

    def _cached_agent_for(self, route):
        return None

    def _adapter_credential_fingerprint(self, adapter):
        # Production derives this from the live connector credential; the harness pins a
        # stable value so a stored envelope's provenance resolves to the same adapter.
        return "adapter:discord:harness"

    def _transport_owner(self, source):
        adapter = self.adapters.get(getattr(source, "platform", None))
        return (adapter, None) if adapter is not None else None

    def _restored_source(self, entry):
        return getattr(entry, "origin", None)


@pytest.fixture
def gateway_fixture(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(home / "state.db")
    store = SessionStore(config=GatewayConfig(), sessions_dir=home / "sessions")
    store._db = db

    adapter = MockDiscordAdapter()
    runner = GatewayRunnerStub(home, store, adapter)
    epoch = begin_runtime_epoch(db, instance_id="test-runner")
    authority = SessionAuthority(runner, profile_id="default", instance_id="test-runner", db=db, epoch=epoch)
    runner.session_authority = authority
    # The production server registers its transport home at server creation; without it
    # restore_provenance refuses with profile_mismatch and the failure mode under test is
    # masked by a harness artifact.
    from gateway.session_ingress_context import register_transport_home
    register_transport_home(runner, None, home)

    # Disable scheduling/inference so turns don't execute in background during test
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)

    # Stored origin as created by native-start
    native_origin = SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        user_id=USER_ID,
        user_name=USER_NAME,
        thread_id=THREAD_ID,
        parent_chat_id=CHANNEL_ID,
    )
    ref = authority.register(native_origin)

    yield SimpleNamespace(
        home=home,
        db=db,
        store=store,
        adapter=adapter,
        runner=runner,
        authority=authority,
        ref=ref,
        origin=native_origin,
    )
    db.close()


@pytest.mark.asyncio
async def test_native_start_session_must_admit_process_completion_without_human_message(gateway_fixture):
    """RED Tracer Bullet: A native-start session commits local_operator_v1.
    When a background process exits, its completion notification must be admitted
    into the session authority without requiring a prior human native_text_v1 turn.

    CURRENT BEHAVIOR (RED): snapshot_automation raises RuntimeStoreError('not_found')
    because committed_route_envelope finds no native_text_v1 admission.
    """
    gf = gateway_fixture

    # 1. Commit the native-start admission (local_operator_v1)
    brief = "# Card24 native start brief"
    receipt = await gf.authority.submit(
        Principal("native-start", "default", frozenset({"session:submit"}), "control-socket"),
        Submission("card24-grant-1", gf.ref, {"text": brief}, "queue"),
    )
    assert receipt.status in {"queued", "started"}

    initial_admissions = list_session_admissions(gf.db, session_id=gf.ref.session_id)
    assert len(initial_admissions) == 1
    assert "local_operator_v1" in initial_admissions[0]["payload"]
    assert "native_text_v1" not in initial_admissions[0]["payload"]

    # 2. Background process proc_d6b0181b856b finishes with exit code 0
    evt = {
        "type": "completion",
        "session_id": "proc_d6b0181b856b",
        "started_at": 1792979819.0,
        "exit_code": 0,
        "completion_reason": "exited",
        "session_key": ROUTE,
        "parent_session_id": gf.ref.session_id,
        "platform": "discord",
        "chat_id": THREAD_ID,
        "thread_id": THREAD_ID,
        "chat_type": "thread",
        "user_id": USER_ID,
        "user_name": USER_NAME,
    }

    # 3. Gateway attempts to inject the watch notification into the session authority.
    # On current code, this raises WakeNotAccepted("not_found") because committed_route_envelope
    # returns None when only local_operator_v1 exists.
    delivered = await gf.runner._inject_watch_notification(
        "[IMPORTANT: Background process proc_d6b0181b856b completed]", evt, raise_not_accepted=True,
    )
    assert delivered is True, "Process completion must be admitted by the session authority"

    admissions = list_session_admissions(gf.db, session_id=gf.ref.session_id)
    automation_rows = [r for r in admissions if r["principal_id"].startswith("automation:")]
    assert len(automation_rows) == 1, "Expected exactly one admitted automation row for the completion"
    assert automation_rows[0]["status"] == "queued"


@pytest.mark.asyncio
async def test_process_completion_must_admit_despite_discord_channel_metadata_enrichment(gateway_fixture):
    """RED Test 2: Once a human Discord message is admitted, its native_text_v1
    envelope retains Discord-specific channel metadata (guild_id, scope_id, chat_name).
    Subsequent process completions (whose event.source is rebuilt from stored origin)
    must still be admitted without admission_conflict.

    CURRENT BEHAVIOR (RED): snapshot_automation raises RuntimeStoreError('admission_conflict')
    because _route_identity(restored.source) != _route_identity(event.source).
    """
    gf = gateway_fixture

    # 1. Commit native-start
    await gf.authority.submit(
        Principal("native-start", "default", frozenset({"session:submit"}), "control-socket"),
        Submission("card24-grant-1", gf.ref, {"text": "brief"}, "queue"),
    )

    # 2. Simulate user sending "is it done yet" via Discord admitted to state.db
    human_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        user_id=USER_ID,
        user_name=USER_NAME,
        thread_id=THREAD_ID,
        parent_chat_id=CHANNEL_ID,
        guild_id=GUILD_ID,
        scope_id=GUILD_ID,
        chat_name="AI Bots / #tele-mac / getrich GR11-04",
        is_bot=False,
    )
    human_source_dict = human_source.to_dict()
    human_source_dict["is_bot"] = False
    human_payload = {
        "text": "is it done yet",
        "native_text_v1": {
            "source": human_source_dict,
            "provenance": {
                "connector": "adapter:discord:harness",
                "platform": "discord",
                "runtime_home": str(gf.home),
                "transport_home": str(gf.home),
            },
            "route": ROUTE,
            "timestamp": "2026-10-26T02:30:14.387000+00:00",
            "event": {"message_id": "1564103837009182791"},
        },
    }
    admit_session_input(
        gf.db,
        epoch=gf.authority.epoch,
        principal_id="messaging:[null,\"discord\",...]",
        session_id=gf.ref.session_id,
        request_id="1564103837009182791",
        payload=human_payload,
        intent="queue",
    )

    # 3. Now second background process proc_1c4a39db5615 finishes
    evt = {
        "type": "completion",
        "session_id": "proc_1c4a39db5615",
        "started_at": 1792981910.0,
        "exit_code": 0,
        "completion_reason": "exited",
        "session_key": ROUTE,
        "parent_session_id": gf.ref.session_id,
        "platform": "discord",
        "chat_id": THREAD_ID,
        "thread_id": THREAD_ID,
        "chat_type": "thread",
        "user_id": USER_ID,
        "user_name": USER_NAME,
    }

    # On current code, this raises WakeNotAccepted("admission_conflict")
    delivered = await gf.runner._inject_watch_notification(
        "[IMPORTANT: Background process proc_1c4a39db5615 completed]", evt, raise_not_accepted=True,
    )
    assert delivered is True, "Process completion must not be refused due to Discord channel metadata enrichment"


@pytest.mark.asyncio
async def test_persisted_origin_placeholder_user_name_must_not_refuse_completion(gateway_fixture):
    """Regression (live 2026-10-26 post-deploy): native_start builds the thread's source with

    ``user_name=str(operator.get("user_name") or "operator")`` — the bridge grant carries the
    operator id but often no display name, so the PERSISTED origin stores the literal
    placeholder ``"operator"`` while the human message commits the real display name
    (``"ggaier"``). ``user_name`` is a presentation label, not a routing key: a completion must
    still be admitted. Production logs proved this exact difference refused every completion
    with ``admission_conflict`` after the restart, looping once per retry per session.
    """
    gf = gateway_fixture

    await gf.authority.submit(
        Principal("native-start", "default", frozenset({"session:submit"}), "control-socket"),
        Submission("card24-grant-1", gf.ref, {"text": "brief"}, "queue"),
    )

    human_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        user_id=USER_ID,
        user_name=USER_NAME,
        thread_id=THREAD_ID,
        parent_chat_id=CHANNEL_ID,
        guild_id=GUILD_ID,
        scope_id=GUILD_ID,
        chat_name="AI Bots / #tele-mac / placeholder name",
        is_bot=False,
    )
    human_source_dict = human_source.to_dict()
    human_source_dict["is_bot"] = False
    admit_session_input(
        gf.db,
        epoch=gf.authority.epoch,
        principal_id="messaging:[null,\"discord\",...]",
        session_id=gf.ref.session_id,
        request_id="1564116193932222497",
        payload={
            "text": "payload built with the placeholder user_name the persisted origin carries",
            "native_text_v1": {
                "source": human_source_dict,
                "provenance": {
                    "connector": "adapter:discord:harness",
                    "platform": "discord",
                    "runtime_home": str(gf.home),
                    "transport_home": str(gf.home),
                },
                "route": ROUTE,
                "timestamp": "2026-10-26T10:11:03.000000+00:00",
                "event": {"message_id": "1564116193932222497"},
            },
        },
        intent="queue",
    )

    # The stored origin the bridge actually wrote: same route, placeholder display name.
    entry = gf.store.lookup_by_session_key(ROUTE)
    from gateway.session_identity import replace_source
    entry.origin = replace_source(gf.origin, user_name="operator")

    evt = {
        "type": "completion",
        "session_id": "proc_placeholder_name",
        "started_at": 1792982910.0,
        "exit_code": 0,
        "completion_reason": "exited",
        "session_key": ROUTE,
        "parent_session_id": gf.ref.session_id,
        "platform": "discord",
        "chat_id": THREAD_ID,
        "thread_id": THREAD_ID,
        "chat_type": "thread",
        "user_id": USER_ID,
        "user_name": "operator",
    }
    delivered = await gf.runner._inject_watch_notification(
        "[IMPORTANT: Background process proc_placeholder_name completed]", evt, raise_not_accepted=True,
    )
    assert delivered is True, "A placeholder display name must not refuse the completion"


@pytest.mark.asyncio
async def test_conflicting_user_id_still_refuses_completion(gateway_fixture):
    """Negative control for the identity relaxation: ``user_name`` is presentation, ``user_id``
    is not. A source whose real user id differs must still be refused."""
    gf = gateway_fixture

    await gf.authority.submit(
        Principal("native-start", "default", frozenset({"session:submit"}), "control-socket"),
        Submission("card24-grant-1", gf.ref, {"text": "brief"}, "queue"),
    )
    other_source = SessionSource(
        platform=Platform.DISCORD, chat_id=THREAD_ID, chat_type="thread",
        user_id="999000111", user_name="someone-else", thread_id=THREAD_ID,
        parent_chat_id=CHANNEL_ID, guild_id=GUILD_ID, scope_id=GUILD_ID,
    )
    other_dict = other_source.to_dict()
    other_dict["is_bot"] = False
    admit_session_input(
        gf.db, epoch=gf.authority.epoch,
        principal_id="messaging:[null,\"discord\",...]", session_id=gf.ref.session_id,
        request_id="human-other-user",
        payload={"text": "hi", "native_text_v1": {
            "source": other_dict,
            "provenance": {"connector": "adapter:discord:harness", "platform": "discord",
                           "runtime_home": str(gf.home), "transport_home": str(gf.home)},
            "route": ROUTE, "timestamp": "2026-10-26T10:11:03.000000+00:00",
            "event": {"message_id": "human-other-user"}}},
        intent="queue",
    )
    evt = {
        "type": "completion", "session_id": "proc_foreign_user", "started_at": 1792983910.0,
        "exit_code": 0, "completion_reason": "exited", "session_key": ROUTE,
        "parent_session_id": gf.ref.session_id, "platform": "discord",
        "chat_id": THREAD_ID, "thread_id": THREAD_ID, "chat_type": "thread",
        "user_id": USER_ID, "user_name": USER_NAME,
    }
    with pytest.raises(WakeNotAccepted):
        await gf.runner._inject_watch_notification(
            "[IMPORTANT: Background process proc_foreign_user completed]", evt, raise_not_accepted=True,
        )


@pytest.mark.asyncio
async def test_negative_controls_foreign_owner_and_adapter_remain_refused(gateway_fixture):
    """Negative controls: Genuine security and ownership boundaries must remain enforced.
    1. A completion event claiming a foreign parent_session_id must be refused (admission_conflict).
    2. A completion event targeting an unauthorized adapter must be refused.
    """
    gf = gateway_fixture

    # 1. Foreign owner parent_session_id
    foreign_evt = {
        "type": "completion",
        "session_id": "proc_d6b0181b856b",
        "started_at": 1792979819.0,
        "exit_code": 0,
        "session_key": ROUTE,
        "parent_session_id": "foreign_owner_session_9999",
        "platform": "discord",
        "chat_id": THREAD_ID,
        "thread_id": THREAD_ID,
        "chat_type": "thread",
    }
    with pytest.raises(WakeNotAccepted, match="admission_conflict"):
        await gf.runner._inject_watch_notification("completion", foreign_evt, raise_not_accepted=True)

    # 2. Unauthorized / unsupported adapter platform
    bad_platform_evt = {
        "type": "completion",
        "session_id": "proc_d6b0181b856b",
        "started_at": 1792979819.0,
        "exit_code": 0,
        "session_key": ROUTE,
        "parent_session_id": gf.ref.session_id,
        "platform": "telegram",  # Telegram adapter not configured
        "chat_id": "99999",
        "chat_type": "dm",
    }
    result = await gf.runner._inject_watch_notification("completion", bad_platform_evt, raise_not_accepted=False)
    assert result is False or result is None
