"""Card27 stage 2: the real durable FIFO around a legacy automation head and a broken head.

Stage 1 pinned the envelope decoder alone. These tests keep the real ``SessionAuthority``,
``SessionDB`` queue and preclaim (``check_native_route``/``check_automation_route``) and only
record ``_handle_message`` so no model runs.

A. A legacy internal-automation row without ``is_bot`` must claim, run once, and leave the
   committed payload/digest untouched; a following human input runs once, in order.
B. A malformed persisted head (``KeyError``/``TypeError``/``ValueError`` inside the native
   decode of the preclaim) must become a controlled ``invalid_params`` pause: the pending
   waiter is released with that reason, rows stay queued, nothing runs, the log names the
   session and the exact exception type with traceback, and another session's healthy input
   still runs.
C. Repeating that broken head stays bounded, produces one user-visible pause notice instead of
   a false success, and authorization mismatches remain refused.
"""
import asyncio
from copy import deepcopy
from datetime import UTC, datetime
import logging
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_automation import check_automation_route
from gateway.session_cron import unbind_owner
from gateway.session_ingress import admit_message
from gateway.session_ingress_context import (
    native_callback,
    register_transport_home,
    resolve_operator_provenance,
)
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError,
    admit_session_input,
    list_session_admissions,
)
from plugins.platforms.discord.adapter import DiscordAdapter

OPERATOR = "42"


def _raw_admission(db, admission_id):
    with db._read_ctx() as conn:
        return dict(conn.execute(
            "SELECT * FROM session_admissions WHERE admission_id=?", (admission_id,)).fetchone())


def _automation_payload(runner, authority, entry, source, identity, text):
    """The operator-branch wake shape committed before the producer stamped ``is_bot``."""
    adapter, provenance = resolve_operator_provenance(runner, source, None)
    assert adapter is runner._adapter_for_source(source)
    source_dict = deepcopy(source.to_dict())
    source_dict.pop("message_id", None)
    source_dict.pop("is_bot", None)
    assert "is_bot" not in source_dict
    return {
        "text": text,
        "native_text_v1": {
            "source": source_dict,
            "route": entry.session_key,
            "timestamp": datetime.fromtimestamp(0, UTC).isoformat(),
            "event": {"message_id": identity},
            "provenance": provenance,
            "automation": {
                "identity": identity,
                "owner": authority.logical_owner(entry.session_id),
            },
        },
    }


MISSING = object()


def _malformed_native_payload(entry, source, runner, identity, *, timestamp=MISSING):
    _adapter, provenance = resolve_operator_provenance(runner, source, None)
    source_dict = deepcopy(source.to_dict())
    source_dict["is_bot"] = False
    envelope = {
        "source": source_dict,
        "route": entry.session_key,
        "event": {"message_id": identity},
        "provenance": provenance,
    }
    if timestamp is not MISSING:
        envelope["timestamp"] = timestamp
    return {"text": "broken head", "native_text_v1": envelope}


# Each durable-head shape is broken at a different point of the native decode.
MALFORMED_HEADS = [
    pytest.param(MISSING, "KeyError", id="timestamp-missing"),
    pytest.param(None, "TypeError", id="timestamp-none"),
    pytest.param("not-a-timestamp", "ValueError", id="timestamp-invalid"),
]


async def _wait_for_waiter(authority, timeout=5):
    for _ in range(int(timeout / 0.01)):
        if authority.waiters:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("delivery waiter was never registered")


@pytest_asyncio.fixture
async def fifo(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", OPERATOR)
    # Nothing claims a row in the background: these tests drive ``_drain`` by hand.
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)

    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    db = SessionDB(home / "state.db")
    runner.session_store._db = db
    runner._session_db = db
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token",
                                            typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="card27-fifo", db=db)
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, home)

    executed = []

    async def handle(event):
        executed.append(event.text)
        return ""

    monkeypatch.setattr(runner, "_handle_message", handle)
    namespace = SimpleNamespace(
        home=home, db=db, runner=runner, adapter=adapter, authority=authority,
        executed=executed,
        source=lambda chat_id="42": adapter.build_source(
            chat_id=chat_id, chat_type="dm", user_id=OPERATOR),
    )
    try:
        yield namespace
    finally:
        unbind_owner(authority)
        db.close()


def _register(runner, authority, source):
    ref = authority.register(source)
    entry = runner.session_store.lookup_by_session_key(
        runner.session_store._generate_session_key(source))
    assert entry is not None
    return ref, entry


async def _admit_human(runner, authority, home, source, text, message_id):
    event = MessageEvent(text=text, source=source, message_id=message_id)
    with native_callback(runner, event, home):
        receipt = await authority.admit_native(event)
    assert receipt.status == "queued"
    return event, receipt


@pytest.mark.asyncio
async def test_legacy_automation_then_human_drains_once_in_order_without_payload_migration(fifo):
    gf = fifo
    source = gf.source()
    ref, entry = _register(gf.runner, gf.authority, source)

    legacy = _automation_payload(
        gf.runner, gf.authority, entry, source, "legacy-wake", "legacy wake")
    legacy_row = admit_session_input(
        gf.db, epoch=gf.authority.epoch,
        principal_id="automation:" + entry.session_key, session_id=ref.session_id,
        request_id="legacy-wake", payload=legacy, intent="queue")
    assert legacy_row["status"] == "queued"

    _human_event, human_row = await _admit_human(
        gf.runner, gf.authority, gf.home, source, "after legacy", "human-after")

    rows = list_session_admissions(gf.db, session_id=ref.session_id)
    assert [row["admission_id"] for row in rows] == [
        legacy_row["admission_id"], human_row.admission_id]
    before = {row["admission_id"]: _raw_admission(gf.db, row["admission_id"]) for row in rows}

    await asyncio.wait_for(gf.authority._drain(ref), 10)
    assert gf.executed == ["legacy wake", "after legacy"]

    rows = list_session_admissions(
        gf.db, session_id=ref.session_id, pending_only=False)
    assert [(row["admission_id"], row["status"], row["outcome"]) for row in rows] == [
        (legacy_row["admission_id"], "terminal", "completed"),
        (human_row.admission_id, "terminal", "completed"),
    ]

    # Repeated drain must not replay either terminal row.
    await asyncio.wait_for(gf.authority._drain(ref), 10)
    assert gf.executed == ["legacy wake", "after legacy"]

    # The compatibility default is decode-only: the stored wake keeps the legacy shape.
    for admission_id, raw in before.items():
        after = _raw_admission(gf.db, admission_id)
        assert after["payload_json"] == raw["payload_json"]
        assert after["payload_digest"] == raw["payload_digest"]
    import json
    restored = json.loads(_raw_admission(gf.db, legacy_row["admission_id"])["payload_json"])
    assert "is_bot" not in restored["native_text_v1"]["source"]


@pytest.mark.parametrize("timestamp, exc_name", MALFORMED_HEADS)
@pytest.mark.asyncio
async def test_malformed_head_pauses_without_raising_or_hanging_and_other_session_runs(
        timestamp, exc_name, fifo, caplog):
    gf = fifo
    source = gf.source("42")
    ref, entry = _register(gf.runner, gf.authority, source)

    broken = _malformed_native_payload(
        entry, source, gf.runner, "broken-head", timestamp=timestamp)
    broken_row = admit_session_input(
        gf.db, epoch=gf.authority.epoch, principal_id="messaging:broken",
        session_id=ref.session_id, request_id="broken-head", payload=broken, intent="queue")
    loop = asyncio.get_running_loop()
    head_waiter = gf.authority.waiters.setdefault(
        broken_row["admission_id"], loop.create_future())
    gf.authority.native_waiters.add(broken_row["admission_id"])

    # A valid input behind the broken head must stay queued, not run around it.
    await _admit_human(gf.runner, gf.authority, gf.home, source, "queued behind", "behind")

    # Another session's healthy input must not be collateral damage.
    other_source = gf.source("43")
    other_ref, _other_entry = _register(gf.runner, gf.authority, other_source)
    await _admit_human(gf.runner, gf.authority, gf.home, other_source, "healthy", "healthy-1")

    caplog.set_level(logging.WARNING, logger="gateway.session_authority")
    await asyncio.wait_for(
        asyncio.gather(gf.authority._drain(ref), gf.authority._drain(other_ref)), 10)

    assert head_waiter.done()
    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        await head_waiter

    pending = list_session_admissions(gf.db, session_id=ref.session_id)
    assert [(row["request_id"], row["status"]) for row in pending] == [
        ("broken-head", "queued"), ("behind", "queued")]
    assert gf.executed == ["healthy"]

    healthy = list_session_admissions(
        gf.db, session_id=other_ref.session_id, pending_only=False)
    assert [(row["status"], row["outcome"]) for row in healthy] == [("terminal", "completed")]

    log = caplog.text
    assert ref.session_id in log
    assert "invalid_params" in log
    assert "Traceback (most recent call last)" in log, log
    assert f"{exc_name}: " in log, log


@pytest.mark.parametrize("timestamp, _exc_name", MALFORMED_HEADS)
@pytest.mark.asyncio
async def test_repeated_malformed_head_is_bounded_and_user_visible_not_success(
        timestamp, _exc_name, fifo):
    gf = fifo
    source = gf.source("44")
    ref, entry = _register(gf.runner, gf.authority, source)

    broken = _malformed_native_payload(
        entry, source, gf.runner, "broken-repeat", timestamp=timestamp)
    admit_session_input(
        gf.db, epoch=gf.authority.epoch, principal_id="messaging:broken",
        session_id=ref.session_id, request_id="broken-repeat", payload=broken, intent="queue")

    # First drain observes the broken head and pauses.
    await asyncio.wait_for(gf.authority._drain(ref), 5)

    # A user message arriving on the paused conversation gets a pause notice, not a success.
    event = MessageEvent(text="still there?", source=source, message_id="after-pause")
    notice_task = asyncio.create_task(
        admit_message_in_callback(gf.runner, gf.authority, gf.home, event))
    await _wait_for_waiter(gf.authority)
    await asyncio.wait_for(gf.authority._drain(ref), 5)
    notice = await asyncio.wait_for(notice_task, 5)

    assert notice is not None
    assert "paused" in notice.lower() and "invalid_params" in notice
    assert gf.executed == []

    # A repeat drain of the same broken head is bounded: no raise, no execution, all queued.
    await asyncio.wait_for(gf.authority._drain(ref), 5)
    rows = list_session_admissions(gf.db, session_id=ref.session_id)
    assert [(row["request_id"], row["status"]) for row in rows] == [
        ("broken-repeat", "queued"), ("after-pause", "queued")]
    assert gf.executed == []


async def admit_message_in_callback(runner, authority, home, event):
    with native_callback(runner, event, home):
        return await admit_message(authority, event)


@pytest.mark.asyncio
async def test_foreign_owner_automation_head_stays_refused(fifo, caplog):
    """Containment must not turn an authorization mismatch into a silent success, and must not
    relabel it: the drain pauses on ``admission_conflict``, the refusal the preflight raised."""
    gf = fifo
    source = gf.source("45")
    ref, entry = _register(gf.runner, gf.authority, source)

    forged = _automation_payload(
        gf.runner, gf.authority, entry, source, "foreign", "foreign wake")
    forged["native_text_v1"]["automation"]["owner"] = "foreign-owner"
    row = admit_session_input(
        gf.db, epoch=gf.authority.epoch, principal_id="automation:" + entry.session_key,
        session_id=ref.session_id, request_id="foreign", payload=forged, intent="queue")
    loop = asyncio.get_running_loop()
    head_waiter = gf.authority.waiters.setdefault(
        row["admission_id"], loop.create_future())
    gf.authority.native_waiters.add(row["admission_id"])
    await _admit_human(gf.runner, gf.authority, gf.home, source, "behind", "behind-foreign")

    with pytest.raises(RuntimeStoreError, match="admission_conflict"):
        check_automation_route(gf.runner, forged, ref.session_id, source, gf.adapter)

    caplog.set_level(logging.WARNING, logger="gateway.session_authority")
    await asyncio.wait_for(gf.authority._drain(ref), 5)

    # The real drain's refusal reaches the delivery waiter with the domain reason, not a
    # containment code invented for a decode failure.
    assert head_waiter.done()
    with pytest.raises(RuntimeStoreError) as refusal:
        await head_waiter
    assert refusal.value.reason == "admission_conflict"

    rows = list_session_admissions(gf.db, session_id=ref.session_id)
    assert [(row["request_id"], row["status"]) for row in rows] == [
        ("foreign", "queued"), ("behind-foreign", "queued")]
    assert row["status"] == "queued"
    assert gf.executed == []

    log = caplog.text
    assert f"{ref.session_id} paused: admission_conflict" in log, log
    assert "invalid_params" not in log, log


@pytest.mark.asyncio
async def test_forged_native_route_pauses_with_domain_reason_not_invalid_params(fifo, caplog):
    """Same class of mistake on the plain native preflight: a head that decodes (valid
    timestamp) but names a foreign route is refused as ``admission_conflict``."""
    gf = fifo
    source = gf.source("46")
    ref, entry = _register(gf.runner, gf.authority, source)

    forged = _malformed_native_payload(
        entry, source, gf.runner, "forged-route",
        timestamp=datetime.fromtimestamp(0, UTC).isoformat())
    forged["native_text_v1"]["route"] = "discord:dm:forged-lane"
    row = admit_session_input(
        gf.db, epoch=gf.authority.epoch, principal_id="messaging:forged",
        session_id=ref.session_id, request_id="forged-route", payload=forged, intent="queue")
    loop = asyncio.get_running_loop()
    head_waiter = gf.authority.waiters.setdefault(
        row["admission_id"], loop.create_future())
    gf.authority.native_waiters.add(row["admission_id"])

    caplog.set_level(logging.WARNING, logger="gateway.session_authority")
    await asyncio.wait_for(gf.authority._drain(ref), 5)

    assert head_waiter.done()
    with pytest.raises(RuntimeStoreError) as refusal:
        await head_waiter
    assert refusal.value.reason == "admission_conflict"

    rows = list_session_admissions(gf.db, session_id=ref.session_id)
    assert [(row["request_id"], row["status"]) for row in rows] == [
        ("forged-route", "queued")]
    assert gf.executed == []

    log = caplog.text
    assert f"{ref.session_id} paused: admission_conflict" in log, log
    assert "invalid_params" not in log, log
