"""Card27 regression: an operator-branch automation snapshot must survive the FIFO preclaim.

Proven incident: a session started by ``native-start`` holds one completed
``local_operator_v1`` input and no committed ``native_text_v1`` envelope. When a background
process finishes, ``snapshot_automation``'s operator branch
(``gateway/session_automation.py:458``) builds the wake source from
``event.source.to_dict()``. ``SessionSource.to_dict`` never carries ``is_bot`` (it is a
platform-neutral flag, ``gateway/session.py:81``), so the payload it commits has no
``is_bot`` key. The admission itself succeeds and the row is queued.

The failure lands at the next boundary: the FIFO preclaim ``check_automation_route`` restores
the queued payload with ``restore_native`` -> ``restore_native_source``, which reads
``envelope['source']['is_bot']`` unconditionally (``gateway/session_envelope.py:242``). The
wake dies with ``KeyError('is_bot')``: the head stays ``queued`` forever, later user messages
block behind it, and ``/stop`` sees no active task.

RED baseline: ``check_automation_route`` raises ``KeyError('is_bot')``.
GREEN: it returns this route's restored source and that source is ``is_bot is False``.

The second stage pins the decoder contract the producer fix cannot supply alone. Rows already
committed by the broken producer (production ``session_admissions`` seq728/732) have no
``is_bot`` and must still be claimable, because the compatibility is scoped to an INTERNAL
automation descriptor: an ordinary native payload missing the flag, or any payload carrying a
malformed value, must refuse with ``RuntimeStoreError('invalid_params')`` instead of surfacing
a bare ``KeyError`` or silently coercing the sender identity.
"""
from copy import deepcopy

import pytest

from gateway.session_automation import check_automation_route
from gateway.session_contract import Principal, Submission
from gateway.session_envelope import check_native_route, restore_native_source
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

# Reuse the real SessionDB/SessionAuthority/SessionStore harness (private tmp home, registered
# transport home + adapter fingerprint, scheduling disabled only) rather than re-stubbing it.
from tests.gateway.test_native_start_process_completion_admission import (
    ROUTE,
    THREAD_ID,
    USER_ID,
    USER_NAME,
    gateway_fixture,
)

_MALFORMED_IS_BOT = ["false", 0, 1, None, [], {}]


def _ordinary_native_payload(source_dict):
    """A committed native input with no automation descriptor (an ordinary human message)."""
    return {
        "text": "hello",
        "native_text_v1": {
            "source": source_dict,
            "route": ROUTE,
            "timestamp": "2026-10-27T10:10:26.000000+00:00",
            "event": {"message_id": "card27-ordinary"},
        },
    }


async def _queue_operator_completion(gf, proc_id):
    """Drive the real producer: native-start operator admission then a completion wake."""
    receipt = await gf.authority.submit(
        Principal("native-start", "default", frozenset({"session:submit"}), "control-socket"),
        Submission("card27-grant-1", gf.ref, {"text": "card27 brief"}, "queue"),
    )
    assert receipt.status in {"queued", "started"}

    evt = {
        "type": "completion",
        "session_id": proc_id,
        "started_at": 1792981010.0,
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
    delivered = await gf.runner._inject_watch_notification(
        f"[IMPORTANT: Background process {proc_id} completed]", evt, raise_not_accepted=True,
    )
    assert delivered is True

    rows = [row for row in list_session_admissions(gf.db, session_id=gf.ref.session_id)
            if row["principal_id"].startswith("automation:")]
    assert len(rows) == 1
    head = rows[0]
    assert head["status"] == "queued"
    return head, gf.store.lookup_by_session_key(ROUTE)


def _legacy_operator_payload(head):
    """The exact seq728/732 shape: a generated operator wake with only ``is_bot`` removed."""
    payload = deepcopy(head["payload"])
    assert "automation" in payload["native_text_v1"]
    source = payload["native_text_v1"]["source"]
    assert source["is_bot"] is False
    del source["is_bot"]
    assert "is_bot" not in source
    return payload


@pytest.mark.asyncio
async def test_native_start_operator_automation_is_preclaimable(gateway_fixture):
    gf = gateway_fixture

    # 1. native-start commits the session's only admission: local_operator_v1.
    receipt = await gf.authority.submit(
        Principal("native-start", "default", frozenset({"session:submit"}), "control-socket"),
        Submission("card27-grant-1", gf.ref, {"text": "card27 brief"}, "queue"),
    )
    assert receipt.status in {"queued", "started"}
    assert "local_operator_v1" in list_session_admissions(gf.db, session_id=gf.ref.session_id)[0]["payload"]

    # 2. A background process finishes and its completion is admitted as an automation wake.
    evt = {
        "type": "completion",
        "session_id": "proc_card27",
        "started_at": 1792981010.0,
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
    delivered = await gf.runner._inject_watch_notification(
        "[IMPORTANT: Background process proc_card27 completed]", evt, raise_not_accepted=True,
    )
    assert delivered is True

    rows = [row for row in list_session_admissions(gf.db, session_id=gf.ref.session_id)
            if row["principal_id"].startswith("automation:")]
    assert len(rows) == 1
    head = rows[0]
    # The wake queues fine; the incident's damage starts when the FIFO tries to run it.
    assert head["status"] == "queued"
    # The producer must persist the flag explicitly: the decoder indexes this key.
    stored_source = head["payload"]["native_text_v1"]["source"]
    assert "is_bot" in stored_source
    assert stored_source["is_bot"] is False

    # 3. FIFO preclaim of that queued head, exactly as the drain runs it.
    entry = gf.store.lookup_by_session_key(ROUTE)
    source, route = check_automation_route(
        gf.runner, head["payload"], head["target_session_id"], entry.origin, gf.adapter,
    )

    assert route == ROUTE
    assert source.platform.value == "discord"
    assert source.is_bot is False


@pytest.mark.asyncio
async def test_legacy_operator_automation_without_is_bot_is_preclaimable(gateway_fixture):
    """seq728/732 compatibility: a stored internal-automation wake without ``is_bot`` claims."""
    gf = gateway_fixture
    head, entry = await _queue_operator_completion(gf, "proc_card27_legacy")
    payload = _legacy_operator_payload(head)

    source, route = check_automation_route(
        gf.runner, payload, head["target_session_id"], entry.origin, gf.adapter,
    )

    assert route == ROUTE
    assert source.platform.value == "discord"
    assert source.is_bot is False


@pytest.mark.asyncio
async def test_legacy_operator_automation_still_enforces_owner_adapter_and_provenance(gateway_fixture):
    """The legacy default must not loosen the preclaim's ownership checks."""
    gf = gateway_fixture
    head, entry = await _queue_operator_completion(gf, "proc_card27_legacy_guard")
    base = _legacy_operator_payload(head)
    session_id, available_source, adapter = head["target_session_id"], entry.origin, gf.adapter

    foreign_owner = deepcopy(base)
    foreign_owner["native_text_v1"]["automation"]["owner"] = "foreign-owner"
    with pytest.raises(RuntimeStoreError, match="admission_conflict"):
        check_automation_route(gf.runner, foreign_owner, session_id, available_source, adapter)

    with pytest.raises(RuntimeStoreError, match="admission_conflict"):
        check_automation_route(gf.runner, deepcopy(base), session_id, available_source, None)

    forged_provenance = deepcopy(base)
    forged_provenance["native_text_v1"]["provenance"]["connector"] = "adapter:discord:forged"
    with pytest.raises(RuntimeStoreError, match="profile_mismatch"):
        check_automation_route(gf.runner, forged_provenance, session_id, available_source, adapter)


@pytest.mark.asyncio
async def test_ordinary_native_payload_without_is_bot_fails_closed(gateway_fixture):
    """Only internal automation may default the flag; an ordinary sender must refuse."""
    gf = gateway_fixture
    source_dict = gf.origin.to_dict()
    assert "is_bot" not in source_dict
    payload = _ordinary_native_payload(source_dict)

    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        await check_native_route(gf.runner, payload, gf.ref.session_id, gf.origin, gf.adapter)


@pytest.mark.parametrize("is_bot", _MALFORMED_IS_BOT)
def test_malformed_is_bot_in_ordinary_native_payload_is_refused(gateway_fixture, is_bot):
    payload = _ordinary_native_payload(gateway_fixture.origin.to_dict())
    payload["native_text_v1"]["source"]["is_bot"] = is_bot

    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        restore_native_source(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_bot", _MALFORMED_IS_BOT)
async def test_malformed_is_bot_in_legacy_automation_is_refused(gateway_fixture, is_bot):
    gf = gateway_fixture
    head, entry = await _queue_operator_completion(gf, "proc_card27_legacy_bad")
    payload = _legacy_operator_payload(head)
    payload["native_text_v1"]["source"]["is_bot"] = is_bot

    with pytest.raises(RuntimeStoreError, match="invalid_params"):
        check_automation_route(
            gf.runner, payload, head["target_session_id"], entry.origin, gf.adapter,
        )


@pytest.mark.parametrize("is_bot", [False, True])
def test_valid_is_bot_bool_is_preserved(gateway_fixture, is_bot):
    payload = _ordinary_native_payload(gateway_fixture.origin.to_dict())
    payload["native_text_v1"]["source"]["is_bot"] = is_bot

    restored = restore_native_source(payload)

    assert restored.is_bot is is_bot
    assert type(restored.is_bot) is bool
    assert restored.platform.value == "discord"
