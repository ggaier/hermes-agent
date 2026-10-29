"""Registered notifier lifetimes versus atomic approval admission (#33)."""

import threading

import pytest

from tools import approval as mod
from tools import approval_gateway_wait as wait_mod
from tests.tools import test_approval_cancelled_attribution as attribution


gateway_session = attribution.gateway_session
SESSION_KEY = attribution.SESSION_KEY
APPROVAL = {"command": "rm -rf .git", "pattern_key": "recursive delete",
            "pattern_keys": ["recursive delete"], "description": "recursive delete"}


@pytest.fixture(autouse=True)
def record_lifecycle_hooks(gateway_session, monkeypatch):
    # Keep these thread deadlines about approval admission, independent of plugin discovery.
    # The existing plugin-hook suite exercises the real dispatcher separately.
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook",
                        lambda name, **kw: gateway_session.append((name, kw)))


@pytest.mark.parametrize("boundary", ["clear", "replace", "unregister_reregister"])
def test_captured_notifier_cannot_enter_a_later_lifetime(gateway_session, monkeypatch, boundary):
    captured = threading.Event()
    release = threading.Event()
    stale_notifications = []
    current_notifications = []
    holder = {}
    lookup = mod._gateway_notify_cb
    original_callback = stale_notifications.append

    def paused_lookup(session_key):
        callback = lookup(session_key)
        captured.set()
        assert release.wait(10)
        return callback

    monkeypatch.setattr(mod, "_gateway_notify_cb", paused_lookup)
    mod.register_gateway_notify(SESSION_KEY, original_callback)
    thread = threading.Thread(target=lambda: holder.setdefault(
        "result", mod.check_all_command_guards("rm -rf .git", "local")), daemon=True)
    thread.start()
    try:
        assert captured.wait(10)
        if boundary == "clear":
            mod.clear_session(SESSION_KEY)
        elif boundary == "replace":
            mod.register_gateway_notify(SESSION_KEY, current_notifications.append)
        else:
            mod.unregister_gateway_notify(SESSION_KEY)
            # Deliberately the identical callable: callback identity alone misses this ABA.
            mod.register_gateway_notify(SESSION_KEY, original_callback)
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        cause = "the session ended" if boundary == "clear" else "the turn ended"
        attribution._assert_withdrawn(holder["result"], cause)
        assert not mod.has_blocking_approval(SESSION_KEY)
        assert stale_notifications == current_notifications == []
        assert not mod._session_approved.get(SESSION_KEY)
        assert mod.resolve_gateway_approval(SESSION_KEY, "always") == 0
        # clear_session retains a usable channel for subsequent requests.
        current = lookup(SESSION_KEY)
        assert current is not None

        def answer(event, session_key, *, interrupt_log):
            assert mod.resolve_gateway_approval(session_key, "once") == 1
            return "set"

        monkeypatch.setattr(wait_mod, "_poll_event", answer)
        result = wait_mod._await_gateway_decision(SESSION_KEY, current, APPROVAL)
        assert result == {"resolved": True, "choice": "once", "reason": None}
        assert len(stale_notifications + current_notifications) == 1
    finally:
        release.set()
        mod.unregister_gateway_notify(SESSION_KEY)
        thread.join(10)


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("choice", ["once", "session", "deny"])
def test_live_registered_and_standalone_callbacks_can_answer(gateway_session, registered, choice):
    notifications = []

    def notify(data):
        notifications.append(data)
        # Acquires the registry lock itself: notify must run outside that lock.
        assert mod.resolve_gateway_approval(SESSION_KEY, choice) == 1

    if registered:
        mod.register_gateway_notify(SESSION_KEY, notify)
        callback = mod._gateway_notify_cb(SESSION_KEY)
    else:
        callback = notify
    result = wait_mod._await_gateway_decision(SESSION_KEY, callback, APPROVAL)
    assert result == {"resolved": True, "choice": choice, "reason": None}
    assert len(notifications) == 1
    assert not mod.has_blocking_approval(SESSION_KEY)


@pytest.mark.parametrize("choice", ["session", "deny"])
def test_registered_followers_share_explicit_settlement(gateway_session, monkeypatch, choice):
    notified = threading.Event()
    following = threading.Event()
    results = []
    notifications = []
    real_follow = wait_mod._await_coalesced_leader

    def follow(*args):
        following.set()
        return real_follow(*args)

    def notify(data):
        notifications.append(data)
        notified.set()

    monkeypatch.setattr(wait_mod, "_await_coalesced_leader", follow)
    mod.register_gateway_notify(SESSION_KEY, notify)
    callback = mod._gateway_notify_cb(SESSION_KEY)
    threads = [threading.Thread(target=lambda: results.append(
        wait_mod._await_gateway_decision(SESSION_KEY, callback, APPROVAL)), daemon=True) for _ in range(2)]
    try:
        threads[0].start()
        assert notified.wait(10)
        threads[1].start()
        assert following.wait(10)
        assert mod.resolve_gateway_approval(SESSION_KEY, choice) == 1
        # A later teardown cannot overwrite an already committed user choice.
        mod.unregister_gateway_notify(SESSION_KEY)
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
        assert len(notifications) == 1
        assert [r["choice"] for r in results] == [choice, choice]
        assert all(r["resolved"] and not r.get("cancelled") for r in results)
        assert sum(bool(r.get("coalesced")) for r in results) == 1
    finally:
        mod.unregister_gateway_notify(SESSION_KEY)
        for thread in threads:
            if thread.ident is not None:
                thread.join(10)


def test_registered_choice_after_deadline_check_is_still_an_answer(gateway_session, monkeypatch):
    mod.register_gateway_notify(SESSION_KEY, lambda data: None)

    def late_answer(event, session_key, *, interrupt_log):
        assert mod.resolve_gateway_approval(session_key, "once") == 1
        return "timeout"

    monkeypatch.setattr(wait_mod, "_poll_event", late_answer)
    result = wait_mod._await_gateway_decision(SESSION_KEY, mod._gateway_notify_cb(SESSION_KEY), APPROVAL)
    assert result == {"resolved": True, "choice": "once", "reason": None}


@pytest.mark.parametrize("entrypoint", ["protected_write", "elicitation"])
def test_other_registered_callers_cancel_before_enqueue(gateway_session, monkeypatch, entrypoint):
    from tools.approval_prompt import request_elicitation_consent
    from tools.file_tools_write_guards import _request_protected_instruction_approval

    captured = threading.Event()
    release = threading.Event()
    notifications = []
    results = []
    original_entry = wait_mod._ApprovalEntry

    class PausedEntry(original_entry):
        def __init__(self, data):
            super().__init__(data)
            captured.set()
            assert release.wait(10)

    def run():
        if entrypoint == "protected_write":
            results.append(_request_protected_instruction_approval(["AGENTS.md"]))
        else:
            results.append(request_elicitation_consent("Allow access?", "server consent"))

    monkeypatch.setattr(wait_mod, "_ApprovalEntry", PausedEntry)
    mod.register_gateway_notify(SESSION_KEY, notifications.append)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert captured.wait(10)
        mod.unregister_gateway_notify(SESSION_KEY)
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        if entrypoint == "elicitation":
            assert results == ["cancel"]
        else:
            assert "approval was withdrawn" in results[0]
            assert "denied by the user" not in results[0]
        assert notifications == []
        assert not mod.has_blocking_approval(SESSION_KEY)
        assert not mod._session_approved.get(SESSION_KEY)
    finally:
        release.set()
        mod.unregister_gateway_notify(SESSION_KEY)
        thread.join(10)
