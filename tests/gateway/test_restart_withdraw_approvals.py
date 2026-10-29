"""Tests for explicit --withdraw-approvals policy and graceful restart approval retention (#33/A6)."""

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli import gateway as gateway_cli
from hermes_cli.subcommands.gateway import build_gateway_parser
from tests.gateway.restart_test_helpers import make_restart_runner
from tools import approval as approval_mod
from tools.approval_gateway_wait import _await_gateway_decision


def _parse_cli(*argv):
    parser = argparse.ArgumentParser()
    build_gateway_parser(
        parser.add_subparsers(dest="command"),
        cmd_gateway=lambda a: None,
        cmd_proxy=lambda a: None,
        cmd_gateway_enroll=lambda a: None,
    )
    return parser.parse_args(["gateway", *argv])


def _approval_data(cmd="test-cmd", session="test:1"):
    return {
        "command": cmd,
        "description": "test dangerous command",
        "pattern_key": cmd,
        "pattern_keys": [cmd],
        "session_key": session,
        "surface": session.split(":")[0],
    }


def _start_waiter(session_key, cmd="test-cmd", on_notify=None):
    result = {}
    started = threading.Event()

    def _notify(data):
        if on_notify:
            on_notify(data)
        started.set()

    def _run():
        result["decision"] = _await_gateway_decision(
            session_key, _notify, _approval_data(cmd, session_key)
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result, started


def test_cli_parser_and_policy_transmission_to_launchd(monkeypatch):
    """1. Policy transmission RED: parse CLI --withdraw-approvals and drive control-verb transmission."""
    args = _parse_cli("restart", "--withdraw-approvals")
    assert getattr(args, "withdraw_approvals", False) is True

    recorded_control_queries = []
    signals_sent = []

    monkeypatch.setattr(gateway_cli, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gateway_cli, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gateway_cli, "_installed_service_kind_for", lambda *a, **k: "launchd")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 42424)
    monkeypatch.setattr(gateway_cli, "refresh_launchd_plist_if_needed", lambda: True)

    def _fake_query_control(home, verb, *, params=None, timeout=None):
        recorded_control_queries.append((verb, params))
        return {
            "pid": 42424,
            "accepted": True,
            "policy": "withdraw_approvals",
            "withdraw_approvals": True,
        }

    monkeypatch.setattr("gateway.control_socket.query_gateway_control", _fake_query_control)
    import hermes_cli.gateway_launchd as gw_launchd

    monkeypatch.setattr(gw_launchd, "query_gateway_control", _fake_query_control, raising=False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals_sent.append((pid, sig)))

    gateway_cli._cmd_restart(args)

    assert len(recorded_control_queries) == 1
    verb, params = recorded_control_queries[0]
    assert verb == "restart-withdraw-approvals"
    assert params.get("policy") == "withdraw_approvals" or params.get("withdraw_approvals") is True
    assert signals_sent == []


@pytest.mark.asyncio
async def test_graceful_restart_keeps_already_pending_and_late_approvals_answerable():
    """2. Graceful keeps answers RED: real request_restart + 2 waiters stay pending through wait and are resolved."""
    if hasattr(approval_mod, "reset_approval_admission_for_tests"):
        approval_mod.reset_approval_admission_for_tests()

    s1 = "discord:test-graceful-1"
    s2 = "telegram:test-graceful-2"
    s3 = "slack:test-graceful-3"

    with approval_mod._lock:
        approval_mod._gateway_queues.pop(s1, None)
        approval_mod._gateway_queues.pop(s2, None)
        approval_mod._gateway_queues.pop(s3, None)

    runner, _adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 2.0

    t1, res1, started1 = _start_waiter(s1, cmd="cmd-1")
    t2, res2, started2 = _start_waiter(s2, cmd="cmd-2")
    assert started1.wait(5.0)
    assert started2.wait(5.0)

    # Runner requests plain graceful restart (withdraw_approvals=False); yield once so the
    # scheduled restart task starts, without waiting on the wall clock.
    runner.request_restart(detached=False, via_service=True)
    await asyncio.sleep(0)

    # Waiters must still be on the queue (NOT eagerly withdrawn)
    assert len(approval_mod.list_gateway_approvals(s1)) == 1
    assert len(approval_mod.list_gateway_approvals(s2)) == 1

    # Late-arriving prompt arriving during restart wait is also admitted and answerable
    t3, res3, started3 = _start_waiter(s3, cmd="cmd-3")
    assert started3.wait(5.0)
    assert len(approval_mod.list_gateway_approvals(s3)) == 1

    # Resolve all three
    approval_mod.resolve_gateway_approval(s1, "always")
    approval_mod.resolve_gateway_approval(s2, "deny", reason="not allowed")
    approval_mod.resolve_gateway_approval(s3, "session")

    t1.join(2.0)
    t2.join(2.0)
    t3.join(2.0)

    assert not t1.is_alive() and not t2.is_alive() and not t3.is_alive()
    assert res1["decision"]["resolved"] is True
    assert res1["decision"]["choice"] == "always"
    assert res2["decision"]["resolved"] is True
    assert res2["decision"]["choice"] == "deny"
    assert res3["decision"]["resolved"] is True
    assert res3["decision"]["choice"] == "session"


def test_withdraw_mode_late_arrival_fence():
    """3. Withdraw mode late-arrival RED: after withdrawal, late arrivals fail closed immediately."""
    s = "discord:test-late-arrival"
    with approval_mod._lock:
        approval_mod._gateway_queues.pop(s, None)

    cause = "the gateway restarted while this approval was pending"
    approval_mod.begin_approval_withdrawal(cause)

    notify_called = []
    res = {}
    done = threading.Event()

    def _notify(data):
        notify_called.append(data)

    def _run():
        res["decision"] = _await_gateway_decision(s, _notify, _approval_data("late-cmd", s))
        done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert done.wait(2.0), "late approval must return promptly without blocking"
    thread.join(1.0)

    decision = res["decision"]
    assert decision["resolved"] is False
    assert decision["choice"] is None
    assert decision.get("cancelled") == cause
    assert notify_called == [], "no prompt should be published after fence is closed"
    assert approval_mod.list_gateway_approvals(s) == []


def test_fence_lifetime_bound_to_process_and_reset():
    """4. Fence lifetime: bound to process/owner; retiring owner never reopens; test reset works."""
    s = "discord:test-fence-lifetime"
    with approval_mod._lock:
        approval_mod._gateway_queues.pop(s, None)

    cause = "the gateway restarted while this approval was pending"
    approval_mod.begin_approval_withdrawal(cause)

    # Retiring owner never reopens:
    dec = _await_gateway_decision(s, lambda d: None, _approval_data("cmd1", s))
    assert dec["resolved"] is False
    assert dec.get("cancelled") == cause

    # Test reset:
    approval_mod.reset_approval_admission_for_tests()

    t, res, started = _start_waiter(s, cmd="cmd2")
    assert started.wait(5.0)
    assert len(approval_mod.list_gateway_approvals(s)) == 1

    approval_mod.resolve_gateway_approval(s, "once")
    t.join(2.0)
    assert res["decision"]["resolved"] is True
    assert res["decision"]["choice"] == "once"


@pytest.mark.asyncio
async def test_safety_negative_controls(monkeypatch):
    """5. Safety negative controls: cap honoured, unsupported targets refuse non-zero without signals, --force intact."""
    # A. Real non-approval work still honours the cap
    runner, _adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 0.3
    runner._active_work_count = lambda: 1
    runner._awaitable_work_count = lambda: 1
    runner._wedged_agent_count = lambda: 0
    runner._restart_safe_cron_count = lambda: 0
    loop = asyncio.get_running_loop()
    begun = loop.time()
    drained = await runner._await_active_work_before_restart()
    elapsed = loop.time() - begun
    assert drained is False
    assert elapsed >= 0.25

    # B. --withdraw-approvals combined with --all refuses with non-zero exit, issues no signal/kill
    args_all = SimpleNamespace(withdraw_approvals=True, all=True, system=False, force=False)
    monkeypatch.setattr(gateway_cli, "_refuse_from_inside_gateway", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc_info:
        gateway_cli._cmd_restart(args_all)
    assert exc_info.value.code != 0

    # C. --withdraw-approvals combined with --system refuses with non-zero exit
    args_sys = SimpleNamespace(withdraw_approvals=True, all=False, system=True, force=False)
    with pytest.raises(SystemExit) as exc_info:
        gateway_cli._cmd_restart(args_sys)
    assert exc_info.value.code != 0

    # D. --withdraw-approvals on systemd backend refuses with non-zero exit
    args_systemd = SimpleNamespace(withdraw_approvals=True, all=False, system=False, force=False)
    monkeypatch.setattr(gateway_cli, "_installed_service_kind_for", lambda *a, **k: "systemd")
    with pytest.raises(SystemExit) as exc_info:
        gateway_cli._cmd_restart(args_systemd)
    assert exc_info.value.code != 0

    # E. Unsupported ACK refuses with non-zero exit and issues no signal/kill
    args_bad_ack = SimpleNamespace(withdraw_approvals=True, all=False, system=False, force=False)
    monkeypatch.setattr(gateway_cli, "_installed_service_kind_for", lambda *a, **k: "launchd")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 9911)
    monkeypatch.setattr(gateway_cli, "refresh_launchd_plist_if_needed", lambda: True)
    monkeypatch.setattr("gateway.control_socket.query_gateway_control", lambda *a, **k: None)
    import hermes_cli.gateway_launchd as gw_launchd

    monkeypatch.setattr(gw_launchd, "query_gateway_control", lambda *a, **k: None, raising=False)
    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

    with pytest.raises(SystemExit) as exc_info:
        gateway_cli._cmd_restart(args_bad_ack)
    assert exc_info.value.code != 0
    assert signals == []

    # F. --force still routes the named-profile escape unchanged
    args_force = _parse_cli("restart", "--force")
    assert args_force.force is True
    assert getattr(args_force, "withdraw_approvals", False) is False
