"""#33/A6: a restart must not sit out the whole cap behind a human approval.

Production symptom (2026-10-29): `hermes gateway restart --force` waited 2000+ s. The after-turn
wait counts a chat turn whose only activity is "waiting for user approval" as awaitable work, and
that turn heartbeats, so it never reads as wedged: the wait blocks for the entire
`restart_after_turn_timeout` cap while nobody is going to answer the card.

The bounded repair is fail-closed, not a kill shortcut: at restart, pending approvals are withdrawn
with a restart cause, so each waiting tool denies itself, its turn ends, and the drain proceeds.
Nothing is replayed and no approval is granted.
"""

import asyncio
import threading
import time

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tools import approval as approval_mod
from tools.approval_gateway_wait import _await_gateway_decision

KEY = "discord:restart-bound"


def _reset():
    if hasattr(approval_mod, "reset_approval_admission_for_tests"):
        approval_mod.reset_approval_admission_for_tests()
    with approval_mod._lock:
        approval_mod._gateway_queues.pop(KEY, None)


def _data(command="rm -rf /tmp/card33-restart"):
    return {"command": command, "description": "dangerous command",
            "pattern_key": command, "pattern_keys": [command],
            "session_key": KEY, "surface": "discord"}


def _start_waiter(settled=None):
    """A real backend approval wait, as a chat turn would block the drain."""
    result, started = {}, threading.Event()
    settled = settled if settled is not None else []

    def _notify(data):
        approval_mod.register_gateway_settle(KEY, data["request_id"], settled.append)
        started.set()

    def _run():
        result["decision"] = _await_gateway_decision(KEY, _notify, _data())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert started.wait(5.0), "precondition: the approval prompt reached the queue"
    return thread, result, settled


@pytest.mark.asyncio
async def test_restart_wait_does_not_sit_out_the_cap_behind_an_approval():
    _reset()
    runner, _adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 60.0  # the production cap is minutes; the test must not be
    thread, result, settled = _start_waiter()
    runner._active_work_count = lambda: 1 if thread.is_alive() else 0
    runner._running_agent_count = runner._active_work_count
    runner.request_restart(withdraw_approvals=True)
    loop = asyncio.get_running_loop()
    begun = loop.time()
    try:
        await asyncio.wait_for(runner._await_active_work_before_restart(), timeout=20)
    finally:
        elapsed = loop.time() - begun
    thread.join(2.0)

    assert elapsed < 5.0, (
        "the restart wait held for an approval nobody can answer "
        f"({elapsed:.1f}s; the cap was never the point — the turn can never finish)")
    assert not thread.is_alive(), "the waiting tool must be released, not left blocked through the restart"
    decision = result["decision"]
    assert decision["choice"] is None, "a restart must never approve the pending command"
    assert decision.get("cancelled"), "the withdrawal cause must reach the tool, not a silent deny"
    assert approval_mod.list_gateway_approvals(KEY) == []
    assert settled and settled[0] != "resolved"


@pytest.mark.asyncio
async def test_after_a_withdrawn_approval_the_session_is_not_wedged():
    """A2 shape: once the wait ends, the next approval in that session is admitted immediately."""
    _reset()
    runner, _adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 60.0
    thread, _result, _settled = _start_waiter()
    runner._active_work_count = lambda: 1 if thread.is_alive() else 0
    runner._running_agent_count = runner._active_work_count
    runner.request_restart(withdraw_approvals=True)
    try:
        await asyncio.wait_for(runner._await_active_work_before_restart(), timeout=20)
        thread.join(2.0)
        assert not thread.is_alive()
        # The replacement process starts with a fresh fence (simulated via reset):
        approval_mod.reset_approval_admission_for_tests()
        # The queue is empty again: a later request in the same session is admitted, not blocked
        # behind the dead prompt.
        later = {}
        began = time.monotonic()
        def _notify(data):
            approval_mod.resolve_gateway_approval(KEY, "deny", request_id=data["request_id"])
        later["decision"] = _await_gateway_decision(KEY, _notify, _data("rm -rf /tmp/card33-later"))
        assert time.monotonic() - began < 5.0
        assert later["decision"]["choice"] == "deny"
    finally:
        _reset()


@pytest.mark.asyncio
async def test_a_real_running_turn_still_holds_the_restart_wait():
    """Positive control: only approval waits are released early; genuine work is still awaited."""
    _reset()
    runner, _adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 0.3
    runner._active_work_count = lambda: 1
    runner._running_agent_count = lambda: 1
    begun = asyncio.get_running_loop().time()
    drained = await runner._await_active_work_before_restart()
    elapsed = asyncio.get_running_loop().time() - begun
    assert drained is False
    assert elapsed >= 0.25, "the cap must still apply to work that is not an approval wait"
