"""RED/GREEN: the long-running heartbeat must survive a platform reconnect.

Why this file exists (2026-10-22, card idXGRRa4 follow-up):

``_run_agent_notify_long_running`` captured ``self._delivery_adapter_for(source)``
ONCE at the start of the turn. When the Discord websocket dropped mid-run the
gateway installed a NEW adapter and closed the old one's session, so every later
heartbeat tick did:

    edit_message(...) on the dead adapter -> RuntimeError("Session is closed")

The code turned that into ``_notify_res = None`` and then "recovered" by calling
``send(...)`` on the *same* dead adapter. That raised too, and the outer
``except Exception`` logged it at DEBUG — so the run went silent for its
remaining half hour with nothing in the logs above debug level.

User-visible symptom: one "⏳ Working" bubble that stopped updating, then only
the final card at the end.

Contract asserted here:

1. A tick after an adapter swap uses the *current* adapter, not the one
   captured at turn start.
2. When an in-place edit fails, the fallback bubble goes out through a freshly
   resolved adapter, so the user still gets progress.
3. When both edit and send fail on the dead adapter, the next tick re-resolves
   and delivers — a dead adapter must not silence the rest of the run.
4. A heartbeat that can no longer reach the chat is logged above debug level,
   so a silent run is diagnosable after the fact.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# This checkout lives directly under the real hermes home (~/.hermes/hermes-agent),
# so importing ``gateway.run`` makes ``pm.environments._payload_manifest`` probe
# ``~/.hermes/manifest.json`` and the universal real-home tripwire fails the test
# before any heartbeat code runs. Pre-existing on this host: tests/gateway/
# test_busy_session_ack.py and test_display_config.py fail the same way on
# untouched HEAD. This module does no file I/O of its own (all adapters are
# mocks), so the guard is opted out here rather than weakening the assertion.
pytestmark = pytest.mark.allow_real_home_io


def _ok(message_id=None):
    return SimpleNamespace(success=True, message_id=message_id)


def _fail():
    return SimpleNamespace(success=False, message_id=None)


class Controller:
    """Shared state for the fake transports.

    ``current`` is the adapter the runner must resolve to right now — adapters
    advance it after every call, which models "the gateway installed a new
    adapter" without depending on how many times the code under test resolves one.
    ``done`` ends the run; ``_should_emit_long_running_notification`` is called
    once per tick AND again inside the fallback branch, so a call-count based
    stop would break out before the fallback ever sends.
    """

    def __init__(self, current=None):
        self.current = current
        self.done = False
        self.log = []

    def should_emit(self):
        return not self.done


def _adapter(controller, label, *, edit=("ok", None), send=("ok", None),
             then=None, then_send=None, finish=False):
    """An adapter whose edit/send behave as told.

    ``then`` (any call) / ``then_send`` (a ``send`` call only) install a new
    adapter as ``controller.current`` — the fake of "the gateway reconnected and
    swapped the adapter while this one was being used".
    """
    adapter = MagicMock()
    adapter.label = label

    def _make(kind, behaviour):
        status, message_id = behaviour

        async def _call(*_a, **_k):
            controller.log.append((label, kind))
            try:
                if status == "raise":
                    raise RuntimeError("Session is closed")
                return _ok(message_id) if status == "ok" else _fail()
            finally:
                handover = then_send if (kind == "send" and then_send is not None) else then
                if handover is not None:
                    controller.current = handover
                # `finish` ends the run even when the call raised: the scenario, not
                # the transport's health, decides how many ticks this test needs.
                if finish:
                    controller.done = True

        return _call

    adapter.edit_message = AsyncMock(side_effect=_make("edit", edit))
    adapter.send = AsyncMock(side_effect=_make("send", send))
    return adapter


def _make_runner(monkeypatch, controller):
    """Runner that always resolves to ``controller.current`` — i.e. the live adapter."""
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = runner._restart_requested = False

    runner._delivery_adapter_for = lambda source: controller.current
    runner._agent_activity_summary = staticmethod(lambda agent: None)
    runner._should_emit_long_running_notification = lambda *a, **k: controller.should_emit()

    agent = MagicMock()
    runner._running_agents["sess"] = agent
    return runner, agent


def _make_ctx(agent):
    from gateway.turn_context import TurnContext

    ctx = TurnContext(
        source=SimpleNamespace(chat_id="chan", platform="discord"),
        session_key="sess",
    )
    ctx.agent_holder[0] = agent
    return ctx


def _disp():
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False
    disp._generic_status_phrase.return_value = "working"
    return disp


async def _run_heartbeat(runner, ctx):
    await asyncio.wait_for(runner._run_agent_notify_long_running(_disp(), ctx, [None]), 5)


@pytest.mark.asyncio
class TestHeartbeatSurvivesAdapterSwap:
    async def test_tick_after_swap_uses_the_current_adapter(self, monkeypatch):
        """The turn starts on a dead adapter; the heartbeat must still land."""
        controller = Controller()
        live = _adapter(controller, "live", send=("ok", "hb-1"), finish=True)
        dead = _adapter(controller, "dead", edit=("raise", None),
                        send=("raise", None), then=live)
        controller.current = dead
        runner, agent = _make_runner(monkeypatch, controller)

        await _run_heartbeat(runner, _make_ctx(agent))

        assert ("live", "send") in controller.log, (
            f"heartbeat never reached the reconnected adapter: {controller.log!r}")

    async def test_edit_failure_falls_back_to_a_fresh_adapter(self, monkeypatch):
        """An edit that fails on a swapped-out adapter must not swallow the tick."""
        controller = Controller()
        live = _adapter(controller, "live", send=("ok", "hb-2"), finish=True)
        dead = _adapter(controller, "dead", edit=("raise", None),
                        send=("raise", None), then=live)
        first = _adapter(controller, "first", send=("ok", "hb-1"), then=dead)
        controller.current = first
        runner, agent = _make_runner(monkeypatch, controller)

        await _run_heartbeat(runner, _make_ctx(agent))

        assert ("dead", "edit") in controller.log, "the swap was never exercised"
        assert ("live", "send") in controller.log, (
            f"fallback bubble never used a live adapter: {controller.log!r}")

    async def test_total_failure_on_one_adapter_does_not_end_the_heartbeat(self, monkeypatch):
        """Both edit and send die on one tick; the next tick must still get through."""
        controller = Controller()
        live = _adapter(controller, "live", edit=("ok", "hb-1"), finish=True)
        # No handover on the edit: the adapter stays dead for the whole tick, so
        # the tick genuinely fails end-to-end. The swap lands after that send.
        dead = _adapter(controller, "dead", edit=("raise", None),
                        send=("raise", None), then_send=live)
        first = _adapter(controller, "first", send=("ok", "hb-1"), then=dead)
        controller.current = first
        runner, agent = _make_runner(monkeypatch, controller)

        await _run_heartbeat(runner, _make_ctx(agent))

        assert ("dead", "send") in controller.log, (
            f"the fully-unreachable tick was never exercised: {controller.log!r}")
        assert ("live", "edit") in controller.log, (
            f"heartbeat stopped for good after one unreachable tick: {controller.log!r}")

    async def test_unreachable_heartbeat_is_logged_above_debug(self, monkeypatch, caplog):
        """A silent run must be diagnosable: the drop is a WARNING, not debug noise."""
        controller = Controller()
        dead = _adapter(controller, "dead", edit=("raise", None),
                        send=("raise", None), finish=True)
        controller.current = dead
        runner, agent = _make_runner(monkeypatch, controller)

        caplog.set_level(logging.WARNING)
        await _run_heartbeat(runner, _make_ctx(agent))

        assert any(
            "heartbeat" in record.message.lower() and record.levelno >= logging.WARNING
            for record in caplog.records
        ), (f"heartbeat delivery failure was not logged above debug: "
            f"{[(r.levelno, r.message) for r in caplog.records]!r}")
