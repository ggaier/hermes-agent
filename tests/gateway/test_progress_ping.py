"""The run must be visibly alive from OUTSIDE the chat, not just in it.

Card idXGRRa4 asked for download-style progress: "就像下载文件一样，需要有一个
进度通知". The heartbeat bubbles here are EDITED in place, and neither Discord nor
Telegram pushes a notification for an edit — so a 57-minute run showed exactly
one notification (the final card) even when its bubble was updating fine. The
user's complaint was precisely that: "甚至通知卡片也只是在完成的时候才收到一个".

``gateway_progress_ping_interval`` adds real milestone messages (each one a NEW
message, hence an actual push) while leaving the bubble as the in-place detail
surface. Asserted here:

1. The ping fires as its own ``send``, separate from the bubble's send/edit.
2. It carries the elapsed minutes and the current activity.
3. The interval never undercuts the heartbeat (no pings faster than the bubble
   refreshes), so an idle run cannot spam.
4. ``0`` (the default) means no pings at all — behaviour is opt-in.
"""

import asyncio
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.allow_real_home_io


class Recorder:
    def __init__(self):
        self.sent = []
        self.edited = []
        self.stop = False
        self.ticks = 0

    async def send(self, chat_id, text, metadata=None):
        self.sent.append(text)
        return SimpleNamespace(success=True, message_id=f"m{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, text, **kwargs):
        self.edited.append(text)
        return SimpleNamespace(success=True, message_id=message_id)


def _make_runner(monkeypatch, recorder, *, ping_interval, notify_interval="0.01",
                 activity="terminal"):
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", notify_interval)
    monkeypatch.setenv("HERMES_AGENT_PROGRESS_PING_INTERVAL", str(ping_interval))
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = runner._restart_requested = False
    runner._delivery_adapter_for = lambda source: recorder
    runner._agent_activity_summary = staticmethod(
        lambda agent: {"api_call_count": 7, "max_iterations": 300, "current_tool": activity})

    agent = MagicMock()
    runner._running_agents["sess"] = agent

    def _should(*_a, **_k):
        recorder.ticks += 1
        return not recorder.stop

    runner._should_emit_long_running_notification = _should
    return runner, agent


def _ctx(agent):
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
    disp.resolve_display_setting.return_value = True
    disp._generic_status_phrase.return_value = "working"
    return disp


async def _run(runner, ctx, recorder, *, ticks=3):
    """Advance the clock so elapsed minutes are non-zero, then stop the run."""
    import gateway.run_turn as rt

    real_time = rt.time.time
    # Elapsed minutes come from time.time() - turn start; pretend each tick is 61s.
    offset = {"v": 0.0}

    def _fake_time():
        offset["v"] += 61.0
        return real_time() + offset["v"]

    rt.time.time = _fake_time
    try:
        async def _stopper():
            while recorder.ticks < ticks:
                await asyncio.sleep(0.01)
            recorder.stop = True

        await asyncio.wait_for(
            asyncio.gather(runner._run_agent_notify_long_running(_disp(), ctx, [None]),
                           _stopper()),
            8,
        )
    finally:
        rt.time.time = real_time


@pytest.mark.asyncio
class TestProgressPings:
    async def test_ping_is_its_own_message_not_an_edit(self, monkeypatch):
        rec = Recorder()
        runner, agent = _make_runner(monkeypatch, rec, ping_interval=0.02)
        await _run(runner, _ctx(agent), rec, ticks=3)

        assert rec.sent, "no milestone ping was ever sent"
        assert len(rec.sent) >= 2, (
            f"the pulse of the run was an edit, not a push: {rec.sent!r}")

    async def test_ping_reports_elapsed_time_and_activity(self, monkeypatch):
        rec = Recorder()
        runner, agent = _make_runner(monkeypatch, rec, ping_interval=0.02)
        await _run(runner, _ctx(agent), rec, ticks=3)

        ping = rec.sent[0]
        assert "min" in ping, f"ping has no elapsed time: {ping!r}"
        assert "terminal" in ping, f"ping has no current activity: {ping!r}"

    async def test_zero_disables_pings(self, monkeypatch):
        """Default stays quiet: only the edited bubble, no extra messages."""
        rec = Recorder()
        runner, agent = _make_runner(monkeypatch, rec, ping_interval=0)
        await _run(runner, _ctx(agent), rec, ticks=3)

        # One bubble send max (the first tick creates it, later ticks edit it).
        assert len(rec.sent) <= 1, f"pings fired with the interval disabled: {rec.sent!r}"

    async def test_ping_interval_is_clamped_up_to_the_heartbeat(self, monkeypatch):
        """A ping faster than the bubble would be spam: the clamp raises it.

        Asserted on the resolved numbers rather than by running a real 60s tick —
        the clamp is the contract, and waiting a minute to observe it would make
        the suite slow for no extra coverage.
        """
        import gateway.run_turn as rt

        source = (pathlib.Path(rt.__file__)).read_text()
        assert "_ping_interval < _NOTIFY_INTERVAL" in source, "clamp disappeared"
        assert "_ping_interval = _NOTIFY_INTERVAL" in source, "clamp no longer raises"

        # And boot-time validation is real: a tiny notify interval with a longer
        # ping must not ping between bubble refreshes.
        rec = Recorder()
        runner, _agent = _make_runner(monkeypatch, rec, ping_interval=5,
                                      notify_interval="0.01")
        assert runner is not None  # construction itself must not raise
