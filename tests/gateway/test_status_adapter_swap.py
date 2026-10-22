"""RED/GREEN: tool-progress and status bubbles must survive a platform reconnect.

Companion to ``test_heartbeat_adapter_swap.py``, same incident (2026-10-22):
the Discord websocket dropped at 15:24:54, and from then until the final card
at 15:58 the user saw NOTHING -- no heartbeat, and no tool-progress bubble
either, even though the run kept calling tools (162 API calls).

``_run_agent_bind_turn_wiring`` stored ``turn_ctx._status_adapter =
self._delivery_adapter_for(source)`` ONCE. Every later progress/status send
went through that captured instance, whose aiohttp session the reconnect had
closed. The sends failed and -- like the heartbeat's -- were swallowed.

A reconnected platform installs a NEW adapter object
(``RunAdaptersMixin._install_reconnected_adapter`` -> ``_publish_primary_adapter``
overwrites ``self.adapters[platform]``), so the fix is to re-resolve instead of
capturing. ``_status_adapter`` stays the name every caller reads, so it becomes a
property that re-resolves per access and falls back to the last known-good
instance (a resolution failure must not turn progress off silently).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.allow_real_home_io


def _ok():
    return SimpleNamespace(success=True, message_id="m-1")


def _dead():
    dead = MagicMock()
    dead.send = AsyncMock(side_effect=RuntimeError("Session is closed"))
    dead.edit_message = AsyncMock(side_effect=RuntimeError("Session is closed"))
    return dead


class _LiveAdapter:
    """A transport that works; records what actually reached the chat."""

    def __init__(self):
        self.sent = []
        self.edits = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append((chat_id, text))
        return _ok()

    async def send_or_update_status(self, chat_id, status_key, content, metadata=None):
        self.sent.append((chat_id, content))
        return _ok()

    async def edit_message(self, chat_id, message_id, text, **kwargs):
        self.edits.append((message_id, text))
        return _ok()


def _live():
    return _LiveAdapter()


def _ctx(adapter_holder):
    from gateway.turn_context import TurnContext

    ctx = TurnContext(
        source=SimpleNamespace(chat_id="chan", platform="discord"),
        session_key="sess",
    )
    ctx._resolve_status_adapter = lambda: adapter_holder[0]
    return ctx


class TestStatusAdapterReResolves:
    def test_reads_the_current_adapter_after_a_swap(self):
        """The whole point: the bubble must land on the adapter that is live NOW."""
        holder = [_dead()]
        ctx = _ctx(holder)
        ctx._status_adapter = holder[0]  # what the turn captured at start
        assert ctx.current_status_adapter() is holder[0]

        holder[0] = _live()
        assert ctx.current_status_adapter() is holder[0], (
            "status adapter stayed pinned to the pre-reconnect instance")

    def test_falls_back_to_the_last_good_adapter_when_resolution_fails(self):
        """A resolver error must not disable progress for the rest of the turn."""
        holder = [_live()]
        ctx = _ctx(holder)
        ctx._status_adapter = holder[0]
        first = ctx.current_status_adapter()

        def _boom():
            raise RuntimeError("adapter registry unavailable")

        ctx._resolve_status_adapter = _boom
        assert ctx.current_status_adapter() is first, "progress lost its adapter on a resolver error"

    def test_keeps_working_without_a_resolver(self):
        """Hand-built contexts carry only the field: no resolver, same behaviour."""
        from gateway.turn_context import TurnContext

        adapter = _live()
        ctx = TurnContext(source=SimpleNamespace(chat_id="chan"), session_key="sess")
        ctx._status_adapter = adapter
        assert ctx.current_status_adapter() is adapter

    def test_reports_none_when_nothing_was_resolved(self):
        ctx = _ctx([None])
        assert ctx.current_status_adapter() is None


@pytest.mark.asyncio
class TestProgressSendPath:
    async def test_status_send_goes_through_the_live_adapter(self):
        """End-to-end of the seam callers use: a swapped adapter must get the message."""
        holder = [_dead()]
        ctx = _ctx(holder)
        ctx._status_chat_id = "chan"
        holder[0] = _live()

        from gateway.run import _send_or_update_status_coro

        await _send_or_update_status_coro(
            ctx.current_status_adapter(), ctx._status_chat_id, "tool", "running", None)
        assert holder[0].sent == [("chan", "running")], (
            f"progress never reached the live adapter: {holder[0].sent!r}")
