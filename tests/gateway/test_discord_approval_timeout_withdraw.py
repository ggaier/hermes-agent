"""#33 regression: a Discord approval card that expires must end the backend wait.

Verified production failure (2026-10-28/29): the Discord ``ExecApprovalView`` timed out after
``approvals.discord_prompt_timeout`` (900s) and only disabled its buttons; the agent thread kept
waiting ``approvals.timeout`` (86400s), so three threads stayed "running" overnight and the
restart drain waited for them.

Two behaviours are pinned here:
  * expiry withdraws THAT request id fail-closed (never approves, never runs, never leaves a
    newer queued command answerable by the stale card), and
  * a click may only settle its own request id — a stale card must not deny a newer command.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, ExecApprovalView
from tools import approval as approval_mod
from tools.approval_gateway_wait import _ApprovalEntry, _await_gateway_decision, _poll_event

KEY = "discord:555"


@pytest.fixture(autouse=True)
def _fake_discord_color_has_every_named_colour(monkeypatch):
    """The shared discord stub omits a few named colours these paths stamp on the card."""
    from plugins.platforms.discord import adapter as adapter_mod

    color = adapter_mod.discord.Color
    if hasattr(color, "dark_grey"):
        return
    monkeypatch.setattr(adapter_mod.discord, "Color", SimpleNamespace(
        green=lambda: 1, grey=lambda: 2, blurple=lambda: 2, red=lambda: 3, purple=lambda: 5,
        dark_grey=lambda: 6, orange=lambda: 7, gold=lambda: 8, greyple=lambda: 9), raising=False)



def _reset():
    with approval_mod._lock:
        approval_mod._gateway_queues.pop(KEY, None)


def _entry(request_id: str, command: str) -> _ApprovalEntry:
    return _ApprovalEntry({
        "command": command, "description": "dangerous command",
        "pattern_key": command, "pattern_keys": [command],
        "session_key": KEY, "surface": "discord", "request_id": request_id,
    })


def _enqueue(*entries):
    with approval_mod._lock:
        approval_mod._gateway_queues.setdefault(KEY, []).extend(entries)


def _interaction(uid="42"):
    async def _noop(*_a, **_k):
        return None
    return SimpleNamespace(
        user=SimpleNamespace(id=uid, display_name="operator", roles=[]),
        response=SimpleNamespace(send_message=AsyncMock(side_effect=_noop),
                                 edit_message=AsyncMock(side_effect=_noop)),
        message=SimpleNamespace(embeds=[]),
    )


def _view(request_id):
    return ExecApprovalView(session_key=KEY, allowed_user_ids={"42"}, request_id=request_id)


def test_on_timeout_withdraws_exact_request_and_wakes_backend_waiter():
    """The card's expiry is the end of the wait — fail-closed, promptly."""
    _reset()
    entry = _entry("req-a", "rm -rf /tmp/card33-a")
    _enqueue(entry)
    waiter = {}
    started = threading.Event()

    def _wait():
        started.set()
        waiter["state"] = _poll_event(entry.event, KEY, interrupt_log="waiter")

    thread = threading.Thread(target=_wait, daemon=True)
    thread.start()
    assert started.wait(2.0), "precondition: the backend waiter thread started"
    assert thread.is_alive(), "precondition: the backend waiter is still blocked"

    try:
        asyncio.run(_view("req-a").on_timeout())
        thread.join(2.0)
        assert not thread.is_alive(), (
            "backend wait still blocked after the Discord card expired — this is the 24h hang")
        assert waiter["state"] == "set"
        assert entry.cancelled == "timeout", "expiry must be recorded as a timeout, not a user deny"
        assert entry.result is None, "an expired card must never approve the command"
        assert approval_mod.list_gateway_approvals(KEY) == []
    finally:
        entry.event.set()


@pytest.mark.asyncio
async def test_stale_card_cannot_answer_a_newer_queued_command():
    """A old card whose request is gone must not resolve the entry now at the queue head."""
    _reset()
    stale = _entry("req-old", "rm -rf /tmp/card33-old")
    newer = _entry("req-new", "rm -rf /tmp/card33-new")
    _enqueue(stale, newer)
    with approval_mod._lock:
        approval_mod._gateway_queues[KEY].remove(stale)

    await _view("req-old")._resolve(_interaction(), "deny", None, "platform.discord.approval.resolved_deny")

    assert newer.result is None, "a stale card denied a newer command that the user never saw"
    assert newer.event.is_set() is False
    assert [d["request_id"] for d in approval_mod.list_gateway_approvals(KEY)] == ["req-new"]


@pytest.mark.asyncio
async def test_click_resolves_only_its_own_request():
    """Positive control: with two queued commands, a click settles exactly the card's request."""
    _reset()
    first = _entry("req-1", "rm -rf /tmp/card33-1")
    second = _entry("req-2", "rm -rf /tmp/card33-2")
    _enqueue(first, second)

    await _view("req-2")._resolve(_interaction(), "deny", None, "platform.discord.approval.resolved_deny")

    assert second.result == "deny"
    assert second.event.is_set() is True
    assert first.result is None, "the older queued command was answered by the newer card"
    assert first.event.is_set() is False


def test_prompt_expiry_settles_as_timeout_so_the_notice_is_posted():
    """End-to-end through the real wait entry: a resolved-with-no-choice wait is a timeout."""
    _reset()
    settled = []

    def _settle(reason):
        settled.append(reason)

    def _notify(data):
        rid = data["request_id"]

        def _expire():
            approval_mod.withdraw_gateway_approval(KEY, rid, "timeout")

        approval_mod.register_gateway_settle(KEY, rid, _settle)
        threading.Timer(0.4, _expire).start()

    result = _await_gateway_decision(KEY, _notify, {
        "command": "rm -rf /tmp/card33-e2e", "description": "dangerous command",
        "pattern_key": "rm", "pattern_keys": ["rm"], "session_key": KEY, "surface": "discord",
    })

    assert result["resolved"] is False
    assert result["choice"] is None
    assert result.get("cancelled") == "timeout"
    assert settled == ["timeout"], (
        "the wait ended with no choice, so the settle hook must report 'timeout' "
        f"(that is what posts the honest timeout notice); got {settled!r}")


@pytest.mark.asyncio
async def test_timeout_without_request_id_leaves_queue_untouched():
    """Legacy callers that never learned a request id must not withdraw someone else's entry."""
    _reset()
    other = _entry("req-other", "rm -rf /tmp/card33-other")
    _enqueue(other)

    await ExecApprovalView(session_key=KEY, allowed_user_ids={"42"}).on_timeout()

    assert other.result is None
    assert other.event.is_set() is False
    assert [d["request_id"] for d in approval_mod.list_gateway_approvals(KEY)] == ["req-other"]


def test_expiry_withdraws_even_when_the_card_edit_hangs():
    """The embed edit is a network call: a hung edit must not keep the backend waiting."""
    _reset()
    entry = _entry("req-hang", "rm -rf /tmp/card33-hang")
    _enqueue(entry)

    class _HangingMessage:
        embeds: list = []

        async def edit(self, **_kwargs):
            await asyncio.Event().wait()  # never completes

    view = _view("req-hang")
    view._message = _HangingMessage()

    async def _expire():
        await asyncio.wait_for(view.on_timeout(), timeout=2.0)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(_expire())

    assert entry.event.is_set(), "withdraw waited on the Discord message edit instead of running first"
    assert entry.cancelled == "timeout"
    assert entry.result is None
