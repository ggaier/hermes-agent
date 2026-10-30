"""The reconnect scan's yield budget must never cost the NEWEST messages (#39 N4).

Production evidence: an outage left one message in the thread plus ~56 thread messages
inside the 1h window. The scan reported ``scanned=50 missed=0`` (its limit is 50) and the
newest message — the one the user actually sent during the outage — was never yielded:
no ledger row at all. The budget was split round-robin across sources, and inside each
source history is delivered oldest-first, so the newest message was always last in line
and the cap always cut it.

Contract these tests pin:
  * the retained ``limit`` messages are the newest ones, delivered oldest-first;
  * a truncated scan is a coverage gap, never a clean ``success``.
"""

import datetime as dt
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=1, secondary=2, danger=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import discord
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

BASE = datetime(2026, 10, 30, 16, 0, tzinfo=dt.UTC)


def _msg(message_id, channel, *, minutes=0.0, author_id=42):
    return SimpleNamespace(
        id=message_id,
        content=f"msg {message_id}",
        reactions=[],
        author=SimpleNamespace(id=author_id, bot=False, display_name="Emo", name="emo"),
        channel=channel,
        guild=getattr(channel, "guild", None),
        created_at=BASE + dt.timedelta(minutes=minutes),
        attachments=[],
        mentions=[],
        reference=None,
        type=discord.MessageType.default,
    )


class HistoryChannel:
    """Mimics discord.py history(): newest-first unless oldest_first=True."""

    def __init__(self, channel_id, count, *, start_minute=0.0, parent_id=None):
        self.id = channel_id
        self.parent_id = parent_id
        self.name = f"chan-{channel_id}"
        self.guild = SimpleNamespace(id=777, name="emo")
        self.topic = None
        self.threads = []
        self._messages = [
            _msg(channel_id * 1000 + i, self, minutes=start_minute + i) for i in range(count)
        ]

    def history(self, **kwargs):
        limit = kwargs.get("limit", 100)
        ordered = list(self._messages)
        if not kwargs.get("oldest_first"):
            ordered = list(reversed(ordered))
        window = [m for m in ordered if kwargs.get("after") is None or m.created_at >= kwargs["after"]]
        sliced = window[:limit]

        async def _gen():
            for message in sliced:
                yield message

        return _gen()


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=999, bot=True, display_name="Hermes", name="hermes"),
        get_channel=lambda _id: None,
    )
    adapter._ready_event.set()
    adapter._handle_message = AsyncMock(return_value=True)
    adapter._post_reply = AsyncMock(return_value=SimpleNamespace(id="1"))
    return adapter


def _wire(adapter, channels):
    mapping = {c.id: c for c in channels}
    adapter._client.get_channel = lambda channel_id: mapping.get(channel_id)
    adapter._client.fetch_channel = AsyncMock(side_effect=lambda i: mapping[i])
    return mapping


@pytest.mark.asyncio
async def test_global_cap_keeps_newest_message_of_the_busiest_source(adapter, monkeypatch):
    """The production N4 shape: quiet channel + one very busy source, budget smaller than the
    backlog. The newest message is the one the user sent during the outage — it must survive."""
    busy = HistoryChannel(200, 56)
    quiet = HistoryChannel(100, 1)
    _wire(adapter, [busy, quiet])
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 50)

    got = [m.id async for m in adapter._iter_missed_message_backfill_candidates({"100", "200"})]

    assert len(got) == 50
    newest = busy._messages[-1].id
    assert newest in got, "the newest message in the busiest channel was cut by the yield budget"
    assert got[-1] == newest
    assert got == sorted(got)  # oldest-first delivery preserved


@pytest.mark.asyncio
async def test_truncated_scan_records_a_coverage_gap(adapter, monkeypatch):
    """A scan that could not reach every candidate in the window must not read as success."""
    busy = HistoryChannel(200, 56)
    quiet = HistoryChannel(100, 1)
    _wire(adapter, [busy, quiet])
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 50)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"100", "200"})
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=False))

    await adapter._run_missed_message_backfill()

    def _row(conn):
        return conn.execute(
            "SELECT status, scanned, error FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    status, scanned, error = adapter._with_discord_recovery_db(_row)
    assert scanned == 50
    assert status == "partial", "a truncated scan reported a clean success"
    assert "truncated" in (error or "").lower()


@pytest.mark.asyncio
async def test_scan_that_fits_the_budget_stays_clean(adapter, monkeypatch):
    """No truncation → no debt: status stays success so the ledger only cries wolf when it should."""
    busy = HistoryChannel(200, 10)
    quiet = HistoryChannel(100, 2)
    _wire(adapter, [busy, quiet])
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 50)
    monkeypatch.setattr(adapter, "_missed_message_backfill_channels", lambda: {"100", "200"})
    monkeypatch.setattr(adapter, "_should_backfill_discord_message", AsyncMock(return_value=False))

    await adapter._run_missed_message_backfill()

    def _row(conn):
        return conn.execute(
            "SELECT status, scanned, error FROM discord_recovery_scans ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

    status, scanned, error = adapter._with_discord_recovery_db(_row)
    assert (status, scanned, error) == ("success", 12, None)


@pytest.mark.asyncio
async def test_single_source_still_keeps_the_latest_n_oldest_first(adapter, monkeypatch):
    """Regression guard for the pre-existing single-channel contract."""
    chan = HistoryChannel(300, 4)
    _wire(adapter, [chan])
    monkeypatch.setattr(adapter, "_missed_message_backfill_limit", lambda: 3)

    got = [m.id async for m in adapter._iter_missed_message_backfill_candidates({"300"})]

    assert got == [chan._messages[1].id, chan._messages[2].id, chan._messages[3].id]
