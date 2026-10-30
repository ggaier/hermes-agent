"""Recovery receipts must not be stored on the Discord message object (#39 N5).

Production crash: ``message._hermes_recovery_receipt = None`` raised
``AttributeError: 'Message' object has no attribute '_hermes_recovery_receipt' and no
__dict__ for setting new attributes`` — discord.Message defines ``__slots__``, so arbitrary
attributes cannot be set on a real message. The dispatch died BEFORE the consumer ran, and
the ledger recorded ``failed`` (not retryable), so the message was lost.

The test doubles below deliberately mimic ``__slots__`` for that reason: a plain
SimpleNamespace hides this class of defect.
"""

import sys
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


class SlottedMessage:
    """Mimics discord.Message: no __dict__, so attribute assignment raises AttributeError."""

    __slots__ = (
        "attachments",
        "author",
        "channel",
        "content",
        "created_at",
        "guild",
        "id",
        "mentions",
        "reactions",
        "reference",
        "type",
    )

    def __init__(self, message_id, channel=None):
        self.id = message_id
        self.content = "please ingest"
        self.author = SimpleNamespace(id=42, bot=False, display_name="Emo", name="emo")
        self.channel = channel or SimpleNamespace(
            id=123, parent_id=None, name="wiki-inbox", guild=SimpleNamespace(id=777, name="emo"),
        )
        self.guild = SimpleNamespace(id=777, name="emo")
        self.created_at = object()
        self.attachments = []
        self.mentions = []
        self.reference = None
        self.type = discord.MessageType.default
        self.reactions = []


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
    # Reach the admission gate; the un-addressed branch has its own test below.
    monkeypatch.setattr(adapter, "_discord_require_mention", lambda: False)
    return adapter


@pytest.mark.asyncio
async def test_dispatch_on_a_slotted_message_does_not_raise(adapter, monkeypatch):
    """The production shape: discord.Message has __slots__. Dispatch must not write to it."""
    message = SlottedMessage(5001)
    monkeypatch.setattr(adapter, "_discord_message_admission", lambda msg, claim=False: (True, True))
    adapter._handle_message = AsyncMock(return_value=True)

    receipt = await adapter._dispatch_recovered_message(message)

    assert receipt.accepted is True
    assert not hasattr(message, "_hermes_recovery_receipt")


@pytest.mark.asyncio
async def test_slotted_message_refusal_is_carried_out_of_band(adapter, monkeypatch):
    """A refusal stamped by the consumer must still reach the scan, without touching the message."""
    message = SlottedMessage(5002)
    monkeypatch.setattr(adapter, "_discord_message_admission", lambda msg, claim=False: (True, True))

    async def _refuse(msg, **kwargs):
        adapter._record_recovery_receipt(msg.id, {"accepted": False, "refused": True})
        return False

    adapter._handle_message = AsyncMock(side_effect=_refuse)

    receipt = await adapter._dispatch_recovered_message(message)

    assert receipt.accepted is False
    assert receipt.refused is True
    assert not hasattr(message, "_hermes_recovery_receipt")


@pytest.mark.asyncio
async def test_receipt_is_not_reused_for_a_second_dispatch(adapter, monkeypatch):
    """Stale receipts would make a later dispatch inherit an earlier outcome."""
    message = SlottedMessage(5003)
    monkeypatch.setattr(adapter, "_discord_message_admission", lambda msg, claim=False: (True, True))
    adapter._handle_message = AsyncMock(side_effect=lambda msg, **k: (
        adapter._record_recovery_receipt(msg.id, {"accepted": False, "refused": True}) or False
    ))
    first = await adapter._dispatch_recovered_message(message)
    assert first.refused is True

    adapter._handle_message = AsyncMock(return_value=True)
    second = await adapter._dispatch_recovered_message(message)
    assert second.accepted is True
    assert second.refused is False


@pytest.mark.asyncio
async def test_unknown_outcome_stays_fail_closed(adapter, monkeypatch):
    """No receipt and no refusal: the scan must not learn 'refused' (it would retry a consumed msg)."""
    message = SlottedMessage(5004)
    monkeypatch.setattr(adapter, "_discord_message_admission", lambda msg, claim=False: (True, True))
    adapter._handle_message = AsyncMock(return_value=False)

    receipt = await adapter._dispatch_recovered_message(message)

    assert receipt.accepted is False
    assert receipt.refused is False


@pytest.mark.asyncio
async def test_unaddressed_message_returns_a_refusal_receipt(adapter, monkeypatch):
    """The mention gate used to `return False` from a function whose callers read
    `receipt.accepted` — every un-addressed candidate crashed the scan."""
    monkeypatch.setattr(adapter, "_discord_require_mention", lambda: True)
    monkeypatch.setattr(adapter, "_discord_free_response_channels", lambda: set())
    monkeypatch.setattr(adapter, "_in_bot_thread", lambda msg: False)
    monkeypatch.setattr(adapter, "_self_is_explicitly_mentioned", lambda msg: False)
    message = SlottedMessage(5005)

    receipt = await adapter._dispatch_recovered_message(message)

    assert receipt.accepted is False
    assert receipt.refused is True
