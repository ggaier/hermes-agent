"""Restart-time approval policy (#33/A6).

Kept out of ``gateway/run_shutdown.py`` so that facade stays within its code-health line cap.
The withdrawal itself lives in ``tools.approval`` next to the queue it fences; this module only
carries the restart-facing wrapper and its logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gateway.run import GatewayRunner

logger = logging.getLogger(__name__)


def withdraw_pending_approvals_for_restart() -> int:
    """End approval waits nobody can answer (fail-closed) so they cannot hold the restart.

    Sets the process-wide admission fence as well as waking the entries already queued, so a
    prompt admitted after this call cannot become an orphan waiter during the drain. Nothing is
    approved, replayed or killed: each waiting tool denies itself with the restart cause.
    """
    try:
        from tools.approval import begin_approval_withdrawal

        count = begin_approval_withdrawal("the gateway restarted while this approval was pending")
    except Exception:  # a fence failure must not abort the restart path
        logger.debug("Restart: withdrawing pending approvals failed", exc_info=True)
        return 0
    if count:
        logger.warning(
            "Restart: withdrew %d pending approval(s) fail-closed — their turns deny and the "
            "drain proceeds instead of waiting for an answer that cannot arrive", count)
    return count
