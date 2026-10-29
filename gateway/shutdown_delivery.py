"""Adapter send-result probes shared by the shutdown/notice paths (#33).

Split out of ``gateway/run_shutdown.py`` so that facade stays inside its code-health line cap.
Both helpers are deliberately trivial: a missing ``success`` attribute counts as success because
most adapters return a plain dict/None rather than a ``SendResult``.
"""

from __future__ import annotations

from typing import Any


def send_failed(result: Any) -> bool:
    """True when an adapter ``send()`` result explicitly reports failure."""
    return result is not None and getattr(result, "success", True) is False


def send_error(result: Any) -> str:
    """Error text of a failed ``send()`` result (adapters may omit it)."""
    return getattr(result, "error", "send returned success=False")
