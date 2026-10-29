"""Small output helpers for the gateway CLI (#33).

Split out of ``hermes_cli/gateway.py`` so that facade stays inside its code-health line cap;
``gateway`` re-imports the name, so ``_gw()._print_indented`` (the setup wizard/service modules)
keeps resolving.
"""

from __future__ import annotations


def print_indented(text: str, emit=print) -> None:
    for line in text.splitlines():
        emit(f"  {line}")
