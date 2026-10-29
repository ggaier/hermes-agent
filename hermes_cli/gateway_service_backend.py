"""Host service-manager predicate for the gateway CLI (#33).

Split out of ``hermes_cli/gateway.py`` so that facade stays inside its code-health line cap;
``gateway`` re-imports the name, so ``gw._service_backend`` stays patchable at the same path.
"""

from __future__ import annotations


def service_backend(*, windows: bool = True) -> str | None:
    """Host service manager: ``"systemd"`` / ``"launchd"`` / ``"windows"`` / None, in the canonical
    predicate order every subcommand routes on. ``windows=False`` never probes ``is_windows()``."""
    from hermes_cli import gateway as gw  # lazy: the facade imports this module

    if gw.supports_systemd_services():
        return "systemd"
    if gw.is_macos():
        return "launchd"
    if windows and gw.is_windows():
        return "windows"
    return None
