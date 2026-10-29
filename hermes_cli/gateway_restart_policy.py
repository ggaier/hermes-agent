"""Target validation for ``hermes gateway restart --withdraw-approvals`` (#33/A6).

Split out of ``hermes_cli/gateway.py`` so that module stays within its code-health line
cap and ``_cmd_restart`` keeps its cyclomatic-complexity margin. Import lazily: this
module is only reached from the CLI path, which imports the launcher on demand.
"""

from __future__ import annotations

_UNSUPPORTED = "Error: --withdraw-approvals is not supported for this target in this release."


def restart_withdraw_approvals(args) -> bool:
    """Validate the target for ``--withdraw-approvals`` and return the requested policy.

    The policy withdraws approval waits fail-closed on the process being restarted; it never
    approves, replays or force-kills anything. Only a macOS/launchd restart of this profile's
    own gateway can carry it in this release, so every other target refuses with a non-zero
    exit BEFORE any stop/signal is issued rather than silently degrading to a bare SIGUSR1
    (or a hard kill) and reporting a safety guarantee it did not obtain.
    """
    import sys

    if not getattr(args, "withdraw_approvals", False):
        return False
    from hermes_cli import gateway as gw

    if getattr(args, "all", False) or getattr(args, "system", False):
        print(_UNSUPPORTED)
        sys.exit(1)
    name = gw._current_profile_name()
    if name and name != "default" and not getattr(args, "force", False):
        from hermes_cli.gateway_profile_lifecycle import profile_is_standalone

        if not profile_is_standalone(gw.get_hermes_home()) and (
            gw._served_by_another_host_gateway() is not None
            or gw.named_profile_served_by_running_multiplexer()
        ):
            print(_UNSUPPORTED)
            sys.exit(1)
    try:
        from hermes_cli.service_manager import detect_service_manager

        in_s6 = detect_service_manager() == "s6"
    except (ImportError, OSError):
        # s6 detection is best-effort: the launchd check below still refuses every non-macOS host.
        in_s6 = False
    if in_s6 or gw._installed_service_kind_for(gw.is_windows) != "launchd":
        print(_UNSUPPORTED)
        sys.exit(1)
    return True
