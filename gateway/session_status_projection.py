"""Exact-session messaging status from the existing authority projection."""
import sqlite3

from gateway.session_authorities import active_authority
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError


def status_projection(runner, session_id, session_key):
    authority = active_authority(runner)
    if authority is None:
        return None
    try:
        ref = SessionRef(authority.profile_id, authority.logical_owner(session_id))
        live = authority.sessions.get(ref.session_id)
        if live is None or live.route != session_key:
            return None
        with live.event_stream.lock:
            handle = authority._handle(ref)
            pending = authority._pending_rows(ref)
            running = handle.execution_state == "running" and not live.superseded
            prompts = live.controls.snapshot(ref.session_id, handle.execution_generation if running else None)
            if live.superseded:
                return None
            queued = sum(row["status"] == "queued" for row in pending)
            if handle.execution_state == "unknown":
                phase = "unknown"
            elif running:
                kinds = {prompt["kind"] for prompt in prompts}
                phase = ("waiting_for_approval" if "approval" in kinds else
                         "waiting_for_clarification" if "clarify" in kinds else "running")
            else:
                phase = "running" if runner._is_session_running(session_key) else "queued" if queued else "idle"
            return phase, queued
    except (sqlite3.Error, RuntimeStoreError):
        # A retired/unavailable ledger cannot replace the legacy queue's known count.
        return None
