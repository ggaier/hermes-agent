"""Canonical authorized native Discord-thread session start.

Reached from the gateway's trusted control socket (``native-start-grant`` then
``native-start-thread``).  The local peer identity is the authority: the socket
itself is the ACL boundary, and the grant records an immutable request digest so
a start cannot be re-pointed at another profile/card/parent/operator/brief.

Everything durable lives under ``$HERMES_HOME/native_start/`` with the gateway
lock file serializing writers, so a restart and concurrent retries converge on
one thread/session.  Thread creation, the brief, and participation go through the
real Discord adapter; the turn is admitted through ``SessionAuthority.submit``
and executed by the gateway's normal FIFO drain.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_contract import Principal, Submission

_ROOT = "native_start"

# A ``starting`` reservation is reclaimable only when its owning process is gone. The
# 60 s control-socket deadline bounds the caller, never the coroutine, and an adapter
# HTTP call can outlive any lease, so a timestamp alone must not displace a live owner:
# two live attempts would both send the brief and admit a turn.


def _dir(home: Path) -> Path:
    return Path(home) / _ROOT


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_digest(profile: str, card_id: str, parent_channel_id: str,
                   operator: dict, brief: str) -> str:
    return hashlib.sha256(_canonical({
        "profile": str(profile), "card_id": str(card_id),
        "parent_channel_id": str(parent_channel_id),
        "operator": {k: str(operator.get(k, "")) for k in ("platform", "user_id", "guild_id")},
        "brief": str(brief),
    }).encode("utf-8")).hexdigest()


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, payload: dict) -> None:
    from gateway.platforms.helpers import atomic_json_write
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, payload, indent=None)


@contextlib.contextmanager
def _locked(home: Path):
    path = _dir(home) / "write.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _trusted_peer(peer_subject: Optional[str]) -> bool:
    """The control socket already enforces the OS ACL; this pins the caller to the
    account that owns the gateway home rather than trusting a params string.

    A POSIX peer is labelled ``uid:<n>`` from the kernel's getpeereid/SO_PEERCRED, so
    that verified label names the owning account and must be accepted. The bare account
    name stays valid only for in-process callers that already hold the process identity.
    """
    if not peer_subject:
        return False
    if peer_subject == getpass.getuser():
        return True
    getuid = getattr(os, "getuid", None)
    return getuid is not None and peer_subject == f"uid:{getuid()}"


def _fail(code: str, message: str) -> dict:
    return {"ok": False, "code": code, "error": message}


def record_grant(home: Path, params: dict, peer_subject: Optional[str]) -> dict:
    if not _trusted_peer(peer_subject):
        return _fail("untrusted_peer", "native-start-grant requires the owning local account")
    try:
        profile = str(params["profile"])
        card_id = str(params["card_id"])
        parent = str(params["parent_channel_id"])
        operator = dict(params["operator"])
        brief = str(params["brief"])
        user_id = str(operator["user_id"])
        guild_id = str(operator["guild_id"])
    except (KeyError, TypeError, ValueError):
        return _fail("invalid_params", "profile, card_id, parent_channel_id, operator, brief required")
    if not all((profile, card_id, parent, user_id, guild_id, brief.strip())):
        return _fail("invalid_params", "empty native-start grant field")
    digest = request_digest(profile, card_id, parent, operator, brief)
    record = {"digest": digest, "profile": profile, "card_id": card_id,
              "parent_channel_id": parent, "operator": operator, "brief": brief,
              "granted_by": getpass.getuser()}
    with _locked(home):
        _write(_dir(home) / "grants" / f"{digest}.json", record)
    return {"ok": True, "digest": digest}


async def _resolve_adapter(runner):
    adapters = getattr(runner, "adapters", None) or {}
    adapter = adapters.get(Platform.DISCORD)
    if adapter is None:
        factory = getattr(runner, "_adapter_for_platform", None)
        if callable(factory):
            adapter = factory(Platform.DISCORD)
    return adapter


def _read_run(path: Path) -> Optional[dict]:
    """A run record, ``None`` when absent — a *corrupt* record raises.

    Missing means "never started"; corrupt means "we cannot know": the two must never
    collapse into the same answer, or a damaged record silently becomes a fresh start.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"unreadable native-start run record: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("corrupt native-start run record")
    return data


def _pid_alive(pid: Any) -> bool:
    """Whether the process that wrote a ``starting`` record still exists.

    ``os.kill(pid, 0)`` is the only cross-process liveness probe available; a
    PermissionError means the pid exists but belongs to another account (still alive).
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _starting_owner_alive(prior: dict) -> bool:
    """A ``starting`` attempt is live exactly while its pid exists.

    ``started_at`` is kept on the record for diagnostics only: it bounds nothing, because
    the start coroutine is not bounded by the caller's socket deadline and may legitimately
    run longer than any lease while it is still doing adapter I/O. Only a genuinely dead
    owner (SIGKILL, crash, restart) makes the reservation reclaimable, which is what the
    recovery path needs. A record with no pid predates this protocol and is treated as
    stale rather than trusted forever, which is what used to wedge the card permanently.
    """
    return _pid_alive(prior.get("pid"))


def _claim(home: Path, paths: dict, digest: str, attempt: str, idem_key: str) -> tuple[str, Any]:
    """Reserve the card's single start under the writer lock.

    Returns ``(state, payload)``: ``done`` (completed result), ``mine`` (we own the
    reservation and must do the I/O), ``busy`` (another attempt owns it), ``conflict``,
    ``unauthorized``, ``identity_mismatch`` or ``state_corrupt``.
    """
    with _locked(home):
        try:
            prior = _read_run(paths["run"])
        except ValueError:
            return "state_corrupt", None
        if prior is not None:
            if prior.get("digest") != digest:
                return "conflict", None
            if prior.get("state") == "complete" and isinstance(prior.get("result"), dict):
                return "done", prior["result"]
            if prior.get("attempt") == attempt:
                return "mine", prior
            if prior.get("state") == "pending" and prior.get("thread_id"):
                return "mine", prior  # adopt the thread the failed attempt already created
            if prior.get("state") == "starting":
                if _starting_owner_alive(prior):
                    return "busy", prior
                # The owning process died mid-start: take the reservation over. Thread id,
                # digest and idempotency binding carry across, so the recovering attempt
                # adopts a thread the dead one already created instead of launching a
                # second one. A live owner is never displaced, however long it runs.
                reclaimed = {"state": "starting", "digest": digest, "attempt": attempt,
                             "idempotency_key": idem_key, "pid": os.getpid(),
                             "started_at": time.time(),
                             "reclaimed_from": prior.get("attempt"),
                             "reclaimed_pid": prior.get("pid")}
                if prior.get("thread_id"):
                    reclaimed["thread_id"] = str(prior["thread_id"])
                _write(paths["run"], reclaimed)
                return "mine", reclaimed
            return "busy", prior
        if _read(paths["grant"]) is None:
            if _read(paths["card_grant"]) is not None:
                return "identity_mismatch", None
            return "unauthorized", None
        _write(paths["run"], {"state": "starting", "digest": digest, "attempt": attempt,
                              "idempotency_key": idem_key, "pid": os.getpid(),
                              "started_at": time.time()})
        return "mine", None


def _owned(paths: dict, attempt: str) -> Optional[dict]:
    """The on-disk record if this attempt may still write it, else ``None``.

    A record that names a different attempt belongs to a newer owner (a reclaim can only
    happen after this attempt's pid died, but defence in depth keeps a displaced attempt
    from overwriting the record and submitting a second turn). A record with no attempt
    (``pending``, written by ``_hold``) is unowned and adoptable; a missing record means
    this attempt has nothing left to write to.
    """
    current = _read(paths["run"])
    if current is None:
        return None
    owner = current.get("attempt")
    if owner is not None and owner != attempt:
        return None
    return current


def _reserve_thread(paths: dict, digest: str, attempt: str, idem_key: str, thread_id: str,
                    started_at: float) -> bool:
    """Persist the created thread on the *starting* record before any further I/O.

    A SIGKILL between create and submit then leaves enough on disk for the reclaiming
    attempt to adopt this thread instead of creating a second one. Returns ``False`` when
    another attempt has taken the record over, so the caller must stop before send/submit.
    """
    with _locked(paths["run"].parent.parent):
        if _owned(paths, attempt) is None:
            return False
        _write(paths["run"], {"state": "starting", "digest": digest, "attempt": attempt,
                              "idempotency_key": idem_key, "pid": os.getpid(),
                              "started_at": started_at, "thread_id": str(thread_id)})
        return True


def _release(paths: dict, attempt: str) -> None:
    """Drop our own reservation when the attempt created no thread.

    Guarded by the attempt id: a reclaiming attempt must never delete a record another
    attempt currently owns.
    """
    with _locked(paths["run"].parent.parent):
        current = _read(paths["run"])
        if current is not None and current.get("attempt") == attempt:
            with contextlib.suppress(OSError):
                paths["run"].unlink()


def _hold(paths: dict, digest: str, thread_id: str, idem_key: str, attempt: str) -> bool:
    """Keep an unfinished attempt's thread as a resumable reservation.

    The shipped Discord adapter exposes no delete/archive primitive, so dropping the
    record would orphan a real thread and the retry would create a second one. The
    pending record makes the retry adopt the same thread instead. Returns ``False`` when
    another attempt owns the record, so a displaced attempt never overwrites it.
    """
    with _locked(paths["run"].parent.parent):
        if _owned(paths, attempt) is None:
            return False
        _write(paths["run"], {"state": "pending", "digest": digest, "thread_id": str(thread_id),
                              "idempotency_key": idem_key})
        return True


def _complete(paths: dict, digest: str, idem_key: str, payload: dict, attempt: str) -> bool:
    with _locked(paths["run"].parent.parent):
        if _owned(paths, attempt) is None:
            return False
        _write(paths["run"], {"state": "complete", "digest": digest,
                              "idempotency_key": idem_key, "result": payload})
        return True


async def _rollback_thread(adapter, thread_id: str) -> None:
    """Give back a thread created for an attempt that could not be admitted."""
    for name in ("delete_thread", "delete_handoff_thread", "archive_thread"):
        undo = getattr(adapter, name, None)
        if callable(undo):
            with contextlib.suppress(Exception):
                await undo(str(thread_id))
            return


async def start_native_thread(runner, params: dict, peer_subject: Optional[str]) -> dict:
    if not _trusted_peer(peer_subject):
        return _fail("untrusted_peer", "native-start-thread requires the owning local account")
    authority = getattr(runner, "session_authority", None)
    if authority is None:
        return _fail("unavailable", "the gateway has no session authority")
    try:
        profile = str(params["profile"])
        card_id = str(params["card_id"])
        idem_key = str(params["idempotency_key"])
        parent = str(params["parent_channel_id"])
        operator = dict(params["operator"])
        brief = str(params["brief"])
        user_id = str(operator["user_id"])
        guild_id = str(operator["guild_id"])
        thread_name = str(params.get("thread_name") or f"{card_id} native start")[:80]
    except (KeyError, TypeError, ValueError):
        return _fail("invalid_params", "profile, card_id, idempotency_key, parent_channel_id, operator, brief required")
    if not all((profile, card_id, idem_key, parent, user_id, guild_id, brief.strip())):
        return _fail("invalid_params", "empty native-start field")
    if profile != getattr(authority, "profile_id", profile):
        return _fail("cross_profile", "request profile is not the serving profile")
    digest = request_digest(profile, card_id, parent, operator, brief)
    home = Path(getattr(runner, "hermes_home", None) or os.environ.get("HERMES_HOME", ""))
    if not str(home) or not home.is_dir():
        return _fail("unavailable", "gateway home is not available")
    # One authorized card is one start: the key is the card, never the caller's
    # idempotency string, so a re-keyed retry cannot launch a second thread.
    run_key = _key(f"{profile}\n{card_id}")
    attempt = f"{os.getpid()}:{uuid.uuid4().hex}"
    paths = {"run": _dir(home) / "runs" / f"{run_key}.json",
             "grant": _dir(home) / "grants" / f"{digest}.json",
             "card_grant": _dir(home) / "grants" / f"{_key(profile + card_id)}.json"}

    state, payload = _claim(home, paths, digest, attempt, idem_key)
    deadline = asyncio.get_running_loop().time() + 60.0
    while state == "busy":
        # Another attempt holds the card; wait for its committed result instead of
        # creating a second thread. Nothing is held here, so the loop keeps running.
        if asyncio.get_running_loop().time() > deadline:
            return _fail("in_progress", "another start for this card is still running")
        await asyncio.sleep(0.02)
        state, payload = _claim(home, paths, digest, attempt, idem_key)
    # Ownership is ours from here; this timestamp is the lease start written on the
    # record so a crash after the next write is reclaimable.
    claimed_at = time.time()
    if state == "done":
        return {"ok": True, **payload}
    if state == "conflict":
        return _fail("idempotency_conflict",
                     "the idempotency key was already used for a different request")
    if state == "identity_mismatch":
        return _fail("identity_mismatch", "the grant does not match this request")
    if state == "unauthorized":
        return _fail("unauthorized", "no scoped authorization for this request")
    if state == "state_corrupt":
        return _fail("state_corrupt", "the native-start record is unreadable; refusing to start")

    prior_thread = (payload or {}).get("thread_id")
    adapter = await _resolve_adapter(runner)
    if adapter is None:
        # An adopted thread is the only durable copy of a real Discord thread: keep it as
        # a resumable reservation, or the retry (the grant is still on disk) creates a
        # second thread for the same card and orphans this one.
        if prior_thread:
            _hold(paths, digest, str(prior_thread), idem_key, attempt)
        else:
            _release(paths, attempt)
        return _fail("unavailable", "no Discord adapter is connected")
    thread_id = str(prior_thread or "")
    created_here = False
    try:
        if not thread_id:
            thread_id = await adapter.create_handoff_thread(parent, thread_name)
            if not thread_id:
                _release(paths, attempt)
                return _fail("thread_failed", "the Discord thread could not be created")
            thread_id = str(thread_id)
            created_here = True
        # Persist the thread before membership/send/submit: a crash from here on is
        # recoverable by adopting this thread rather than creating a second one.
        if not _reserve_thread(paths, digest, attempt, idem_key, thread_id, claimed_at):
            if created_here:
                await _rollback_thread(adapter, thread_id)
            return _fail("superseded", "another start took this card over")

        member_add = getattr(adapter, "add_thread_member", None)
        if callable(member_add):
            added = await member_add(thread_id, user_id)
            if not added:
                await _rollback_thread(adapter, thread_id)
                if not _hold(paths, digest, thread_id, idem_key, attempt):
                    return _fail("superseded", "another start took this card over")
                return _fail("visibility_failed", "the intended user could not access the thread")

        result = await adapter.send(
            thread_id, brief, metadata={"thread_id": thread_id, "native_start": True},
        )
        if not getattr(result, "success", False):
            await _rollback_thread(adapter, thread_id)
            if not _hold(paths, digest, thread_id, idem_key, attempt):
                return _fail("superseded", "another start took this card over")
            return _fail("transport_failed", f"brief send failed: {getattr(result, 'error', 'unknown')}")

        tracker = getattr(adapter, "_threads", None)
        if tracker is not None:
            await tracker.mark_async(thread_id)

        source = SessionSource(
            platform=Platform.DISCORD, chat_id=thread_id, chat_type="thread",
            # Never fabricate a display name: an invented one ("operator") is not this user's
            # identity, and a stored origin that carries it cannot be compared against the real
            # envelope later (producer wakes refuse on source drift). The grant identifies the
            # operator by id; the display name is filled in by the operator's own first message.
            user_id=user_id, user_name=(str(operator["user_name"]) if operator.get("user_name") else None),
            thread_id=thread_id, parent_chat_id=parent,
        )
        ref = authority.register(source)
        receipt = await authority.submit(
            Principal("native-start", profile, frozenset({"session:submit"}), "control-socket"),
            Submission(idem_key, ref, {"text": brief}, "queue"),
        )
    except Exception:
        if thread_id:
            await _rollback_thread(adapter, thread_id)
            _hold(paths, digest, thread_id, idem_key, attempt)
        else:
            _release(paths, attempt)
        raise

    # A receipt that is not queued/started means the turn was not admitted; report it
    # and hand the thread back rather than recording a run that never runs.
    status = getattr(receipt, "status", None)
    if status not in {"queued", "started"}:
        await _rollback_thread(adapter, thread_id)
        if not _hold(paths, digest, thread_id, idem_key, attempt):
            return _fail("superseded", "another start took this card over")
        return _fail("admission_refused", f"the turn was not admitted ({status or 'unknown'})")

    payload = {"thread_id": thread_id, "session_id": ref.session_id,
               "admission_id": receipt.admission_id, "digest": digest}
    if not _complete(paths, digest, idem_key, payload, attempt):
        return _fail("superseded", "another start took this card over")
    return {"ok": True, **payload}



def control_handlers(runner, home: Path, loop=None) -> dict:
    """Private control-socket handlers ``(params, peer_subject)``.

    Registered in ``GatewayControlServer.private_handlers``: the socket refuses
    the call unless the peer identity is present and the request is protocol 1.
    ``loop`` is the gateway's one authoritative main loop; a handler dispatched from
    the socket executor must marshal onto it, never build a throwaway event loop
    (the FIFO drain task would die with it).
    """
    import asyncio

    def grant(params: dict, peer_subject: Optional[str]) -> dict:
        return record_grant(home, params, peer_subject)

    def start(params: dict, peer_subject: Optional[str]) -> dict:
        authoritative = loop or getattr(runner, "_main_loop", None)
        coro = start_native_thread(runner, params, peer_subject)
        if authoritative is not None and authoritative.is_running():
            return asyncio.run_coroutine_threadsafe(coro, authoritative).result(timeout=60)
        coro.close()
        return _fail("unavailable", "the gateway main loop is not running")

    return {"native-start-grant": grant, "native-start-thread": start}
