"""The v1 status payload exposes a read-only pending-approval count for the host watchdog."""
import pytest

from gateway.control_socket import build_status_payload
from tools import approval as approval_mod
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = 'card39:approval'


@pytest.fixture(autouse=True)
def _clean_queue():
    with approval_mod._lock:
        approval_mod._gateway_queues.pop(SESSION, None)
    yield
    with approval_mod._lock:
        approval_mod._gateway_queues.pop(SESSION, None)


def test_status_payload_counts_pending_gateway_approvals():
    assert build_status_payload()['pending_approvals'] == 0
    with approval_mod._lock:
        approval_mod._gateway_queues.setdefault(SESSION, []).append(
            _ApprovalEntry({'command': 'rm -rf /tmp/card39', 'pattern_key': 'rm'}))
    assert build_status_payload()['pending_approvals'] == 1
    assert approval_mod.resolve_gateway_approval(SESSION, 'deny') == 1
    assert build_status_payload()['pending_approvals'] == 0


def test_status_payload_reports_unknown_when_collection_fails(monkeypatch):
    def boom():
        raise RuntimeError('queue unavailable')

    monkeypatch.setattr('tools.approval.pending_gateway_approval_count', boom)
    assert build_status_payload()['pending_approvals'] is None
