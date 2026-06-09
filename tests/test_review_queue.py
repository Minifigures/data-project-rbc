"""Human review queue."""

from __future__ import annotations

import pytest

from claimguard.api.review_queue import ReviewQueue


def _add(queue, claim_id="CLM-1"):
    queue.add(claim_id, rule_score=75, band="high", recommendation="priority_investigation",
              explanation="fee outlier", payload={"provider_id": "PRV-1"})


def test_add_and_list_pending(tmp_path):
    q = ReviewQueue(tmp_path / "q.sqlite")
    _add(q)
    pending = q.list("pending")
    assert len(pending) == 1
    assert pending[0]["claim_id"] == "CLM-1"


def test_decide_updates_status(tmp_path):
    q = ReviewQueue(tmp_path / "q.sqlite")
    _add(q)
    updated = q.decide("CLM-1", "flagged", reviewer="marco", note="clear upcode")
    assert updated["status"] == "flagged"
    assert updated["reviewer"] == "marco"
    assert q.list("pending") == []


def test_invalid_decision_raises(tmp_path):
    q = ReviewQueue(tmp_path / "q.sqlite")
    _add(q)
    with pytest.raises(ValueError):
        q.decide("CLM-1", "maybe", reviewer="marco")


def test_unknown_claim_raises(tmp_path):
    q = ReviewQueue(tmp_path / "q.sqlite")
    with pytest.raises(KeyError):
        q.decide("CLM-NOPE", "approved", reviewer="marco")
