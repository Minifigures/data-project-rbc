"""API integration tests via FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate the audit log and review queue to a temp directory.
    monkeypatch.setenv("CLAIMGUARD_AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("CLAIMGUARD_QUEUE_DB", str(tmp_path / "queue.sqlite"))
    from claimguard.api.main import app

    with TestClient(app) as c:
        yield c


HIGH_RISK_CLAIM = {
    "claim_id": "CLM-TEST-1", "claimant_id": "MBR-1", "provider_id": "PRV-7",
    "provider_specialty": "Dental", "claim_type": "dental", "procedure_code": "DN-303",
    "diagnosis_code": "Z00.0", "units": 1, "billed_amount": 2400.0, "allowed_amount": 900.0,
    "date_of_service": "2025-03-10", "date_submitted": "2025-03-18",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["audit_chain_valid"] is True


def test_score_high_risk_claim_enters_review_queue(client):
    r = client.post("/score", json=HIGH_RISK_CLAIM)
    assert r.status_code == 200
    body = r.json()
    assert body["band"] == "high"
    assert body["rule_score"] >= 70
    assert len(body["triggered_rules"]) >= 2

    queue = client.get("/review-queue").json()
    assert any(item["claim_id"] == "CLM-TEST-1" for item in queue["items"])


def test_review_decision_is_audited(client):
    client.post("/score", json=HIGH_RISK_CLAIM)
    r = client.post("/review", json={"claim_id": "CLM-TEST-1", "decision": "flagged", "reviewer": "marco"})
    assert r.status_code == 200
    assert r.json()["status"] == "flagged"

    audit = client.get("/audit").json()
    assert audit["chain_valid"] is True
    events = [rec["event_type"] for rec in audit["records"]]
    assert "scored" in events and "human_review" in events


def test_unknown_review_claim_404(client):
    r = client.post("/review", json={"claim_id": "CLM-NOPE", "decision": "approved", "reviewer": "x"})
    assert r.status_code == 404
