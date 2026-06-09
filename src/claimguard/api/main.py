"""ClaimGuard API.

Submit a claim, get back a scored and explained result; claims above the
auto-pass band are placed on the human-review queue, and every scoring event and
human decision is written to the immutable audit log.

Run it:
    uvicorn claimguard.api.main:app --reload
Then open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from claimguard.api.audit import AuditLog
from claimguard.api.review_queue import ReviewQueue
from claimguard.detection.anomaly import AnomalyModel
from claimguard.detection.rules import RuleEngine
from claimguard.detection.scorer import ClaimScorer, ScoredClaim
from claimguard.detection.supervised import SupervisedModel
from claimguard.pipeline.features import add_features

logger = logging.getLogger("claimguard.api")

MODEL_DIR = Path("models")
_state: dict = {}


class ClaimIn(BaseModel):
    """Inbound claim (no fraud label, that is what we are predicting)."""

    claim_id: str | None = Field(None, description="Optional; generated if omitted")
    claimant_id: str = "MBR-UNKNOWN"
    provider_id: str = "PRV-UNKNOWN"
    provider_specialty: str = "General Practice"
    claim_type: str = "medical"
    procedure_code: str
    diagnosis_code: str | None = None
    units: int = 1
    billed_amount: float
    allowed_amount: float
    paid_amount: float | None = None
    date_of_service: date
    date_submitted: date
    place_of_service: str = "clinic"
    region: str = "ON"

    model_config = {
        "json_schema_extra": {
            "example": {
                "claim_id": "CLM-DEMO-001",
                "claimant_id": "MBR-00042",
                "provider_id": "PRV-007",
                "provider_specialty": "Dental",
                "claim_type": "dental",
                "procedure_code": "DN-303",
                "diagnosis_code": "Z00.0",
                "units": 1,
                "billed_amount": 2400.0,
                "allowed_amount": 900.0,
                "date_of_service": "2025-03-10",
                "date_submitted": "2025-03-18",
                "place_of_service": "clinic",
                "region": "ON",
            }
        }
    }


class ReviewDecision(BaseModel):
    claim_id: str
    decision: str = Field(..., description="approved | flagged | dismissed")
    reviewer: str = "analyst"
    note: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    rules = RuleEngine()
    anomaly = AnomalyModel.load(MODEL_DIR / "anomaly.joblib") if (MODEL_DIR / "anomaly.joblib").exists() else None
    supervised = (
        SupervisedModel.load(MODEL_DIR / "supervised_gb.joblib")
        if (MODEL_DIR / "supervised_gb.joblib").exists()
        else None
    )
    _state["scorer"] = ClaimScorer(rules, anomaly_model=anomaly, supervised_model=supervised)
    _state["audit"] = AuditLog()
    _state["queue"] = ReviewQueue()
    _state["models_loaded"] = {"anomaly": anomaly is not None, "supervised": supervised is not None}
    logger.info("ClaimGuard API ready. Models: %s", _state["models_loaded"])
    yield
    _state.clear()


app = FastAPI(title="ClaimGuard", version="0.1.0", lifespan=lifespan)


def _score_claim(claim: ClaimIn) -> ScoredClaim:
    claim_id = claim.claim_id or f"CLM-{uuid.uuid4().hex[:10]}"
    row = claim.model_dump()
    row["claim_id"] = claim_id
    row["is_fraud"] = 0
    row["fraud_type"] = None
    feat = add_features(pd.DataFrame([row]))
    scorer: ClaimScorer = _state["scorer"]
    return scorer.score_row(feat.iloc[0].to_dict(), claim_id)


@app.get("/health")
def health() -> dict:
    scorer: ClaimScorer = _state["scorer"]
    return {
        "status": "ok",
        "models_loaded": _state.get("models_loaded", {}),
        "policy_version": int(scorer.rules.policy.get("version", 1)),
        "audit_chain_valid": _state["audit"].verify_chain(),
    }


@app.post("/score", response_model=ScoredClaim)
def score(claim: ClaimIn) -> ScoredClaim:
    result = _score_claim(claim)
    audit: AuditLog = _state["audit"]
    audit.append(
        event_type="scored",
        claim_id=result.claim_id,
        payload={
            "rule_score": result.rule_score,
            "band": result.band,
            "recommendation": result.recommendation,
            "triggered_rule_ids": [r.rule_id for r in result.triggered_rules],
            "anomaly_score": result.anomaly_score,
            "fraud_probability": result.fraud_probability,
            "policy_version": result.policy_version,
        },
    )
    if result.band != "low":
        _state["queue"].add(
            claim_id=result.claim_id,
            rule_score=result.rule_score,
            band=result.band,
            recommendation=result.recommendation,
            explanation=result.explanation,
            payload=result.model_dump(),
        )
    return result


@app.get("/review-queue")
def review_queue(status: str = "pending") -> dict:
    return {"status": status, "items": _state["queue"].list(status=status)}


@app.post("/review")
def review(decision: ReviewDecision) -> dict:
    queue: ReviewQueue = _state["queue"]
    try:
        updated = queue.decide(decision.claim_id, decision.decision, decision.reviewer, decision.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _state["audit"].append(
        event_type="human_review",
        claim_id=decision.claim_id,
        payload={"decision": decision.decision, "reviewer": decision.reviewer, "note": decision.note},
    )
    return updated


@app.get("/audit")
def audit(limit: int = 100) -> dict:
    log: AuditLog = _state["audit"]
    records = log.all()
    return {"count": len(records), "chain_valid": log.verify_chain(), "records": records[-limit:]}
