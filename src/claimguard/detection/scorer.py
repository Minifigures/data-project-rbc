"""Claim scorer: fuses the deterministic rules with the ML signals.

Design rule that matters for the interview: the AUTHORITATIVE, customer-facing
number is the deterministic rule score. The ML signals (anomaly score, supervised
fraud probability) are reported ALONGSIDE it as supplementary context, never
folded silently into the headline number. That keeps the score auditable: a
reviewer can always reconstruct it from the rule hits, with no model in the loop.

The optional LLM layer is kept even further out: it only adds a plain-language
narrative, and is never allowed to move the score.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from claimguard.detection.anomaly import AnomalyModel
from claimguard.detection.rules import RuleEngine
from claimguard.detection.supervised import SupervisedModel
from claimguard.pipeline.features import FEATURE_COLUMNS, feature_matrix


class TriggeredRule(BaseModel):
    rule_id: str
    typology: str
    points: int
    reason: str


class ScoredClaim(BaseModel):
    claim_id: str
    rule_score: int
    band: str
    recommendation: str
    explanation: str
    triggered_rules: list[TriggeredRule]
    anomaly_score: float | None = None
    fraud_probability: float | None = None
    policy_version: int | None = None
    narrative: str | None = None  # optional LLM rationale, never affects the score


_BAND_TO_RECOMMENDATION = {
    "low": "auto_pass",
    "review": "route_to_human_review",
    "high": "priority_investigation",
}


class ClaimScorer:
    def __init__(
        self,
        rule_engine: RuleEngine | None = None,
        anomaly_model: AnomalyModel | None = None,
        supervised_model: SupervisedModel | None = None,
    ) -> None:
        self.rules = rule_engine or RuleEngine()
        self.anomaly_model = anomaly_model
        self.supervised_model = supervised_model

    def _ml_signals(self, row: dict) -> tuple[float | None, float | None]:
        """Compute optional ML signals from a feature row, if models are loaded."""
        if self.anomaly_model is None and self.supervised_model is None:
            return None, None
        x = pd.DataFrame([{c: row.get(c, 0.0) for c in FEATURE_COLUMNS}])
        x = feature_matrix(x)
        anomaly = None
        proba = None
        if self.anomaly_model is not None and self.anomaly_model.fitted:
            anomaly = round(float(self.anomaly_model.anomaly_score(x)[0]), 4)
        if self.supervised_model is not None:
            proba = round(float(self.supervised_model.fraud_probability(x)[0]), 4)
        return anomaly, proba

    def score_row(self, row: dict, claim_id: str) -> ScoredClaim:
        """Score one feature row (a dict carrying the engineered feature columns)."""
        rule_result = self.rules.score(row)
        anomaly, proba = self._ml_signals(row)
        return ScoredClaim(
            claim_id=claim_id,
            rule_score=rule_result.score,
            band=rule_result.band,
            recommendation=_BAND_TO_RECOMMENDATION[rule_result.band],
            explanation=rule_result.explanation(),
            triggered_rules=[
                TriggeredRule(rule_id=h.rule_id, typology=h.typology, points=h.points, reason=h.reason)
                for h in rule_result.hits
            ],
            anomaly_score=anomaly,
            fraud_probability=proba,
            policy_version=int(self.rules.policy.get("version", 1)),
        )


def score_dataframe(df_with_features: pd.DataFrame, scorer: ClaimScorer) -> pd.DataFrame:
    """Batch-score a feature DataFrame. Returns the input with score columns added."""
    out = df_with_features.copy()
    records = out.to_dict("records")
    scored = [scorer.score_row(r, str(r.get("claim_id", i))) for i, r in enumerate(records)]
    out["rule_score"] = [s.rule_score for s in scored]
    out["band"] = [s.band for s in scored]
    out["recommendation"] = [s.recommendation for s in scored]
    out["explanation"] = [s.explanation for s in scored]
    if scorer.anomaly_model is not None:
        out["anomaly_score"] = [s.anomaly_score for s in scored]
    if scorer.supervised_model is not None:
        out["fraud_probability"] = [s.fraud_probability for s in scored]
    return out
