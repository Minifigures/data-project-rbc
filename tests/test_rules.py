"""Deterministic rule engine behaviour."""

from __future__ import annotations

from claimguard.detection.rules import RuleEngine


def _benign():
    return {
        "fee_ratio": 1.0, "procedure_complexity": 1, "units": 1, "units_over_max": 0,
        "diagnosis_routine": False, "submit_lag_days": 5, "is_temporal_impossible": False,
        "is_duplicate": False, "provider_claim_count": 10, "provider_avg_fee_ratio": 1.0,
        "billed_zscore_by_code": 0.0, "billed_amount": 100.0,
    }


def test_benign_claim_scores_low():
    eng = RuleEngine()
    res = eng.score(_benign())
    assert res.score == 0
    assert res.band == "low"


def test_temporal_impossible_is_high():
    eng = RuleEngine()
    row = _benign() | {"is_temporal_impossible": True}
    res = eng.score(row)
    assert res.band == "high"
    assert any(h.rule_id == "temporal_impossible" for h in res.hits)


def test_fee_outlier_stacks_and_caps_at_100():
    eng = RuleEngine()
    row = _benign() | {"fee_ratio": 3.0, "billed_amount": 5000.0, "provider_avg_fee_ratio": 2.0}
    res = eng.score(row)
    assert res.score <= 100
    ids = {h.rule_id for h in res.hits}
    assert {"fee_outlier", "extreme_fee_outlier"}.issubset(ids)


def test_upcoding_mismatch_triggers_review():
    eng = RuleEngine()
    row = _benign() | {"procedure_complexity": 3, "diagnosis_routine": True}
    res = eng.score(row)
    assert res.band in {"review", "high"}
    assert any(h.rule_id == "upcoding_mismatch" for h in res.hits)


def test_explanation_is_human_readable():
    eng = RuleEngine()
    res = eng.score(_benign() | {"is_duplicate": True})
    explanation = res.explanation().lower()
    assert "double billing" in explanation
    assert any(h.rule_id == "duplicate_claim" for h in res.hits)
