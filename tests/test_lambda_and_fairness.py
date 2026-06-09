"""AWS Lambda handler (offline) and the fairness check."""

from __future__ import annotations

from claimguard.aws.lambda_handler import score_claim_payload
from claimguard.detection.fairness import disparity_report


def test_lambda_scores_fee_outlier_high():
    scored = score_claim_payload(
        {
            "claim_id": "CLM-L1", "provider_id": "PRV-1", "provider_specialty": "Dental",
            "procedure_code": "DN-301", "diagnosis_code": "K02.9", "units": 1,
            "billed_amount": 360.0, "allowed_amount": 120.0,
            "date_of_service": "2025-04-01", "date_submitted": "2025-04-05",
        }
    )
    assert scored["band"] == "high"
    assert "fee_outlier" in scored["triggered_rules"]
    assert scored["recommendation"] == "priority_investigation"


def test_fairness_report_runs(scored_df):
    rep = disparity_report(scored_df, group_column="provider_specialty")
    assert 0.0 <= rep.disparity_ratio <= 1.0
    assert "Disparity by provider_specialty" in rep.summary()
