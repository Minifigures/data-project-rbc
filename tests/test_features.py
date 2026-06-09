"""Feature engineering correctness."""

from __future__ import annotations

import pandas as pd

from claimguard.data.schema import CLAIM_COLUMNS
from claimguard.pipeline.features import FEATURE_COLUMNS, add_features


def _row(**overrides):
    base = {
        "claim_id": "CLM-1", "claimant_id": "MBR-1", "provider_id": "PRV-1",
        "provider_specialty": "Dental", "claim_type": "dental", "procedure_code": "DN-303",
        "diagnosis_code": "Z00.0", "units": 1, "billed_amount": 900.0, "allowed_amount": 900.0,
        "paid_amount": None, "date_of_service": "2025-03-01", "date_submitted": "2025-03-05",
        "place_of_service": "clinic", "region": "ON", "is_fraud": 0, "fraud_type": None,
    }
    base.update(overrides)
    return base


def test_all_feature_columns_present(featured_df):
    for col in FEATURE_COLUMNS:
        assert col in featured_df.columns


def test_fee_ratio_accounts_for_units():
    df = pd.DataFrame([_row(billed_amount=400.0, allowed_amount=100.0, units=4)], columns=CLAIM_COLUMNS)
    feat = add_features(df)
    # 400 / (100 * 4) == 1.0, not 4.0
    assert abs(feat.iloc[0]["fee_ratio"] - 1.0) < 1e-6


def test_temporal_impossible_when_service_after_submit():
    df = pd.DataFrame([_row(date_of_service="2025-03-10", date_submitted="2025-03-01")], columns=CLAIM_COLUMNS)
    feat = add_features(df)
    assert bool(feat.iloc[0]["is_temporal_impossible"]) is True


def test_duplicate_flagged_on_second_occurrence():
    r1 = _row(claim_id="CLM-1")
    r2 = _row(claim_id="CLM-2")  # same member/provider/code/service-date
    feat = add_features(pd.DataFrame([r1, r2], columns=CLAIM_COLUMNS))
    assert list(feat["is_duplicate"]) == [False, True]
