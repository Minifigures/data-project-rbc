"""Feature engineering shared by the rule engine and the ML models.

One feature table feeds everything, so the deterministic rules and the
statistical models reason over the same, auditable inputs. Features here are
intentionally simple and explainable: ratios, counts, lags, and within-code
z-scores. Nothing here uses the fraud label, so there is no target leakage.

A note on train vs serve: provider-level aggregates are computed over the batch
in this POC. In production you would read a provider's running history from the
feature store / claims database instead of the current batch, so a single
incoming claim can be scored without re-seeing the whole dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from claimguard.data.synthetic import CODE_BOOK

# Flatten the code book into per-code lookups. Unknown codes (e.g. from an open
# dataset we did not author) fall back to safe defaults that do not trigger rules.
_CODE_COMPLEXITY: dict[str, int] = {}
_CODE_MAX_UNITS: dict[str, int] = {}
for _codes in CODE_BOOK.values():
    for _c in _codes:
        _CODE_COMPLEXITY[_c.code] = _c.complexity
        _CODE_MAX_UNITS[_c.code] = _c.max_units

ROUTINE_DIAGNOSES = {"Z00.0", "M54.5", "J06.9", "K02.9", "Z23", "R51"}

# Default max units when a code is unknown: large, so "excessive units" never
# false-fires on data whose plausible limits we do not know.
_DEFAULT_MAX_UNITS = 999
_DEFAULT_COMPLEXITY = 2

FEATURE_COLUMNS: list[str] = [
    "fee_ratio",
    "procedure_complexity",
    "units",
    "units_over_max",
    "diagnosis_routine",
    "submit_lag_days",
    "is_temporal_impossible",
    "is_duplicate",
    "provider_claim_count",
    "provider_avg_fee_ratio",
    "billed_zscore_by_code",
    "billed_amount",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns appended."""
    out = df.copy()

    # Coerce dates (they may arrive as strings after a parquet / SQLite round trip).
    out["date_of_service"] = pd.to_datetime(out["date_of_service"], errors="coerce")
    out["date_submitted"] = pd.to_datetime(out["date_submitted"], errors="coerce")

    # allowed_amount is the PER-UNIT reference fee, so the expected total is
    # allowed x units. Dividing by this separates unit-price inflation (a fee
    # outlier) from quantity inflation (phantom / excessive units), which would
    # otherwise be conflated for any multi-unit claim.
    expected_total = (out["allowed_amount"] * out["units"]).replace(0, np.nan)
    out["fee_ratio"] = (out["billed_amount"] / expected_total).fillna(0.0)

    out["procedure_complexity"] = out["procedure_code"].map(_CODE_COMPLEXITY).fillna(_DEFAULT_COMPLEXITY).astype(int)
    max_units = out["procedure_code"].map(_CODE_MAX_UNITS).fillna(_DEFAULT_MAX_UNITS).astype(int)
    out["units_over_max"] = (out["units"] - max_units).clip(lower=0)

    out["diagnosis_routine"] = out["diagnosis_code"].isin(ROUTINE_DIAGNOSES)

    out["submit_lag_days"] = (out["date_submitted"] - out["date_of_service"]).dt.days
    out["is_temporal_impossible"] = out["submit_lag_days"] < 0

    # Duplicate: same member + provider + code + service date seen earlier.
    dup_key = ["claimant_id", "provider_id", "procedure_code", "date_of_service"]
    out["is_duplicate"] = out.duplicated(subset=dup_key, keep="first")

    # Provider aggregates (label-free).
    grp = out.groupby("provider_id")
    out["provider_claim_count"] = grp["claim_id"].transform("count")
    out["provider_avg_fee_ratio"] = grp["fee_ratio"].transform("mean")

    # Within-code billed-amount z-score: how unusual is this charge for this code.
    code_grp = out.groupby("procedure_code")["billed_amount"]
    mean = code_grp.transform("mean")
    std = code_grp.transform("std").replace(0, np.nan)
    out["billed_zscore_by_code"] = ((out["billed_amount"] - mean) / std).fillna(0.0)

    return out


def feature_matrix(df_with_features: pd.DataFrame) -> pd.DataFrame:
    """Select the numeric feature columns for the ML models, in a stable order.

    Booleans are cast to int so scikit-learn sees a clean numeric matrix.
    """
    x = df_with_features[FEATURE_COLUMNS].copy()
    for col in ("diagnosis_routine", "is_temporal_impossible", "is_duplicate"):
        x[col] = x[col].astype(int)
    return x.fillna(0.0)
