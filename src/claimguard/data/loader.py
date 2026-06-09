"""Data loader: the open-source path with an automatic synthetic fallback.

This is the second of ClaimGuard's two interchangeable data sources. Both emit
the same canonical claim schema, so everything downstream is identical whether
the data is real-open or synthetic.

Priority ladder (mirrors the research data strategy):
  1. Open-source dataset (Kaggle "Healthcare Provider Fraud Detection Analysis",
     uploader rohitrox: real ICD-10 / CPT codes and provider-level fraud labels).
     Requires a Kaggle token (KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json).
  2. Synthetic generator (no token, no PII, deterministic, ground-truth labels).

If the open dataset cannot be fetched (no token, offline, licence gate), the
loader logs the reason and falls back to synthetic so the pipeline never breaks.

Manual download fallback: if kagglehub is not authenticated, download the four
CSVs from
https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
and point CLAIMGUARD_OPEN_DATA_DIR at the unzipped folder.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import pandas as pd

from claimguard.data.schema import CLAIM_COLUMNS
from claimguard.data.synthetic import GeneratorConfig, generate_claims

logger = logging.getLogger("claimguard.loader")

OPEN_DATASET = "rohitrox/healthcare-provider-fraud-detection-analysis"


def load_synthetic(n_claims: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate canonical synthetic claims (the always-available path)."""
    return generate_claims(GeneratorConfig(n_claims=n_claims, seed=seed))


def _find_open_data_dir() -> Path | None:
    """Locate an already-downloaded copy of the open dataset, if any."""
    env_dir = os.environ.get("CLAIMGUARD_OPEN_DATA_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)
    return None


def _adapt_rohitrox(data_dir: Path) -> pd.DataFrame:
    """Map the rohitrox Medicare CSVs into the canonical claim schema.

    The dataset labels fraud at the PROVIDER level, so we propagate each
    provider's label to all of their claims. Real codes are kept; there is no
    fee-guide column, so allowed_amount is set equal to billed_amount (which
    means the fee-ratio rules stay quiet on this source and the supervised model
    carries the load, an honest property to mention in an interview).
    """
    def _first_csv(pattern: str) -> Path:
        matches = glob.glob(str(data_dir / pattern))
        if not matches:
            raise FileNotFoundError(f"Expected a CSV matching {pattern} in {data_dir}")
        return Path(sorted(matches)[0])

    labels = pd.read_csv(_first_csv("Train-*.csv"))
    label_map = dict(zip(labels["Provider"], labels["PotentialFraud"].map({"Yes": 1, "No": 0})))

    frames = []
    for pattern, pos in (("Train_Inpatientdata-*.csv", "hospital"), ("Train_Outpatientdata-*.csv", "clinic")):
        raw = pd.read_csv(_first_csv(pattern))
        proc_cols = [c for c in raw.columns if c.startswith("ClmProcedureCode_")]
        diag_cols = [c for c in raw.columns if c.startswith("ClmDiagnosisCode_")]
        out = pd.DataFrame()
        out["claim_id"] = raw["ClaimID"]
        out["claimant_id"] = raw["BeneID"]
        out["provider_id"] = raw["Provider"]
        out["provider_specialty"] = "Unknown"
        out["claim_type"] = "medical"
        out["procedure_code"] = raw[proc_cols].bfill(axis=1).iloc[:, 0].fillna("NONE").astype(str) if proc_cols else "NONE"
        out["diagnosis_code"] = raw[diag_cols].bfill(axis=1).iloc[:, 0] if diag_cols else None
        out["units"] = 1
        out["billed_amount"] = pd.to_numeric(raw["InscClaimAmtReimbursed"], errors="coerce").fillna(0.0)
        out["allowed_amount"] = out["billed_amount"]
        out["paid_amount"] = out["billed_amount"]
        out["date_of_service"] = pd.to_datetime(raw["ClaimStartDt"], errors="coerce").dt.date
        out["date_submitted"] = pd.to_datetime(raw["ClaimEndDt"], errors="coerce").dt.date
        out["place_of_service"] = pos
        out["region"] = "US"
        out["is_fraud"] = raw["Provider"].map(label_map).fillna(0).astype(int)
        out["fraud_type"] = None
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    return combined[CLAIM_COLUMNS]


def load_open_dataset() -> pd.DataFrame:
    """Fetch and adapt the open dataset. Raises if it cannot be obtained."""
    data_dir = _find_open_data_dir()
    if data_dir is None:
        # Try kagglehub (needs a token). Import locally so the dependency is optional.
        import kagglehub  # noqa: PLC0415

        path = kagglehub.dataset_download(OPEN_DATASET)
        data_dir = Path(path)
    logger.info("Loading open dataset from %s", data_dir)
    return _adapt_rohitrox(data_dir)


def load_claims(source: str = "auto", n_claims: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Load canonical claims from the requested source.

    source: "synthetic" | "open" | "auto" (open first, then synthetic fallback).
    Returns a DataFrame in the canonical schema. Provenance is logged.
    """
    if source == "synthetic":
        logger.info("Data source: synthetic (%d claims, seed=%d)", n_claims, seed)
        return load_synthetic(n_claims=n_claims, seed=seed)

    if source in ("open", "auto"):
        try:
            df = load_open_dataset()
            logger.info("Data source: open dataset (%d claims)", len(df))
            return df
        except Exception as exc:  # noqa: BLE001 - fallback is the whole point
            if source == "open":
                raise
            logger.warning("Open dataset unavailable (%s); falling back to synthetic.", exc)
            return load_synthetic(n_claims=n_claims, seed=seed)

    raise ValueError(f"Unknown source: {source!r}. Use 'synthetic', 'open', or 'auto'.")
