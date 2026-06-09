"""Synthetic insurance-claim generator with injected, labelled fraud.

Why this exists: in an insurance setting you usually cannot touch real customer
claims (PII, privacy law, model-risk sign-off). A synthetic generator gives you
data that is safe to share, fully reproducible, and carries GROUND-TRUTH labels
so you can measure a detector honestly.

The generator is deterministic given a seed, so the exact dataset behind any
model run can be regenerated for audit. Each injected fraud typology is grounded
in an NHCAA / CHCAA category and is built to leave a transparent, explainable
signal that the rule engine can catch, while the unsupervised model is there to
catch combinations we did not hand-code.

Codes here are SYNTHETIC and shaped like real CPT / CDA / ICD-10 codes. They are
not real billing codes and the fees are invented, which keeps the data honest
and free of licensing concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from claimguard.data.schema import CLAIM_COLUMNS, ClaimType, FraudType


@dataclass(frozen=True)
class ProcedureCode:
    code: str
    description: str
    base_fee: float          # reference / fee-guide amount in CAD
    complexity: int          # 1 = routine, 3 = top tier
    max_units: int           # plausible maximum units billable in one day


# Synthetic fee schedule. Maps a provider specialty to its billable procedures.
CODE_BOOK: dict[str, list[ProcedureCode]] = {
    "General Practice": [
        ProcedureCode("GP-101", "Brief office visit", 45, 1, 1),
        ProcedureCode("GP-102", "Standard office visit", 75, 2, 1),
        ProcedureCode("GP-103", "Complex consultation", 180, 3, 1),
    ],
    "Physiotherapy": [
        ProcedureCode("PT-201", "Physio session 30 min", 60, 1, 2),
        ProcedureCode("PT-202", "Physio session 60 min", 100, 2, 1),
        ProcedureCode("PT-203", "Extended rehab programme", 220, 3, 1),
    ],
    "Dental": [
        ProcedureCode("DN-301", "Routine cleaning", 120, 1, 1),
        ProcedureCode("DN-302", "Filling", 200, 2, 4),
        ProcedureCode("DN-303", "Root canal", 900, 3, 1),
    ],
    "Radiology": [
        ProcedureCode("RD-401", "X-ray", 90, 1, 3),
        ProcedureCode("RD-402", "Ultrasound", 220, 2, 1),
        ProcedureCode("RD-403", "MRI scan", 700, 3, 1),
    ],
    "Cardiology": [
        ProcedureCode("CD-501", "ECG", 110, 1, 1),
        ProcedureCode("CD-502", "Stress test", 350, 2, 1),
        ProcedureCode("CD-503", "Echocardiogram", 600, 3, 1),
    ],
    "Pharmacy": [
        ProcedureCode("PH-601", "Generic dispense", 25, 1, 3),
        ProcedureCode("PH-602", "Brand dispense", 90, 2, 3),
        ProcedureCode("PH-603", "Specialty drug", 450, 3, 1),
    ],
}

SPECIALTY_TO_CLAIM_TYPE: dict[str, ClaimType] = {
    "General Practice": ClaimType.MEDICAL,
    "Physiotherapy": ClaimType.MEDICAL,
    "Dental": ClaimType.DENTAL,
    "Radiology": ClaimType.MEDICAL,
    "Cardiology": ClaimType.MEDICAL,
    "Pharmacy": ClaimType.PHARMACY,
}

# Synthetic ICD-10-shaped diagnosis codes, split by clinical acuity. A top-tier
# (complexity 3) procedure paired with a ROUTINE diagnosis is the upcoding tell.
ROUTINE_DIAGNOSES = ["Z00.0", "M54.5", "J06.9", "K02.9", "Z23", "R51"]
ACUTE_DIAGNOSES = ["I21.9", "J18.9", "S72.0", "I50.9", "N17.9"]


@dataclass
class GeneratorConfig:
    n_claims: int = 5000
    fraud_rate: float = 0.05          # share of claims that are fraudulent (rare, like reality)
    n_providers: int = 80
    n_claimants: int = 1500
    seed: int = 42
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2025, 12, 31)
    # Fraud is concentrated: this share of providers are "bad actors" who carry
    # the bulk of the fraud, so provider-level features carry real signal.
    bad_actor_provider_share: float = 0.12
    bad_actor_fraud_concentration: float = 0.6
    # Share of fee-outlier fraud made "subtle" (signal just under the rule
    # threshold). These evade the deterministic rules so the ML layer has to
    # earn its keep, which keeps the synthetic evaluation honest.
    subtle_fraud_share: float = 0.35
    fraud_type_weights: dict[FraudType, float] = field(
        default_factory=lambda: {
            FraudType.FEE_OUTLIER: 0.30,
            FraudType.UPCODING: 0.25,
            FraudType.DUPLICATE_BILLING: 0.20,
            FraudType.PHANTOM_BILLING: 0.15,
            FraudType.IMPOSSIBLE_SEQUENCE: 0.10,
        }
    )


def _random_date(rng: np.random.Generator, start: date, end: date) -> date:
    span = (end - start).days
    return start + timedelta(days=int(rng.integers(0, span + 1)))


def generate_claims(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate a labelled synthetic claims table.

    Returns a DataFrame with the canonical claim columns plus ``is_fraud`` and
    ``fraud_type`` ground-truth labels. Deterministic for a given config.seed.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    specialties = list(CODE_BOOK.keys())

    # --- Build the provider population ---
    provider_ids = [f"PRV-{i:03d}" for i in range(cfg.n_providers)]
    provider_specialty = {
        pid: specialties[int(rng.integers(0, len(specialties)))] for pid in provider_ids
    }
    n_bad = max(1, int(cfg.n_providers * cfg.bad_actor_provider_share))
    bad_actor_ids = set(rng.choice(provider_ids, size=n_bad, replace=False).tolist())

    claimant_ids = [f"MBR-{i:05d}" for i in range(cfg.n_claimants)]

    # --- Decide which claim indices are fraudulent, biased toward bad actors ---
    n_fraud = int(round(cfg.n_claims * cfg.fraud_rate))
    fraud_types = list(cfg.fraud_type_weights.keys())
    fraud_probs = np.array([cfg.fraud_type_weights[t] for t in fraud_types], dtype=float)
    fraud_probs = fraud_probs / fraud_probs.sum()

    rows: list[dict] = []

    # Pre-assign providers per claim so we can bias fraud toward bad actors.
    weights = np.array(
        [
            (cfg.bad_actor_fraud_concentration if pid in bad_actor_ids else (1 - cfg.bad_actor_fraud_concentration))
            for pid in provider_ids
        ]
    )
    weights = weights / weights.sum()
    fraud_provider_pool = rng.choice(provider_ids, size=n_fraud, p=weights).tolist()

    fraud_flags = np.zeros(cfg.n_claims, dtype=int)
    fraud_flags[:n_fraud] = 1
    rng.shuffle(fraud_flags)

    fraud_cursor = 0
    for i in range(cfg.n_claims):
        is_fraud = bool(fraud_flags[i])
        if is_fraud:
            provider_id = fraud_provider_pool[fraud_cursor]
            fraud_cursor += 1
        else:
            provider_id = provider_ids[int(rng.integers(0, cfg.n_providers))]

        specialty = provider_specialty[provider_id]
        codes = CODE_BOOK[specialty]
        claimant_id = claimant_ids[int(rng.integers(0, cfg.n_claimants))]

        # --- Start from a legitimate claim, then mutate if fraudulent ---
        proc = codes[int(rng.integers(0, len(codes)))]
        diagnosis = (
            ACUTE_DIAGNOSES[int(rng.integers(0, len(ACUTE_DIAGNOSES)))]
            if proc.complexity == 3
            else ROUTINE_DIAGNOSES[int(rng.integers(0, len(ROUTINE_DIAGNOSES)))]
        )
        # Some legitimate claims bill multiple units (e.g. several fillings).
        if proc.max_units > 1 and rng.random() < 0.3:
            units = int(rng.integers(1, proc.max_units + 1))
        else:
            units = 1
        allowed = proc.base_fee  # per-unit reference fee
        # Legit billing wobbles a little around units x the fee guide.
        billed = round(allowed * units * float(rng.normal(1.0, 0.04)), 2)
        d_service = _random_date(rng, cfg.start_date, cfg.end_date)
        d_submit = d_service + timedelta(days=int(rng.integers(1, 21)))
        fraud_type: FraudType | None = None
        duplicate_of: dict | None = None

        if is_fraud:
            fraud_type = fraud_types[int(rng.choice(len(fraud_types), p=fraud_probs))]

            if fraud_type == FraudType.FEE_OUTLIER:
                # Bill above the fee guide (unit-price inflation). A share are
                # "subtle": just under the 1.5x rule threshold, so only the ML
                # layer, seeing the within-code z-score, has a chance to catch them.
                if rng.random() < cfg.subtle_fraud_share:
                    mult = float(rng.uniform(1.2, 1.48))
                else:
                    mult = float(rng.uniform(1.6, 3.2))
                billed = round(allowed * units * mult, 2)

            elif fraud_type == FraudType.UPCODING:
                # Bill the top-tier code but with a routine diagnosis (mismatch).
                # The fee itself looks normal; the tell is the code/diagnosis pairing.
                top = max(codes, key=lambda c: c.complexity)
                proc = top
                units = 1
                allowed = top.base_fee
                billed = round(top.base_fee * float(rng.normal(1.05, 0.03)), 2)
                diagnosis = ROUTINE_DIAGNOSES[int(rng.integers(0, len(ROUTINE_DIAGNOSES)))]

            elif fraud_type == FraudType.PHANTOM_BILLING:
                # Implausible number of units in a single day.
                units = int(rng.integers(proc.max_units + 3, proc.max_units * 4 + 5))
                billed = round(allowed * units * float(rng.normal(1.0, 0.03)), 2)

            elif fraud_type == FraudType.IMPOSSIBLE_SEQUENCE:
                # Service dated AFTER it was submitted: cannot happen legitimately.
                d_submit = d_service - timedelta(days=int(rng.integers(2, 30)))

            elif fraud_type == FraudType.DUPLICATE_BILLING:
                # Mark this row; we will clone an earlier legit row's key below.
                duplicate_of = {"flag": True}

        row = {
            "claim_id": f"CLM-{i:06d}",
            "claimant_id": claimant_id,
            "provider_id": provider_id,
            "provider_specialty": specialty,
            "claim_type": SPECIALTY_TO_CLAIM_TYPE[specialty].value,
            "procedure_code": proc.code,
            "diagnosis_code": diagnosis,
            "units": units,
            "billed_amount": max(0.0, billed),
            "allowed_amount": allowed,
            "paid_amount": None,
            "date_of_service": d_service,
            "date_submitted": d_submit,
            "place_of_service": "clinic",
            "region": "ON",
            "is_fraud": int(is_fraud),
            "fraud_type": fraud_type.value if fraud_type else None,
        }

        # Duplicate billing: re-use an existing legit claim's identifying key so
        # the (claimant, provider, code, service-date) tuple collides on purpose.
        if duplicate_of and rows:
            legit_rows = [r for r in rows if r["is_fraud"] == 0]
            if legit_rows:
                base = legit_rows[int(rng.integers(0, len(legit_rows)))]
                row["claimant_id"] = base["claimant_id"]
                row["provider_id"] = base["provider_id"]
                row["provider_specialty"] = base["provider_specialty"]
                row["claim_type"] = base["claim_type"]
                row["procedure_code"] = base["procedure_code"]
                row["diagnosis_code"] = base["diagnosis_code"]
                row["units"] = base["units"]
                row["billed_amount"] = base["billed_amount"]
                row["allowed_amount"] = base["allowed_amount"]
                row["date_of_service"] = base["date_of_service"]
                row["date_submitted"] = base["date_submitted"] + timedelta(days=int(rng.integers(1, 10)))

        rows.append(row)

    df = pd.DataFrame(rows, columns=CLAIM_COLUMNS)
    # Stable, realistic ordering: by submission date.
    df = df.sort_values("date_submitted").reset_index(drop=True)
    return df


def fraud_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick label breakdown, handy for the README and for sanity checks."""
    total = len(df)
    frauds = int(df["is_fraud"].sum())
    by_type = (
        df[df["is_fraud"] == 1]["fraud_type"].value_counts().rename_axis("fraud_type").reset_index(name="count")
    )
    overall = pd.DataFrame(
        {
            "metric": ["total_claims", "fraud_claims", "fraud_rate"],
            "value": [total, frauds, round(frauds / total, 4) if total else 0.0],
        }
    )
    return overall, by_type


if __name__ == "__main__":
    # Quick manual check: python -m claimguard.data.synthetic
    frame = generate_claims()
    overall, by_type = fraud_summary(frame)
    print(overall.to_string(index=False))
    print()
    print(by_type.to_string(index=False))
