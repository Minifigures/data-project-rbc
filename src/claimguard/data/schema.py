"""Canonical claim contract for ClaimGuard.

Everything in the system speaks this one schema. The synthetic generator and the
open-source dataset loader both emit ``Claim`` records, which is what makes the two
data sources interchangeable.

Privacy is structural here, not a policy. There is deliberately NO field for a
name, address, date of birth, phone number, or any free-text PII. A claim is
identified only by pseudonymous IDs, procedure / diagnosis codes, amounts, and
dates. The system cannot leak personal information because the contract has
nowhere to store it.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt


class ClaimType(str, Enum):
    """Line of business for a claim. Medical and dental mirror the candidate's
    prior insurance-fraud work and the primary open dataset's domain."""

    MEDICAL = "medical"
    DENTAL = "dental"
    PHARMACY = "pharmacy"
    AUTO = "auto"


class FraudType(str, Enum):
    """Injected fraud typologies, aligned with NHCAA / CHCAA categories.

    These are ground-truth labels used ONLY for synthetic data and for
    measuring detector quality. Real production claims arrive unlabelled.
    """

    UPCODING = "upcoding"               # billing a higher-value code than performed
    DUPLICATE_BILLING = "duplicate_billing"   # same service billed more than once
    PHANTOM_BILLING = "phantom_billing"       # services never rendered / excessive units
    IMPOSSIBLE_SEQUENCE = "impossible_sequence"  # care timeline that cannot happen
    FEE_OUTLIER = "fee_outlier"         # amount far above the reference fee


class Claim(BaseModel):
    """A single insurance claim line. PII-free by construction.

    ``is_fraud`` and ``fraud_type`` are present only so we can train and
    evaluate detectors on labelled data. In production these are unknown and
    are exactly what the system is trying to predict.
    """

    model_config = {"extra": "forbid"}

    # --- Identity (pseudonymous only) ---
    claim_id: str = Field(..., description="Unique claim identifier, e.g. CLM-000001")
    claimant_id: str = Field(..., description="Pseudonymous member id, e.g. MBR-00042")
    provider_id: str = Field(..., description="Pseudonymous provider id, e.g. PRV-001")
    provider_specialty: str = Field(..., description="Provider specialty / category")

    # --- Clinical / billing content ---
    claim_type: ClaimType = ClaimType.MEDICAL
    procedure_code: str = Field(..., description="Primary procedure / service code (CPT/CDA-like)")
    diagnosis_code: str | None = Field(None, description="Primary diagnosis code (ICD-10-like)")
    units: NonNegativeInt = Field(1, description="Number of units / sessions billed")

    # --- Money (CAD) ---
    billed_amount: NonNegativeFloat = Field(..., description="Amount the provider billed")
    allowed_amount: NonNegativeFloat = Field(..., description="Reference / fee-guide amount for this code")
    paid_amount: NonNegativeFloat | None = Field(None, description="Amount actually paid, if adjudicated")

    # --- Context ---
    date_of_service: date
    date_submitted: date
    place_of_service: str = Field("clinic", description="Setting, e.g. clinic, hospital, pharmacy")
    region: str = Field("ON", description="Province / region code")

    # --- Ground truth (synthetic / labelled data only) ---
    is_fraud: int = Field(0, ge=0, le=1, description="1 if known-fraudulent (labelled data only)")
    fraud_type: FraudType | None = Field(None, description="Which typology, if labelled fraud")

    @property
    def fee_ratio(self) -> float:
        """Billed over reference. > 1 means billed above the fee guide.

        Guarded against a zero reference so the property never raises.
        """
        if self.allowed_amount <= 0:
            return 0.0
        return self.billed_amount / self.allowed_amount


# Column order used whenever claims are written to a flat table (parquet / SQLite / CSV).
CLAIM_COLUMNS: list[str] = [
    "claim_id",
    "claimant_id",
    "provider_id",
    "provider_specialty",
    "claim_type",
    "procedure_code",
    "diagnosis_code",
    "units",
    "billed_amount",
    "allowed_amount",
    "paid_amount",
    "date_of_service",
    "date_submitted",
    "place_of_service",
    "region",
    "is_fraud",
    "fraud_type",
]
