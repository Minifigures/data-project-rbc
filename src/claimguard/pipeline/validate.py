"""Schema and data-quality validation.

A fraud pipeline that scores garbage produces garbage decisions, so validation
is a gate, not an afterthought. This runs cheap vectorised checks over the whole
table and a strict pydantic check over a sample, and returns a structured report
the pipeline can act on (and that can be logged for audit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from claimguard.data.schema import CLAIM_COLUMNS, Claim


@dataclass
class ValidationReport:
    n_rows: int
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        line = f"Validation {status}: {self.n_rows} rows, {len(self.errors)} errors, {len(self.warnings)} warnings"
        return "\n".join([line, *[f"  ERROR: {e}" for e in self.errors], *[f"  WARN:  {w}" for w in self.warnings]])


def validate_claims(df: pd.DataFrame, sample_size: int = 200) -> ValidationReport:
    """Validate a claims table against the canonical schema and quality rules."""
    report = ValidationReport(n_rows=len(df))

    # --- Structural: required columns present ---
    missing = [c for c in CLAIM_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {missing}")
        report.ok = False
        return report  # cannot run value checks without the columns

    if df.empty:
        report.errors.append("Claims table is empty.")
        report.ok = False
        return report

    # --- Value checks (vectorised) ---
    if df["claim_id"].duplicated().any():
        n = int(df["claim_id"].duplicated().sum())
        report.errors.append(f"{n} duplicate claim_id values (claim_id must be unique).")
        report.ok = False

    if df["claim_id"].isna().any():
        report.errors.append("Null claim_id values present.")
        report.ok = False

    for money_col in ("billed_amount", "allowed_amount"):
        if (pd.to_numeric(df[money_col], errors="coerce") < 0).any():
            report.errors.append(f"Negative values in {money_col}.")
            report.ok = False
        if pd.to_numeric(df[money_col], errors="coerce").isna().any():
            report.warnings.append(f"Non-numeric or null values in {money_col} were coerced.")

    # Zero reference fee makes the fee ratio meaningless; warn, do not fail.
    if (pd.to_numeric(df["allowed_amount"], errors="coerce").fillna(0) == 0).any():
        n = int((pd.to_numeric(df["allowed_amount"], errors="coerce").fillna(0) == 0).sum())
        report.warnings.append(f"{n} rows have allowed_amount == 0; fee-ratio rules will not fire for them.")

    # --- Strict per-row schema check on a sample (pydantic) ---
    sample = df.head(sample_size).to_dict("records")
    bad = 0
    for rec in sample:
        rec = {k: (None if pd.isna(v) else v) for k, v in rec.items() if k in CLAIM_COLUMNS}
        try:
            Claim(**rec)
        except Exception:  # noqa: BLE001 - we count, not raise
            bad += 1
    if bad:
        report.errors.append(f"{bad}/{len(sample)} sampled rows failed strict schema validation.")
        report.ok = False

    return report
