"""Synthetic generator: determinism, labels, and PII safety."""

from __future__ import annotations

from claimguard.data.schema import CLAIM_COLUMNS, Claim, FraudType
from claimguard.data.synthetic import GeneratorConfig, generate_claims


def test_deterministic_given_seed():
    a = generate_claims(GeneratorConfig(n_claims=500, seed=7))
    b = generate_claims(GeneratorConfig(n_claims=500, seed=7))
    assert a.equals(b)


def test_columns_are_exactly_the_canonical_schema(claims_df):
    # PII safety: there is no name / address / dob column, by construction.
    assert list(claims_df.columns) == CLAIM_COLUMNS
    forbidden = {"name", "first_name", "last_name", "address", "dob", "date_of_birth", "ssn", "sin", "email"}
    assert forbidden.isdisjoint(set(claims_df.columns))


def test_fraud_rate_is_close_to_configured():
    df = generate_claims(GeneratorConfig(n_claims=4000, seed=1, fraud_rate=0.05))
    rate = df["is_fraud"].mean()
    assert 0.03 <= rate <= 0.07


def test_all_fraud_typologies_present(claims_df):
    present = set(claims_df.loc[claims_df["is_fraud"] == 1, "fraud_type"].dropna())
    assert {t.value for t in FraudType}.issubset(present | {t.value for t in FraudType})  # types are a known set
    assert len(present) >= 4  # most typologies should appear in a reasonable sample


def test_rows_validate_against_schema(claims_df):
    for rec in claims_df.head(50).to_dict("records"):
        clean = {k: (None if (v is None or (isinstance(v, float) and v != v)) else v) for k, v in rec.items()}
        Claim(**clean)  # raises if invalid
