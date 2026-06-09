"""Validation and storage."""

from __future__ import annotations

import pandas as pd

from claimguard.pipeline.store import (
    read_parquet,
    read_sqlite,
    write_parquet,
    write_sqlite,
)
from claimguard.pipeline.validate import validate_claims


def test_valid_claims_pass(claims_df):
    report = validate_claims(claims_df)
    assert report.ok, report.summary()


def test_missing_column_fails(claims_df):
    bad = claims_df.drop(columns=["billed_amount"])
    report = validate_claims(bad)
    assert not report.ok


def test_duplicate_claim_id_fails(claims_df):
    bad = pd.concat([claims_df.head(2), claims_df.head(1)], ignore_index=True)
    report = validate_claims(bad)
    assert not report.ok


def test_negative_amount_fails(claims_df):
    bad = claims_df.copy()
    bad.loc[bad.index[0], "billed_amount"] = -5.0
    report = validate_claims(bad)
    assert not report.ok


def test_parquet_roundtrip(claims_df, tmp_path):
    path = write_parquet(claims_df, tmp_path / "c.parquet")
    back = read_parquet(path)
    assert len(back) == len(claims_df)


def test_sqlite_roundtrip(claims_df, tmp_path):
    path = write_sqlite(claims_df, tmp_path / "c.sqlite")
    back = read_sqlite(path)
    assert len(back) == len(claims_df)
    assert "claim_id" in back.columns
