"""Storage layer: local for the POC, S3-ready for AWS.

For a one-day POC, local SQLite and Parquet are the right call: zero setup, easy
to inspect, and good enough for the data sizes involved. The same code is
"S3-ready" because the only cloud touchpoint is a single put/get helper that
honours AWS_ENDPOINT_URL, so it works unchanged against real AWS or a local
emulator (LocalEmu / LocalStack) at http://localhost:4566.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path("data/claimguard.sqlite")
DEFAULT_PARQUET = Path("data/claims.parquet")


# --- Local: Parquet (columnar, the natural S3 landing format) ---

def write_parquet(df: pd.DataFrame, path: Path | str = DEFAULT_PARQUET) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path | str = DEFAULT_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


# --- Local: SQLite (easy to query and inspect) ---

def write_sqlite(df: pd.DataFrame, db_path: Path | str = DEFAULT_DB, table: str = "claims") -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Dates are stored as ISO strings so SQLite round-trips cleanly.
    safe = df.copy()
    for col in ("date_of_service", "date_submitted"):
        if col in safe.columns:
            safe[col] = safe[col].astype(str)
    with sqlite3.connect(db_path) as conn:
        safe.to_sql(table, conn, if_exists="replace", index=False)
    return db_path


def read_sqlite(db_path: Path | str = DEFAULT_DB, table: str = "claims") -> pd.DataFrame:
    with sqlite3.connect(Path(db_path)) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


# --- Cloud: a single S3 touchpoint (endpoint-configurable) ---

def _s3_client():
    """boto3 S3 client that targets real AWS or a local emulator.

    Set AWS_ENDPOINT_URL=http://localhost:4566 to use LocalEmu / LocalStack;
    leave it unset for real AWS.
    """
    import boto3  # noqa: PLC0415 - optional dependency, imported on use

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or None
    region = os.environ.get("AWS_REGION", "ca-central-1")
    return boto3.client("s3", endpoint_url=endpoint, region_name=region)


def upload_file_to_s3(local_path: Path | str, bucket: str, key: str) -> str:
    """Upload a file to S3 (or the emulator). Returns the s3:// URI."""
    client = _s3_client()
    client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def download_s3_to_file(bucket: str, key: str, local_path: Path | str) -> Path:
    client = _s3_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(local_path))
    return local_path
