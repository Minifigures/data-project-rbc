"""Immutable, tamper-evident audit log.

Every scoring event and every human decision is appended here, never updated or
deleted. Each record carries the hash of the record before it (a hash chain), so
if anyone edits a past entry the chain no longer verifies. That turns "trust us,
the log is accurate" into something you can actually check with verify_chain().

No personal information is ever written: only claim IDs, scores, rule codes,
decisions, and timestamps. Locally this is SQLite; in AWS the same records map to
a DynamoDB table (see aws/lambda_handler.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _default_audit_db() -> Path:
    return Path(os.environ.get("CLAIMGUARD_AUDIT_DB", "data/audit_log.sqlite"))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _record_hash(seq: int, ts: str, event_type: str, claim_id: str, payload: dict, prev_hash: str) -> str:
    body = json.dumps(
        {"seq": seq, "ts": ts, "event_type": event_type, "claim_id": claim_id, "payload": payload, "prev_hash": prev_hash},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _default_audit_db()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    seq INTEGER PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    claim_id TEXT,
                    payload TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    record_hash TEXT NOT NULL
                )
                """
            )

    def _last(self) -> tuple[int, str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT seq, record_hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
        return (row[0], row[1]) if row else (0, GENESIS_HASH)

    def append(self, event_type: str, claim_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one immutable record and return it."""
        last_seq, prev_hash = self._last()
        seq = last_seq + 1
        ts = _utc_now_iso()
        record_hash = _record_hash(seq, ts, event_type, claim_id, payload, prev_hash)
        record = {
            "seq": seq,
            "record_id": str(uuid.uuid4()),
            "ts": ts,
            "event_type": event_type,
            "claim_id": claim_id,
            "payload": payload,
            "prev_hash": prev_hash,
            "record_hash": record_hash,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (seq, record_id, ts, event_type, claim_id, payload, prev_hash, record_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (seq, record["record_id"], ts, event_type, claim_id, json.dumps(payload, default=str), prev_hash, record_hash),
            )
        return record

    def all(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT seq, record_id, ts, event_type, claim_id, payload, prev_hash, record_hash FROM audit_log ORDER BY seq"
            ).fetchall()
        cols = ["seq", "record_id", "ts", "event_type", "claim_id", "payload", "prev_hash", "record_hash"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            d["payload"] = json.loads(d["payload"])
            out.append(d)
        return out

    def verify_chain(self) -> bool:
        """Recompute every hash. Returns False if any record was tampered with."""
        prev_hash = GENESIS_HASH
        for rec in self.all():
            expected = _record_hash(rec["seq"], rec["ts"], rec["event_type"], rec["claim_id"], rec["payload"], prev_hash)
            if expected != rec["record_hash"] or rec["prev_hash"] != prev_hash:
                return False
            prev_hash = rec["record_hash"]
        return True
