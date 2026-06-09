"""Human-in-the-loop review queue.

Claims that score above the auto-pass band land here for a person to decide. The
model never takes an adverse action on its own; it surfaces a scored, explained
case and a human approves, flags, or dismisses it. This is the control Quebec
Law 25 (right to human review) and OSFI E-23 expect for customer-impacting flags.

Backed by SQLite so the queue and its decisions persist across restarts.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_DECISIONS = {"approved", "flagged", "dismissed"}


def _default_queue_db() -> Path:
    return Path(os.environ.get("CLAIMGUARD_QUEUE_DB", "data/review_queue.sqlite"))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ReviewQueue:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _default_queue_db()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_queue (
                    claim_id TEXT PRIMARY KEY,
                    rule_score INTEGER,
                    band TEXT,
                    recommendation TEXT,
                    explanation TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_ts TEXT NOT NULL,
                    decided_ts TEXT,
                    reviewer TEXT,
                    note TEXT,
                    payload TEXT
                )
                """
            )

    def add(self, claim_id: str, rule_score: int, band: str, recommendation: str, explanation: str, payload: dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO review_queue (claim_id, rule_score, band, recommendation, explanation, status, created_ts, payload)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(claim_id) DO UPDATE SET
                    rule_score=excluded.rule_score, band=excluded.band,
                    recommendation=excluded.recommendation, explanation=excluded.explanation,
                    payload=excluded.payload
                """,
                (claim_id, rule_score, band, recommendation, explanation, _utc_now_iso(), json.dumps(payload, default=str)),
            )

    def list(self, status: str = "pending") -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM review_queue WHERE status = ? ORDER BY rule_score DESC, created_ts", (status,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, claim_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM review_queue WHERE claim_id = ?", (claim_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def decide(self, claim_id: str, decision: str, reviewer: str, note: str = "") -> dict[str, Any]:
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE review_queue SET status=?, decided_ts=?, reviewer=?, note=? WHERE claim_id=?",
                (decision, _utc_now_iso(), reviewer, note, claim_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"claim_id {claim_id!r} not found in review queue")
        return self.get(claim_id)  # type: ignore[return-value]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("payload"):
            d["payload"] = json.loads(d["payload"])
        return d
