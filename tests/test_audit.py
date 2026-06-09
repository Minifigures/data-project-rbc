"""Immutable, tamper-evident audit log."""

from __future__ import annotations

import json
import sqlite3

from claimguard.api.audit import AuditLog


def test_append_and_chain_valid(tmp_path):
    log = AuditLog(tmp_path / "audit.sqlite")
    log.append("scored", "CLM-1", {"rule_score": 80, "band": "high"})
    log.append("human_review", "CLM-1", {"decision": "flagged"})
    records = log.all()
    assert len(records) == 2
    assert records[0]["prev_hash"] == "0" * 64  # genesis
    assert log.verify_chain() is True


def test_tampering_breaks_the_chain(tmp_path):
    db = tmp_path / "audit.sqlite"
    log = AuditLog(db)
    log.append("scored", "CLM-1", {"rule_score": 80, "band": "high"})
    log.append("scored", "CLM-2", {"rule_score": 10, "band": "low"})
    assert log.verify_chain() is True

    # Tamper: rewrite a past payload directly in the database.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE audit_log SET payload = ? WHERE seq = 1",
            (json.dumps({"rule_score": 0, "band": "low"}),),
        )
    assert log.verify_chain() is False
