"""Tests for synthetic claim-note rendering and deterministic field extraction.

The core guarantee: a note rendered from a claim can be parsed back into the same
fields (the document-perception round trip), with no PII in the note.
"""

from __future__ import annotations

from datetime import date

from claimguard.data.claim_note import (
    EXTRACT_FIELDS,
    extract_fields_deterministic,
    render_claim_note,
    values_match,
)


def _sample_claim() -> dict:
    return {
        "claim_id": "CLM-000081",
        "claimant_id": "MBR-00438",
        "provider_id": "PRV-056",
        "provider_specialty": "Radiology",
        "claim_type": "medical",
        "procedure_code": "RD-401",
        "diagnosis_code": "J06.9",
        "units": 1,
        "billed_amount": 235.81,
        "allowed_amount": 90.0,
        "date_of_service": date(2025, 1, 6),
        "date_submitted": date(2025, 1, 8),
        "place_of_service": "clinic",
        "region": "ON",
    }


def test_render_is_deterministic() -> None:
    claim = _sample_claim()
    assert render_claim_note(claim) == render_claim_note(claim)


def test_note_is_pii_free() -> None:
    note = render_claim_note(_sample_claim()).lower()
    for token in ("name", "address", "date of birth", "dob", "phone", "email"):
        assert token not in note


def test_note_contains_key_facts() -> None:
    note = render_claim_note(_sample_claim())
    assert "RD-401" in note
    assert "Radiology" in note
    assert "235.81" in note
    assert "2025-01-06" in note


def test_extraction_round_trips() -> None:
    claim = _sample_claim()
    extracted = extract_fields_deterministic(render_claim_note(claim))
    for field_name in EXTRACT_FIELDS:
        assert field_name in extracted, f"parser dropped {field_name}"
        assert values_match(claim[field_name], extracted[field_name]), field_name


def test_extraction_handles_multi_unit_and_multiword_specialty() -> None:
    claim = _sample_claim()
    claim.update(
        provider_specialty="General Practice",
        procedure_code="GP-103",
        diagnosis_code="Z00.0",
        units=3,
        billed_amount=540.0,
        allowed_amount=180.0,
    )
    extracted = extract_fields_deterministic(render_claim_note(claim))
    assert extracted["provider_specialty"] == "General Practice"
    assert extracted["procedure_code"] == "GP-103"
    assert extracted["units"] == 3
    assert values_match(claim["billed_amount"], extracted["billed_amount"])


def test_missing_diagnosis_is_omitted_not_invented() -> None:
    claim = _sample_claim()
    claim["diagnosis_code"] = None
    extracted = extract_fields_deterministic(render_claim_note(claim))
    assert "diagnosis_code" not in extracted


def test_values_match_handles_types() -> None:
    assert values_match(90.0, 90.0)
    assert values_match(1, 1)
    assert values_match(date(2025, 1, 6), "2025-01-06")
    assert values_match("RD-401", "RD-401")
    assert not values_match("RD-401", None)
    assert not values_match(90.0, 91.0)
