"""Synthetic free-text claim notes and a deterministic field extractor.

A real adjuster reviews a claim *document*, not a tidy table row. This module
renders each PII-free claim as a short free-text note (the "paper" a reviewer
reads) and provides a deterministic parser that recovers the structured fields
from that note.

The parser is the no-LLM fallback for the document-perception demo: with an
``ANTHROPIC_API_KEY`` set, the LLM layer (``detection.llm_perception``) does the
extraction; without one, this recovers the same fields so the demo always runs.
Either way, perception stays OUT of the deterministic score.

Privacy: a note is built only from codes, amounts, units, dates, specialty,
place, and region. It carries no name, address, or date of birth, because the
claim schema has none to begin with.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

# The non-identifying fields a note exposes and the parser recovers. Mirrors the
# field set the LLM layer is asked to extract (see detection.llm_perception).
EXTRACT_FIELDS: tuple[str, ...] = (
    "provider_specialty",
    "procedure_code",
    "diagnosis_code",
    "units",
    "billed_amount",
    "allowed_amount",
    "date_of_service",
    "date_submitted",
)


def _is_datelike(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    return callable(getattr(value, "strftime", None))


def _fmt_date(value: Any) -> str:
    """Format a date-like value as ISO YYYY-MM-DD; trim an existing ISO string."""
    if isinstance(value, str):
        return value[:10]
    if _is_datelike(value):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _fmt_amount(value: Any) -> str:
    return f"{float(value):.2f}"


def render_claim_note(claim: Mapping[str, Any]) -> str:
    """Render a deterministic, PII-free free-text adjuster note for one claim.

    The same claim always yields the same note (no randomness), so notes are
    reproducible and testable. The note embeds the claim's non-identifying fields
    in plain language for a reviewer to read and for the perception layer to
    extract.
    """
    claim_id = str(claim.get("claim_id", "UNKNOWN"))
    specialty = str(claim.get("provider_specialty", ""))
    provider = str(claim.get("provider_id", ""))
    claim_type = str(claim.get("claim_type", "claim"))
    proc = str(claim.get("procedure_code", ""))
    diagnosis = claim.get("diagnosis_code")
    units = int(claim.get("units", 1) or 1)
    billed = _fmt_amount(claim.get("billed_amount", 0.0))
    allowed = _fmt_amount(claim.get("allowed_amount", 0.0))
    dos = _fmt_date(claim.get("date_of_service", ""))
    dsub = _fmt_date(claim.get("date_submitted", ""))
    place = str(claim.get("place_of_service", "clinic"))
    region = str(claim.get("region", ""))

    unit_word = "unit" if units == 1 else "units"
    dx_clause = f" with diagnosis {diagnosis}" if diagnosis not in (None, "", "nan") else ""

    lines = [
        f"Claim note for {claim_id} ({claim_type}).",
        (
            f"Provider {provider}, specialty {specialty}, rendered {units} {unit_word} "
            f"of procedure {proc}{dx_clause} at a {place} in {region}."
        ),
        f"Date of service {dos}; date submitted {dsub}.",
        f"Billed amount {billed}; reference allowed amount {allowed}.",
    ]

    # An observational reviewer sentence derived ONLY from claim facts (never the
    # hidden fraud label), worded to avoid the parser's anchor phrases.
    observations = _observations(
        billed=float(claim.get("billed_amount", 0.0) or 0.0),
        allowed=float(claim.get("allowed_amount", 0.0) or 0.0),
        units=units,
        dos=dos,
        dsub=dsub,
    )
    if observations:
        lines.append("Reviewer note: " + " ".join(observations))

    return "\n".join(lines)


def _observations(*, billed: float, allowed: float, units: int, dos: str, dsub: str) -> list[str]:
    obs: list[str] = []
    expected = allowed * max(units, 1)
    unit_word = "unit" if units == 1 else "units"
    if expected > 0 and billed > expected * 1.5:
        obs.append(f"The billed total runs {billed / expected:.1f}x the reference for {units} {unit_word}.")
    if units >= 5:
        obs.append(f"High unit count ({units}) for a single service day.")
    if dos and dsub and dsub < dos:
        obs.append("Submission predates the service date, which cannot happen legitimately.")
    return obs


def extract_fields_deterministic(note: str) -> dict[str, Any]:
    """Recover claim fields from a rendered note with regexes (LLM-free).

    Returns whatever subset of ``EXTRACT_FIELDS`` it can parse, in the same shape
    ``detection.llm_perception.extract_claim_fields`` returns, so the dashboard can
    use either path interchangeably.
    """
    out: dict[str, Any] = {}
    patterns: dict[str, str] = {
        "provider_specialty": r"specialty ([A-Za-z &]+?),",
        "procedure_code": r"procedure ([A-Z]{2}-\d+)",
        "diagnosis_code": r"diagnosis ([A-Z]\d{2}(?:\.\d+)?)",
        "date_of_service": r"[Dd]ate of service (\d{4}-\d{2}-\d{2})",
        "date_submitted": r"date submitted (\d{4}-\d{2}-\d{2})",
    }
    for field_name, pattern in patterns.items():
        match = re.search(pattern, note)
        if match:
            out[field_name] = match.group(1).strip()

    units_match = re.search(r"rendered (\d+) units?", note)
    if units_match:
        out["units"] = int(units_match.group(1))

    billed_match = re.search(r"[Bb]illed amount (\d+(?:\.\d+)?)", note)
    if billed_match:
        out["billed_amount"] = float(billed_match.group(1))

    allowed_match = re.search(r"allowed amount (\d+(?:\.\d+)?)", note)
    if allowed_match:
        out["allowed_amount"] = float(allowed_match.group(1))

    return out


def values_match(record_value: Any, extracted_value: Any) -> bool:
    """Tolerant equality for the extraction comparison (numbers, dates, strings)."""
    if extracted_value is None or record_value is None:
        return False
    try:
        return abs(float(record_value) - float(extracted_value)) < 0.01
    except (TypeError, ValueError):
        pass
    extracted_str = str(extracted_value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", extracted_str) and _is_datelike(record_value):
        return _fmt_date(record_value) == extracted_str
    return str(record_value).strip() == extracted_str
