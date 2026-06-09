"""Optional LLM layer: document perception + plain-language rationale.

Two strict design rules, both interview talking points:

1. The LLM is kept OUT of the numeric score. It can extract fields from a messy
   claim note and it can narrate why a claim was flagged, but it never moves the
   0-100 score. Scoring stays deterministic and auditable.
2. It is fully optional. With no ANTHROPIC_API_KEY set, ClaimGuard runs exactly
   the same, and the rationale falls back to a deterministic template built from
   the rule hits. The model is a convenience, never a dependency.

Privacy: only non-identifying fields (codes, amounts, dates) are ever sent to the
model. The canonical claim schema has no name, address, or date of birth to leak.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("claimguard.llm")

# Cheap, fast model for a short narrative. Override with CLAIMGUARD_LLM_MODEL
# (e.g. claude-opus-4-8) if you want a stronger model. Cost is the user's call.
LLM_MODEL = os.environ.get("CLAIMGUARD_LLM_MODEL", "claude-haiku-4-5")


def llm_available() -> bool:
    """True only if a key is set and the SDK imports. Never raises."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


# Fields we will accept from an extracted claim note. No PII fields by design.
_EXTRACT_FIELDS = [
    "provider_specialty",
    "procedure_code",
    "diagnosis_code",
    "units",
    "billed_amount",
    "allowed_amount",
    "date_of_service",
    "date_submitted",
]


def extract_claim_fields(note: str) -> dict:
    """Extract structured claim fields from a free-text claim note.

    Returns a dict of whatever fields the model could find. Returns an empty dict
    (and logs) if the LLM is unavailable, so callers can fall back to manual entry.
    """
    if not llm_available():
        logger.info("LLM unavailable; skipping field extraction.")
        return {}

    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic()
    system = (
        "You extract structured insurance-claim fields from a short note. "
        "Return ONLY a JSON object with any of these keys you can find: "
        + ", ".join(_EXTRACT_FIELDS)
        + ". Use numbers for amounts and units, ISO dates (YYYY-MM-DD) for dates. "
        "Do NOT invent values, and never include any personal information such as "
        "names, addresses, or dates of birth. If a field is absent, omit it."
    )
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": note}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(text[text.find("{") : text.rfind("}") + 1])
        return {k: v for k, v in data.items() if k in _EXTRACT_FIELDS}
    except Exception as exc:  # noqa: BLE001 - extraction is best-effort
        logger.warning("LLM extraction failed: %s", exc)
        return {}


def explain_claim(claim_id: str, rule_score: int, band: str, rule_reasons: list[str]) -> str:
    """Plain-language rationale for a reviewer. Never affects the score.

    Falls back to a deterministic template if the LLM is unavailable, so a
    rationale is always present.
    """
    deterministic = _template_rationale(rule_score, band, rule_reasons)
    if not llm_available():
        return deterministic

    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic()
    system = (
        "You write a short, neutral, plain-language rationale for an insurance "
        "claims reviewer. You are given a deterministic fraud score and the list of "
        "rules that fired. Explain in 2 to 4 sentences why the claim was flagged, "
        "in plain language. Do NOT change or recompute the score. Do NOT speculate "
        "beyond the rules given. Canadian spelling, no em dashes."
    )
    user = (
        f"Claim {claim_id} scored {rule_score}/100 (band: {band}). "
        f"Rules that fired: {'; '.join(rule_reasons) if rule_reasons else 'none'}."
    )
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        return text or deterministic
    except Exception as exc:  # noqa: BLE001 - narrative is best-effort
        logger.warning("LLM rationale failed: %s", exc)
        return deterministic


def _template_rationale(rule_score: int, band: str, rule_reasons: list[str]) -> str:
    if not rule_reasons:
        return f"Claim scored {rule_score}/100 ({band}). No rules fired; it looks routine."
    joined = "; ".join(rule_reasons)
    action = {
        "low": "No action needed beyond routine processing.",
        "review": "A human reviewer should confirm before any decision.",
        "high": "This should be prioritised for investigation by a human reviewer.",
    }.get(band, "A human reviewer should confirm before any decision.")
    return f"Claim scored {rule_score}/100 ({band}) because: {joined}. {action}"
