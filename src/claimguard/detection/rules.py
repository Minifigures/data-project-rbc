"""Deterministic, auditable rule engine.

Reads the YAML policy and turns a feature row into a 0-100 score with a per-rule
explanation. There is no randomness and no model here: the same claim always
produces the same score and the same reasons, which is exactly what an insurer
needs to defend a decision to a customer, an auditor, or a regulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY_PATH = Path(__file__).with_name("policy.yaml")


@dataclass
class RuleHit:
    rule_id: str
    typology: str
    points: int
    reason: str


@dataclass
class RuleScore:
    score: int
    band: str
    hits: list[RuleHit] = field(default_factory=list)

    def explanation(self) -> str:
        """One-line, human-readable summary of why this score was assigned."""
        if not self.hits:
            return "No rules triggered; claim looks routine."
        parts = [f"{h.reason} (+{h.points})" for h in self.hits]
        return f"Score {self.score}/100 [{self.band}]: " + "; ".join(parts)


def _compare(value: Any, op: str, target: Any = None) -> bool:
    if value is None:
        return False
    if op == "is_true":
        return bool(value)
    if op == "is_false":
        return not bool(value)
    try:
        v = float(value)
        t = float(target)
    except (TypeError, ValueError):
        return False
    if op == ">=":
        return v >= t
    if op == ">":
        return v > t
    if op == "<=":
        return v <= t
    if op == "<":
        return v < t
    if op == "==":
        return v == t
    raise ValueError(f"Unsupported operator: {op}")


def _eval_condition(cond: dict, row: dict) -> bool:
    if "all" in cond:
        return all(_eval_condition(sub, row) for sub in cond["all"])
    if "any" in cond:
        return any(_eval_condition(sub, row) for sub in cond["any"])
    feature = cond["feature"]
    return _compare(row.get(feature), cond["op"], cond.get("value"))


class RuleEngine:
    """Loads a policy file once and scores feature rows against it."""

    def __init__(self, policy_path: Path | str = DEFAULT_POLICY_PATH) -> None:
        self.policy_path = Path(policy_path)
        with open(self.policy_path) as fh:
            self.policy = yaml.safe_load(fh)
        self.score_cap = int(self.policy.get("score_cap", 100))
        self.bands = self.policy.get("bands", {"low": 0, "review": 40, "high": 70})
        self.rules = self.policy.get("rules", [])

    def band_for(self, score: int) -> str:
        if score >= self.bands.get("high", 70):
            return "high"
        if score >= self.bands.get("review", 40):
            return "review"
        return "low"

    def score(self, row: dict) -> RuleScore:
        """Score a single feature row (a dict of feature name -> value)."""
        hits: list[RuleHit] = []
        for rule in self.rules:
            if _eval_condition(rule["condition"], row):
                hits.append(
                    RuleHit(
                        rule_id=rule["id"],
                        typology=rule.get("typology", "unspecified"),
                        points=int(rule["weight"]),
                        reason=rule["description"],
                    )
                )
        total = min(self.score_cap, sum(h.points for h in hits))
        return RuleScore(score=total, band=self.band_for(total), hits=hits)
