"""Fairness / disparity check.

OSFI Guideline E-23 expects model risk to include a bias evaluation. This module
demonstrates that capability: it measures whether the flag rate differs across
groups, using a four-fifths (80%) style disparity ratio.

Important honesty note for an interview: ClaimGuard's claim schema deliberately
carries NO protected attributes (no age, gender, race, or proxies for them), so
this check runs over a non-sensitive grouping such as region or provider
specialty to show the method. In production you would compute the same statistic
across the actual protected attributes, under formal model-governance sign-off,
on real (not synthetic) data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DisparityReport:
    group_column: str
    flag_rates: dict[str, float] = field(default_factory=dict)
    disparity_ratio: float = 1.0  # min rate / max rate; 1.0 = perfectly even
    passes_four_fifths: bool = True
    note: str = ""

    def summary(self) -> str:
        rates = ", ".join(f"{g}={r:.3f}" for g, r in sorted(self.flag_rates.items()))
        status = "PASS" if self.passes_four_fifths else "REVIEW"
        return (
            f"Disparity by {self.group_column} [{status}]: ratio={self.disparity_ratio:.3f} "
            f"(four-fifths rule). Flag rates: {rates}"
        )


def disparity_report(
    scored: pd.DataFrame,
    group_column: str = "region",
    flag_column: str = "band",
    flag_values: tuple[str, ...] = ("review", "high"),
    min_group_size: int = 30,
) -> DisparityReport:
    """Compute per-group flag rates and the four-fifths disparity ratio.

    Groups smaller than ``min_group_size`` are excluded from the ratio (too few
    samples to be meaningful) but still reported.
    """
    if group_column not in scored.columns or flag_column not in scored.columns:
        return DisparityReport(group_column=group_column, note="Required columns missing.")

    flagged = scored[flag_column].isin(flag_values)
    grouped = scored.assign(_flagged=flagged).groupby(group_column)
    rates = grouped["_flagged"].mean()
    sizes = grouped.size()

    flag_rates = {str(g): float(r) for g, r in rates.items()}
    eligible = rates[sizes >= min_group_size]

    if len(eligible) < 2 or eligible.max() == 0:
        return DisparityReport(
            group_column=group_column,
            flag_rates=flag_rates,
            disparity_ratio=1.0,
            passes_four_fifths=True,
            note="Too few comparable groups to assess disparity meaningfully.",
        )

    ratio = float(eligible.min() / eligible.max())
    return DisparityReport(
        group_column=group_column,
        flag_rates=flag_rates,
        disparity_ratio=ratio,
        passes_four_fifths=ratio >= 0.8,
        note="Demonstration on a non-sensitive grouping; production would use protected attributes under governance.",
    )
