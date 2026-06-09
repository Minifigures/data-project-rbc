"""Evaluation metrics tuned for rare-event (fraud) detection.

Accuracy is deliberately excluded. With a 5% fraud rate, a model that flags
nothing is 95% accurate and 0% useful. We report precision, recall, F1, the
confusion-matrix counts, and PR-AUC (average precision), which is the right
summary curve when the positive class is rare.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Precision / recall / F1 / confusion counts, plus PR-AUC if scores given."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, float] = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if y_score is not None:
        out["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
    return out


def format_metrics_table(named: dict[str, dict[str, float]]) -> str:
    """Render a dict of {model_name: metrics} as a fixed-width table."""
    cols = ["precision", "recall", "f1", "pr_auc", "tp", "fp", "fn", "tn"]
    header = f"{'model':22s}" + "".join(f"{c:>9s}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in named.items():
        row = f"{name:22s}"
        for c in cols:
            val = m.get(c, "")
            row += f"{val:>9}" if isinstance(val, int) else f"{val:>9.4f}" if isinstance(val, float) else f"{'':>9}"
        lines.append(row)
    return "\n".join(lines)
