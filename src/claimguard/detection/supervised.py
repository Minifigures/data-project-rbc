"""Supervised fraud classifier (Logistic Regression or Gradient Boosting).

This is the answer to "what do you do when you DO have labelled data?" (for
example the open-source dataset). Two model choices on purpose:

  - logistic  : linear, interpretable. You can read each feature's coefficient
                and explain its direction. Insurers value this.
  - gradient_boosting : non-linear, usually higher recall, less directly
                explainable. The interpretability vs performance trade-off is a
                deliberate talking point.

Class imbalance is handled with class weighting, and we evaluate with
precision / recall / F1 / PR-AUC, never accuracy, because fraud is rare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelType = Literal["logistic", "gradient_boosting"]


class SupervisedModel:
    def __init__(self, model_type: ModelType = "gradient_boosting", random_state: int = 42) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.feature_names: list[str] = []
        if model_type == "logistic":
            # Scale first so coefficients are comparable; balance the rare class.
            self.pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            )
        elif model_type == "gradient_boosting":
            self.pipeline = Pipeline(
                [("clf", GradientBoostingClassifier(random_state=random_state))]
            )
        else:  # pragma: no cover - guarded by typing
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, x: pd.DataFrame, y: pd.Series | np.ndarray) -> SupervisedModel:
        self.feature_names = list(x.columns)
        sample_weight = None
        if self.model_type == "gradient_boosting":
            # GradientBoosting has no class_weight, so weight the rare class by hand.
            y_arr = np.asarray(y)
            pos = max(int(y_arr.sum()), 1)
            neg = max(len(y_arr) - pos, 1)
            w_pos = neg / pos
            sample_weight = np.where(y_arr == 1, w_pos, 1.0)
        self.pipeline.fit(x, y, clf__sample_weight=sample_weight)
        return self

    def fraud_probability(self, x: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(x)[:, 1]

    def explain_global(self) -> dict[str, float]:
        """Feature importances (GB) or coefficients (logistic) for transparency."""
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            values = clf.coef_[0]
        else:
            values = clf.feature_importances_
        return {name: float(v) for name, v in zip(self.feature_names, values)}

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "model_type": self.model_type, "feature_names": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> SupervisedModel:
        blob = joblib.load(path)
        obj = cls(model_type=blob["model_type"])
        obj.pipeline = blob["pipeline"]
        obj.feature_names = blob["feature_names"]
        return obj
