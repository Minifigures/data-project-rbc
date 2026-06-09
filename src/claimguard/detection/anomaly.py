"""Unsupervised anomaly detector (Isolation Forest).

This is the answer to "what do you do when you have NO labelled fraud?". The
model learns the shape of normal claims and flags the ones that sit far from the
crowd, with no fraud labels required at training time. It will not know WHY a
claim is odd, which is exactly why it sits alongside the explainable rule engine
rather than replacing it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        # contamination is our prior on the fraud / anomaly rate. It only sets
        # the binary flag threshold; the continuous score is independent of it.
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._raw_min: float = 0.0
        self._raw_max: float = 1.0
        self.fitted: bool = False

    def fit(self, x: pd.DataFrame) -> AnomalyModel:
        self.model.fit(x)
        raw = -self.model.score_samples(x)  # higher = more anomalous
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())
        self.fitted = True
        return self

    def anomaly_score(self, x: pd.DataFrame) -> np.ndarray:
        """Continuous anomaly score in [0, 1] (1 = most anomalous).

        Normalised against the training distribution so scores are comparable
        across runs.
        """
        raw = -self.model.score_samples(x)
        span = self._raw_max - self._raw_min
        if span <= 0:
            return np.zeros(len(raw))
        return np.clip((raw - self._raw_min) / span, 0.0, 1.0)

    def predict_flag(self, x: pd.DataFrame) -> np.ndarray:
        """Binary anomaly flag (1 = anomalous) using the contamination threshold."""
        return (self.model.predict(x) == -1).astype(int)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "raw_min": self._raw_min, "raw_max": self._raw_max}, path)

    @classmethod
    def load(cls, path: Path | str) -> AnomalyModel:
        blob = joblib.load(path)
        obj = cls()
        obj.model = blob["model"]
        obj._raw_min = blob["raw_min"]
        obj._raw_max = blob["raw_max"]
        obj.fitted = True
        return obj
