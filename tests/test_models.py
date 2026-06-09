"""Anomaly model, supervised model, and metrics."""

from __future__ import annotations

import numpy as np

from claimguard.detection.anomaly import AnomalyModel
from claimguard.detection.supervised import SupervisedModel
from claimguard.mlops.metrics import binary_metrics
from claimguard.pipeline.features import feature_matrix


def test_anomaly_scores_in_unit_range_and_roundtrips(featured_df, tmp_path):
    x = feature_matrix(featured_df)
    model = AnomalyModel(contamination=0.05).fit(x)
    scores = model.anomaly_score(x)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    path = tmp_path / "anomaly.joblib"
    model.save(path)
    reloaded = AnomalyModel.load(path)
    assert np.allclose(reloaded.anomaly_score(x), scores)


def test_supervised_learns_and_predicts(featured_df, tmp_path):
    x = feature_matrix(featured_df)
    y = featured_df["is_fraud"].astype(int).to_numpy()
    model = SupervisedModel(model_type="logistic").fit(x, y)
    proba = model.fraud_probability(x)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # Should be better than chance on data with strong injected signal.
    metrics = binary_metrics(y, (proba >= 0.5).astype(int), proba)
    assert metrics["recall"] > 0.5
    importances = model.explain_global()
    assert len(importances) == x.shape[1]
    path = tmp_path / "sup.joblib"
    model.save(path)
    assert np.allclose(SupervisedModel.load(path).fraud_probability(x), proba)


def test_binary_metrics_on_known_case():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])  # 1 TP, 1 FN, 2 TN, 0 FP
    m = binary_metrics(y_true, y_pred)
    assert m["tp"] == 1 and m["fn"] == 1 and m["tn"] == 2 and m["fp"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
