"""End-to-end integration: the pipeline runner and the training orchestrator."""

from __future__ import annotations

from claimguard.mlops.train import train
from claimguard.pipeline.run_pipeline import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    scored = run_pipeline(
        source="synthetic",
        n_claims=300,
        seed=5,
        out_dir=tmp_path,
        model_dir=tmp_path / "no_models",  # forces rules-only, no model dependency
        audit_db=tmp_path / "audit.sqlite",
    )
    assert len(scored) == 300
    assert {"band", "rule_score", "recommendation", "explanation"}.issubset(scored.columns)
    assert (tmp_path / "scored_claims.parquet").exists()
    assert (tmp_path / "claimguard.sqlite").exists()


def test_train_produces_models_and_metrics(tmp_path):
    results = train(source="synthetic", n_claims=800, seed=5, model_dir=tmp_path, use_mlflow=False)
    assert {"rules", "isolation_forest", "logistic", "gradient_boosting"}.issubset(results)
    assert (tmp_path / "anomaly.joblib").exists()
    assert (tmp_path / "supervised_gb.joblib").exists()
    assert (tmp_path / "training_metadata.json").exists()
    for model_metrics in results.values():
        assert 0.0 <= model_metrics["recall"] <= 1.0
        assert 0.0 <= model_metrics["precision"] <= 1.0
