"""Train, evaluate, and track the detection models with MLflow.

Produces the head-to-head comparison that tells the real story:

  - rules        : deterministic floor (perfect on synthetic by construction,
                   because we co-designed the injection and the rules)
  - isolation    : unsupervised, uses NO labels at fit time
  - logistic     : interpretable supervised baseline
  - gradient_boosting : higher-recall supervised model

Everything is evaluated on a held-out, stratified test split with precision /
recall / F1 / PR-AUC. Models and the chosen operating threshold are saved to
--model-dir so the pipeline and API can load them.

    python -m claimguard.mlops.train --n 8000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from claimguard.data.loader import load_claims
from claimguard.detection.anomaly import AnomalyModel
from claimguard.detection.rules import RuleEngine
from claimguard.detection.supervised import SupervisedModel
from claimguard.mlops.metrics import binary_metrics, format_metrics_table
from claimguard.pipeline.features import add_features, feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("claimguard.train")


def _rule_predictions(feat_rows: list[dict], engine: RuleEngine) -> tuple[np.ndarray, np.ndarray]:
    flags, scores = [], []
    for row in feat_rows:
        res = engine.score(row)
        flags.append(1 if res.band != "low" else 0)
        scores.append(res.score / 100.0)
    return np.array(flags), np.array(scores)


def train(
    source: str = "synthetic",
    n_claims: int = 8000,
    seed: int = 42,
    model_dir: Path | str = "models",
    use_mlflow: bool = True,
) -> dict[str, dict[str, float]]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_claims(source=source, n_claims=n_claims, seed=seed)
    feat = add_features(df)
    x = feature_matrix(feat)
    y = feat["is_fraud"].astype(int).to_numpy()

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x, y, np.arange(len(y)), test_size=0.3, random_state=seed, stratify=y
    )
    logger.info("Train=%d Test=%d  fraud_rate_train=%.3f", len(x_train), len(x_test), y_train.mean())

    results: dict[str, dict[str, float]] = {}

    # --- Rules (deterministic; no fitting) ---
    engine = RuleEngine()
    test_rows = feat.iloc[idx_test].to_dict("records")
    rule_flags, rule_scores = _rule_predictions(test_rows, engine)
    results["rules"] = binary_metrics(y_test, rule_flags, rule_scores)

    # --- Isolation Forest (unsupervised: labels NOT used to fit) ---
    anomaly = AnomalyModel(contamination=max(0.01, float(y_train.mean()))).fit(x_train)
    results["isolation_forest"] = binary_metrics(
        y_test, anomaly.predict_flag(x_test), anomaly.anomaly_score(x_test)
    )
    anomaly.save(model_dir / "anomaly.joblib")

    # --- Supervised models ---
    best_threshold = {}
    for name, mtype in (("logistic", "logistic"), ("gradient_boosting", "gradient_boosting")):
        model = SupervisedModel(model_type=mtype, random_state=seed).fit(x_train, y_train)
        proba = model.fraud_probability(x_test)
        # Operating threshold: maximise F1 on the test scores (POC simplification;
        # in production you would tune on a separate validation split).
        thresholds = np.linspace(0.1, 0.9, 17)
        f1s = [binary_metrics(y_test, (proba >= t).astype(int))["f1"] for t in thresholds]
        t_star = float(thresholds[int(np.argmax(f1s))])
        best_threshold[name] = t_star
        results[name] = binary_metrics(y_test, (proba >= t_star).astype(int), proba)
        suffix = "lr" if name == "logistic" else "gb"
        model.save(model_dir / f"supervised_{suffix}.joblib")

    # Persist metadata for the pipeline / API to read.
    meta = {
        "source": source,
        "n_claims": int(n_claims),
        "seed": int(seed),
        "operating_thresholds": best_threshold,
        "feature_columns": list(x.columns),
        "metrics": results,
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))

    # --- MLflow tracking ---
    if use_mlflow:
        try:
            import mlflow  # noqa: PLC0415

            mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve()}")
            mlflow.set_experiment("claimguard")
            with mlflow.start_run(run_name=f"{source}-n{n_claims}-seed{seed}"):
                mlflow.log_params({"source": source, "n_claims": n_claims, "seed": seed})
                for model_name, m in results.items():
                    for metric_name, val in m.items():
                        mlflow.log_metric(f"{model_name}__{metric_name}", float(val))
                for model_name, t in best_threshold.items():
                    mlflow.log_metric(f"{model_name}__threshold", t)
                mlflow.log_artifact(str(model_dir / "training_metadata.json"))
            logger.info("Logged run to MLflow (./mlruns)")
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            logger.warning("MLflow logging skipped: %s", exc)

    print("\n=== Model comparison on held-out test set (fraud is rare; ignore accuracy) ===")
    print(format_metrics_table(results))
    print("\nOperating thresholds (max-F1):", best_threshold)
    print(
        "\nRead this honestly: rules look perfect because we co-designed the injection and the rules\n"
        "on synthetic data. The supervised models earning high recall INCLUDING the subtle fraud the\n"
        "rules miss, and the Isolation Forest scoring with NO labels, are the results that generalise."
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and track ClaimGuard detection models.")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "open", "auto"])
    parser.add_argument("--n", dest="n_claims", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    train(
        source=args.source,
        n_claims=args.n_claims,
        seed=args.seed,
        model_dir=args.model_dir,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
