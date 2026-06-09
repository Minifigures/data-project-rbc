"""End-to-end batch pipeline (the automation piece).

One command takes raw claims all the way to scored, stored, audited output:

    ingest -> validate -> feature-engineer -> score -> store -> audit

Run it:
    python -m claimguard.pipeline.run_pipeline --source synthetic --n 5000 --seed 42

It loads trained ML models from --model-dir if they exist (so the ML signals
appear), and runs rules-only if they do not. Validation is a hard gate: a failing
report stops the run before anything is scored.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from claimguard.api.audit import AuditLog
from claimguard.data.loader import load_claims
from claimguard.detection.anomaly import AnomalyModel
from claimguard.detection.rules import RuleEngine
from claimguard.detection.scorer import ClaimScorer, score_dataframe
from claimguard.detection.supervised import SupervisedModel
from claimguard.pipeline.features import add_features
from claimguard.pipeline.store import write_parquet, write_sqlite
from claimguard.pipeline.validate import validate_claims

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("claimguard.pipeline")


def _load_models(model_dir: Path) -> tuple[AnomalyModel | None, SupervisedModel | None]:
    anomaly = None
    supervised = None
    a_path = model_dir / "anomaly.joblib"
    s_path = model_dir / "supervised_gb.joblib"
    if a_path.exists():
        anomaly = AnomalyModel.load(a_path)
        logger.info("Loaded anomaly model from %s", a_path)
    if s_path.exists():
        supervised = SupervisedModel.load(s_path)
        logger.info("Loaded supervised model from %s", s_path)
    return anomaly, supervised


def run_pipeline(
    source: str = "synthetic",
    n_claims: int = 5000,
    seed: int = 42,
    out_dir: Path | str = "data",
    model_dir: Path | str = "models",
    audit_db: Path | str | None = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    model_dir = Path(model_dir)

    # 1. Ingest
    df = load_claims(source=source, n_claims=n_claims, seed=seed)
    logger.info("Ingested %d claims from source=%s", len(df), source)

    # 2. Validate (hard gate)
    report = validate_claims(df)
    logger.info("\n%s", report.summary())
    if not report.ok:
        raise ValueError(f"Validation failed; aborting pipeline. Errors: {report.errors}")

    # 3. Feature-engineer
    feat = add_features(df)

    # 4. Score (rules always; ML if models are present)
    anomaly, supervised = _load_models(model_dir)
    scorer = ClaimScorer(RuleEngine(), anomaly_model=anomaly, supervised_model=supervised)
    scored = score_dataframe(feat, scorer)

    # 5. Store
    parquet_path = write_parquet(scored, out_dir / "scored_claims.parquet")
    sqlite_path = write_sqlite(scored, out_dir / "claimguard.sqlite", table="scored_claims")
    logger.info("Stored results -> %s and %s", parquet_path, sqlite_path)

    # 6. Audit (one immutable summary record for the run)
    band_counts = scored["band"].value_counts().to_dict()
    audit = AuditLog(audit_db) if audit_db else AuditLog()
    rec = audit.append(
        event_type="pipeline_run",
        claim_id="BATCH",
        payload={
            "source": source,
            "n_claims": int(len(scored)),
            "band_counts": {k: int(v) for k, v in band_counts.items()},
            "policy_version": int(scorer.rules.policy.get("version", 1)),
            "ml_models": {"anomaly": anomaly is not None, "supervised": supervised is not None},
        },
    )
    logger.info("Audit record #%d written (chain hash %s...)", rec["seq"], rec["record_hash"][:12])

    # Console summary
    flagged = int((scored["band"] != "low").sum())
    print("\n=== ClaimGuard pipeline summary ===")
    print(f"source={source}  claims={len(scored)}  flagged(review+high)={flagged}")
    print("band distribution:", {k: int(v) for k, v in band_counts.items()})
    if "is_fraud" in scored.columns and scored["is_fraud"].sum() > 0:
        tp = int(((scored["band"] != "low") & (scored["is_fraud"] == 1)).sum())
        fp = int(((scored["band"] != "low") & (scored["is_fraud"] == 0)).sum())
        fn = int(((scored["band"] == "low") & (scored["is_fraud"] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        print(f"rule-layer vs labels: precision={prec:.3f} recall={rec_:.3f} (tp={tp} fp={fp} fn={fn})")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClaimGuard scoring pipeline end to end.")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "open", "auto"])
    parser.add_argument("--n", dest="n_claims", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    run_pipeline(
        source=args.source,
        n_claims=args.n_claims,
        seed=args.seed,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
