"""ClaimGuard reviewer dashboard (Streamlit).

The human-in-the-loop surface: a reviewer sees scored claims, the plain-language
reason for each score, the supplementary ML signals, and can triage (approve /
flag / dismiss). Every decision writes to the immutable audit log, and the trail
is shown with a live chain-validity check.

Run it:
    streamlit run src/claimguard/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from claimguard.api.audit import AuditLog
from claimguard.api.review_queue import ReviewQueue
from claimguard.data.synthetic import GeneratorConfig, generate_claims
from claimguard.detection.anomaly import AnomalyModel
from claimguard.detection.fairness import disparity_report
from claimguard.detection.rules import RuleEngine
from claimguard.detection.scorer import ClaimScorer, score_dataframe
from claimguard.detection.supervised import SupervisedModel
from claimguard.pipeline.features import add_features
from claimguard.pipeline.store import DEFAULT_PARQUET

SCORED_PARQUET = Path("data/scored_claims.parquet")
MODEL_DIR = Path("models")

st.set_page_config(page_title="ClaimGuard Reviewer", page_icon="🛡️", layout="wide")


@st.cache_resource
def _scorer() -> ClaimScorer:
    anomaly = AnomalyModel.load(MODEL_DIR / "anomaly.joblib") if (MODEL_DIR / "anomaly.joblib").exists() else None
    supervised = (
        SupervisedModel.load(MODEL_DIR / "supervised_gb.joblib")
        if (MODEL_DIR / "supervised_gb.joblib").exists()
        else None
    )
    return ClaimScorer(RuleEngine(), anomaly_model=anomaly, supervised_model=supervised)


@st.cache_data
def _load_scored() -> pd.DataFrame:
    """Prefer the batch-scored output; otherwise score a fresh synthetic batch."""
    for path in (SCORED_PARQUET, DEFAULT_PARQUET):
        if Path(path).exists():
            try:
                df = pd.read_parquet(path)
                if "band" in df.columns:
                    return df
            except Exception:  # noqa: BLE001
                pass
    df = generate_claims(GeneratorConfig(n_claims=2000, seed=11))
    feat = add_features(df)
    return score_dataframe(feat, _scorer())


def _gauge(score: int, band: str) -> go.Figure:
    colour = {"low": "#2e7d32", "review": "#f9a825", "high": "#c62828"}.get(band, "#555")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Fraud score ({band})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, 40], "color": "#e8f5e9"},
                    {"range": [40, 70], "color": "#fff8e1"},
                    {"range": [70, 100], "color": "#ffebee"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def main() -> None:
    st.title("🛡️ ClaimGuard reviewer")
    st.caption(
        "Auditable, explainable claim fraud triage. The score is deterministic; "
        "ML signals are advisory; every decision is logged to a tamper-evident trail."
    )

    df = _load_scored()
    audit = AuditLog()
    queue = ReviewQueue()

    # --- Top metrics ---
    band_counts = df["band"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims", len(df))
    c2.metric("Auto-pass (low)", int(band_counts.get("low", 0)))
    c3.metric("Review", int(band_counts.get("review", 0)))
    c4.metric("High risk", int(band_counts.get("high", 0)))

    tab_triage, tab_audit, tab_fairness = st.tabs(["Triage queue", "Audit trail", "Fairness"])

    # --- Triage ---
    with tab_triage:
        flagged = df[df["band"] != "low"].sort_values("rule_score", ascending=False)
        st.subheader(f"{len(flagged)} claims above auto-pass")
        if flagged.empty:
            st.info("No flagged claims in this batch.")
        else:
            left, right = st.columns([1, 1])
            with left:
                claim_id = st.selectbox("Select a flagged claim", flagged["claim_id"].tolist())
                row = flagged[flagged["claim_id"] == claim_id].iloc[0]
                st.plotly_chart(_gauge(int(row["rule_score"]), row["band"]), use_container_width=True)
                if "anomaly_score" in row and pd.notna(row.get("anomaly_score")):
                    st.metric("Anomaly score (unsupervised)", f"{row['anomaly_score']:.2f}")
                if "fraud_probability" in row and pd.notna(row.get("fraud_probability")):
                    st.metric("Fraud probability (supervised)", f"{row['fraud_probability']:.2f}")
            with right:
                st.markdown("**Why it was flagged**")
                st.info(row.get("explanation", "No explanation available."))
                st.markdown(
                    f"- Provider: `{row['provider_id']}` ({row['provider_specialty']})\n"
                    f"- Procedure: `{row['procedure_code']}`  Diagnosis: `{row.get('diagnosis_code')}`\n"
                    f"- Billed: ${row['billed_amount']:.2f}  Reference: ${row['allowed_amount']:.2f}  Units: {int(row['units'])}"
                )
                st.markdown("**Decision**")
                b1, b2, b3 = st.columns(3)
                for col, decision, label in (
                    (b1, "approved", "✅ Approve"),
                    (b2, "flagged", "🚩 Flag for investigation"),
                    (b3, "dismissed", "🗑️ Dismiss"),
                ):
                    if col.button(label, key=f"{decision}-{claim_id}"):
                        queue.add(
                            claim_id=str(claim_id),
                            rule_score=int(row["rule_score"]),
                            band=str(row["band"]),
                            recommendation=str(row.get("recommendation", "")),
                            explanation=str(row.get("explanation", "")),
                            payload={"provider_id": str(row["provider_id"])},
                        )
                        queue.decide(str(claim_id), decision, reviewer="dashboard")
                        audit.append(
                            event_type="human_review",
                            claim_id=str(claim_id),
                            payload={"decision": decision, "reviewer": "dashboard"},
                        )
                        st.success(f"Recorded '{decision}' for {claim_id} and wrote to the audit log.")

    # --- Audit ---
    with tab_audit:
        records = audit.all()
        valid = audit.verify_chain()
        st.metric("Audit records", len(records))
        st.markdown(f"**Chain integrity:** {'✅ valid (not tampered)' if valid else '❌ BROKEN'}")
        if records:
            audit_df = pd.DataFrame(
                [
                    {
                        "seq": r["seq"],
                        "ts": r["ts"],
                        "event": r["event_type"],
                        "claim_id": r["claim_id"],
                        "hash": r["record_hash"][:12] + "...",
                    }
                    for r in records[-200:]
                ]
            )
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            st.info("No audit records yet. Triage a claim or run the pipeline.")

    # --- Fairness ---
    with tab_fairness:
        st.subheader("Flag-rate disparity (four-fifths rule)")
        st.caption(
            "Demonstration on a non-sensitive grouping. The claim schema has no protected "
            "attributes by design; production would assess protected attributes under governance."
        )
        for col in ("region", "provider_specialty"):
            if col in df.columns:
                rep = disparity_report(df, group_column=col)
                st.markdown(f"**{col}** — {rep.summary()}")


if __name__ == "__main__":
    main()
