"""AWS Lambda handler: the S3-triggered scoring slice.

The intended cloud shape (see AWS_COWORK_HANDOFF.md):

    claim JSON -> s3://<bucket>/incoming/  --(S3 event)-->  this Lambda
        -> deterministic rule score
        -> write {claimId, score, band, rules, ts} to DynamoDB (immutable audit)
        -> write the scored result to s3://<bucket>/processed/

Only the DETERMINISTIC rule engine runs in Lambda, on purpose: it is small, fast,
needs no model artefacts, and keeps the cloud scorer auditable. The ML models run
in the batch / training path, not in the request-time Lambda.

Everything is endpoint-configurable, so the same code runs against real AWS or a
local emulator (LocalEmu / LocalStack) by setting AWS_ENDPOINT_URL=http://localhost:4566.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import pandas as pd

from claimguard.detection.rules import RuleEngine
from claimguard.pipeline.features import add_features

logger = logging.getLogger("claimguard.lambda")
logger.setLevel(logging.INFO)

# Build the rule engine once per cold start (cheap, deterministic, no model files).
_ENGINE = RuleEngine()


def _client(service: str):
    import boto3  # noqa: PLC0415

    return boto3.client(
        service,
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None,
        region_name=os.environ.get("AWS_REGION", "ca-central-1"),
    )


def score_claim_payload(payload: dict) -> dict:
    """Score a single claim dict with the deterministic rule engine."""
    row = dict(payload)
    row.setdefault("claim_id", "CLM-UNKNOWN")
    row.setdefault("claimant_id", "MBR-UNKNOWN")
    row.setdefault("provider_id", "PRV-UNKNOWN")
    row.setdefault("provider_specialty", "Unknown")
    row.setdefault("claim_type", "medical")
    row.setdefault("diagnosis_code", None)
    row.setdefault("place_of_service", "clinic")
    row.setdefault("region", "ON")
    row.setdefault("paid_amount", None)
    row.setdefault("units", 1)
    row.setdefault("is_fraud", 0)
    row.setdefault("fraud_type", None)
    feat = add_features(pd.DataFrame([row]))
    result = _ENGINE.score(feat.iloc[0].to_dict())
    return {
        "claim_id": str(row["claim_id"]),
        "rule_score": result.score,
        "band": result.band,
        "recommendation": {
            "low": "auto_pass",
            "review": "route_to_human_review",
            "high": "priority_investigation",
        }[result.band],
        "triggered_rules": [h.rule_id for h in result.hits],
        "explanation": result.explanation(),
        "policy_version": int(_ENGINE.policy.get("version", 1)),
        "scored_at": datetime.now(UTC).isoformat(),
    }


def _write_audit(scored: dict) -> None:
    table = os.environ.get("CLAIMGUARD_DDB_TABLE")
    if not table:
        logger.warning("CLAIMGUARD_DDB_TABLE unset; skipping DynamoDB write.")
        return
    ddb = _client("dynamodb")
    ddb.put_item(
        TableName=table,
        Item={
            "claimId": {"S": scored["claim_id"]},
            "auditTimestamp": {"S": scored["scored_at"]},
            "ruleScore": {"N": str(scored["rule_score"])},
            "band": {"S": scored["band"]},
            "recommendation": {"S": scored["recommendation"]},
            "triggeredRules": {"S": json.dumps(scored["triggered_rules"])},
            "policyVersion": {"N": str(scored["policy_version"])},
        },
    )


def _write_processed(scored: dict) -> None:
    bucket = os.environ.get("CLAIMGUARD_S3_BUCKET")
    if not bucket:
        return
    s3 = _client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=f"processed/{scored['claim_id']}.json",
        Body=json.dumps(scored).encode("utf-8"),
        ContentType="application/json",
    )


def handler(event: dict, context=None) -> dict:
    """Lambda entry point. Handles S3 trigger events and direct-invoke payloads."""
    results = []

    # Direct invoke for testing: {"claim": {...}}
    if "claim" in event:
        scored = score_claim_payload(event["claim"])
        _write_audit(scored)
        _write_processed(scored)
        return {"scored": [scored]}

    # S3 trigger: read each created object, score it.
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        s3 = _client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())
        scored = score_claim_payload(payload)
        _write_audit(scored)
        _write_processed(scored)
        logger.info("Scored %s -> %s (%s)", scored["claim_id"], scored["rule_score"], scored["band"])
        results.append(scored)

    return {"scored": results}
