"""End-to-end AWS Lambda path against mocked AWS (S3 + DynamoDB) via moto.

This exercises the exact code that runs in deployment: an S3 ObjectCreated event
invokes the handler, which reads the claim from S3, scores it deterministically,
writes an audit item to DynamoDB, and writes the scored result back to S3. It
proves the boto3 item shapes and the event wiring are correct, with no real AWS.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

REGION = "ca-central-1"
BUCKET = "claimguard-claims-test"
TABLE = "claimguard-audit-test"
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "example_claim_fraud.json"


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.setenv("CLAIMGUARD_S3_BUCKET", BUCKET)
    monkeypatch.setenv("CLAIMGUARD_DDB_TABLE", TABLE)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)  # let moto intercept
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


@mock_aws
def test_s3_trigger_scores_audits_and_writes_processed(aws_env):
    from claimguard.aws.lambda_handler import handler

    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
    ddb = boto3.client("dynamodb", region_name=REGION)
    ddb.create_table(
        TableName=TABLE,
        AttributeDefinitions=[
            {"AttributeName": "claimId", "AttributeType": "S"},
            {"AttributeName": "auditTimestamp", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "claimId", "KeyType": "HASH"},
            {"AttributeName": "auditTimestamp", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    claim = json.loads(SAMPLE.read_text())
    s3.put_object(Bucket=BUCKET, Key="incoming/claim.json", Body=json.dumps(claim).encode())

    event = {"Records": [{"s3": {"bucket": {"name": BUCKET}, "object": {"key": "incoming/claim.json"}}}]}
    result = handler(event, None)

    # Handler returned a high-risk score
    assert result["scored"][0]["band"] == "high"

    # Audit item written to DynamoDB with the expected shape
    items = ddb.scan(TableName=TABLE)["Items"]
    assert len(items) == 1
    assert items[0]["claimId"]["S"] == "CLM-DEMO-FRAUD"
    assert items[0]["band"]["S"] == "high"
    assert int(items[0]["ruleScore"]["N"]) >= 70

    # Scored result written back to S3 processed/
    out = s3.get_object(Bucket=BUCKET, Key="processed/CLM-DEMO-FRAUD.json")
    written = json.loads(out["Body"].read())
    assert written["band"] == "high"
    assert "fee_outlier" in written["triggered_rules"]
