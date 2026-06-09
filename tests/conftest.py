"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from claimguard.data.synthetic import GeneratorConfig, generate_claims
from claimguard.detection.scorer import ClaimScorer, score_dataframe
from claimguard.pipeline.features import add_features


@pytest.fixture(scope="session")
def claims_df():
    return generate_claims(GeneratorConfig(n_claims=1500, seed=123))


@pytest.fixture(scope="session")
def featured_df(claims_df):
    return add_features(claims_df)


@pytest.fixture(scope="session")
def scored_df(featured_df):
    return score_dataframe(featured_df, ClaimScorer())
