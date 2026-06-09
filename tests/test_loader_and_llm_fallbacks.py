"""The graceful-degradation paths: no Kaggle token, no LLM key."""

from __future__ import annotations

import pytest

import claimguard.data.loader as loader
from claimguard.detection import llm_perception


def test_load_synthetic_source():
    df = loader.load_claims("synthetic", n_claims=200, seed=1)
    assert len(df) == 200


def test_auto_falls_back_to_synthetic_when_open_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no Kaggle token")

    monkeypatch.setattr(loader, "load_open_dataset", _boom)
    df = loader.load_claims("auto", n_claims=150, seed=2)
    assert len(df) == 150  # fell back instead of raising


def test_open_source_raises_when_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no Kaggle token")

    monkeypatch.setattr(loader, "load_open_dataset", _boom)
    with pytest.raises(RuntimeError):
        loader.load_claims("open")


def test_llm_unavailable_uses_deterministic_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_perception.llm_available() is False
    text = llm_perception.explain_claim("CLM-1", 80, "high", ["fee outlier"])
    assert "80/100" in text and "high" in text
    assert llm_perception.extract_claim_fields("a claim note") == {}
