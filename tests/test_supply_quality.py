"""
Tests for the supply quality scoring module (Phase 6).
"""

import pytest
import pandas as pd
import numpy as np

from src.data.generate_synthetic_data import generate_ad_data
from src.supply_quality.scorer import (
    run_scoring,
    score_entities,
    _compute_entity_metrics,
    _compute_quality_scores,
    METRIC_CONFIG,
)


@pytest.fixture
def raw_df():
    """Small raw auction dataset including fraud_flag."""
    return generate_ad_data(num_samples=500)


def test_metric_weights_sum_to_one():
    """Scoring weights must sum to exactly 1.0."""
    total = sum(v["weight"] for v in METRIC_CONFIG.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_entity_metrics_shape(raw_df):
    """Aggregation produces one row per publisher and correct columns."""
    metrics = _compute_entity_metrics(raw_df, "publisher_id")
    expected_cols = {
        "entity_id", "impressions", "clicks", "conversions",
        "fraud_count", "total_cost", "total_revenue",
        "ctr", "cvr", "fraud_rate", "avg_roi", "revenue_per_imp",
        "low_volume_flag", "entity_type",
    }
    assert expected_cols.issubset(set(metrics.columns))
    # Number of rows = unique publishers in the dataset
    assert len(metrics) == raw_df["publisher_id"].nunique()


def test_ctr_in_valid_range(raw_df):
    """CTR must be in [0, 1] for all entities."""
    metrics = _compute_entity_metrics(raw_df, "publisher_id")
    assert metrics["ctr"].between(0, 1).all()


def test_fraud_rate_in_valid_range(raw_df):
    """Fraud rate must be in [0, 1]."""
    metrics = _compute_entity_metrics(raw_df, "publisher_id")
    assert metrics["fraud_rate"].between(0, 1).all()


def test_quality_score_range(raw_df):
    """quality_score must be in [0, 100]."""
    pub_scores, _ = run_scoring(raw_df)
    assert pub_scores["quality_score"].between(0, 100).all()


def test_quality_tier_values(raw_df):
    """All quality tiers must be one of the three defined tiers."""
    pub_scores, exch_scores = run_scoring(raw_df)
    for df in (pub_scores, exch_scores):
        assert set(df["quality_tier"].unique()).issubset({"Low", "Medium", "High"})


def test_scores_sorted_descending(raw_df):
    """Results must be sorted by quality_score descending."""
    pub_scores, _ = run_scoring(raw_df)
    scores = pub_scores["quality_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_high_fraud_publisher_scores_low(raw_df):
    """
    pub_20 has 5x fraud multiplier in the generator — it should rank in
    the bottom half of publishers.
    """
    pub_scores, _ = run_scoring(raw_df)
    if "pub_20" not in pub_scores["entity_id"].values:
        pytest.skip("pub_20 not sampled in this run")

    pub20_rank = pub_scores.index[pub_scores["entity_id"] == "pub_20"][0]
    total = len(pub_scores)
    assert pub20_rank > total // 2, (
        f"pub_20 (high fraud) ranked {pub20_rank}/{total}, expected bottom half"
    )


def test_exchange_scoring(raw_df):
    """Exchange scoring produces one row per exchange."""
    _, exch_scores = run_scoring(raw_df)
    assert len(exch_scores) == raw_df["exchange_id"].nunique()
    assert "quality_score" in exch_scores.columns
