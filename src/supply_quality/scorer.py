"""
Supply Quality Scorer — Phase 6.

Computes an interpretable composite quality score (0–100) for each
publisher and exchange in the ad auction dataset.

Scoring methodology
-------------------
For each entity (publisher or exchange) we compute five business metrics:

  Metric              Direction   Weight
  ─────────────────── ─────────── ──────
  ctr                 high=good   0.25
  cvr                 high=good   0.20
  fraud_rate          low=good    0.25   (inverted before scoring)
  avg_roi             high=good   0.20
  revenue_per_imp     high=good   0.10

Each metric is min-max normalised to [0, 1] across all entities, then
the weighted sum (×100) yields the final quality_score.

Usage
-----
    from src.supply_quality.scorer import run_scoring
    pub_scores, exch_scores = run_scoring(df)
"""

import pandas as pd
import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# -----------------------------------------------------------------
METRIC_CONFIG = {
    "ctr":              {"direction": "high", "weight": 0.25},
    "cvr":              {"direction": "high", "weight": 0.20},
    "fraud_rate":       {"direction": "low",  "weight": 0.25},
    "avg_roi":          {"direction": "high", "weight": 0.20},
    "revenue_per_imp":  {"direction": "high", "weight": 0.10},
}

assert abs(sum(v["weight"] for v in METRIC_CONFIG.values()) - 1.0) < 1e-9, \
    "Scoring weights must sum to 1.0"

MIN_IMPRESSIONS = 50   # entities with fewer impressions get flagged as low-volume


def _compute_entity_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate raw auction data into per-entity business metrics.

    Args:
        df:         Processed ad auction DataFrame (fraud NOT filtered — we need fraud_rate).
        group_col:  'publisher_id' or 'exchange_id'

    Returns:
        DataFrame indexed by entity with columns:
        impressions, clicks, conversions, fraud_count,
        total_cost, total_revenue,
        ctr, cvr, fraud_rate, avg_roi, revenue_per_imp
    """
    agg = (
        df.groupby(group_col)
        .agg(
            impressions      = ("request_id",  "count"),
            clicks           = ("actual_click","sum"),
            conversions      = ("conversion",  "sum"),
            fraud_count      = ("fraud_flag",  "sum"),
            total_cost       = ("cost",        "sum"),
            total_revenue    = ("revenue",     "sum"),
        )
        .reset_index()
        .rename(columns={group_col: "entity_id"})
    )

    agg["ctr"]             = agg["clicks"]       / agg["impressions"].clip(lower=1)
    agg["cvr"]             = agg["conversions"]   / agg["clicks"].clip(lower=1)
    agg["fraud_rate"]      = agg["fraud_count"]   / agg["impressions"].clip(lower=1)
    agg["avg_roi"]         = (agg["total_revenue"] - agg["total_cost"]) \
                             / (agg["total_cost"] + 1e-6)
    agg["revenue_per_imp"] = agg["total_revenue"] / agg["impressions"].clip(lower=1)

    agg["low_volume_flag"] = agg["impressions"] < MIN_IMPRESSIONS
    agg["entity_type"]     = group_col

    logger.info(
        f"[{group_col}] aggregated {len(agg)} entities "
        f"({agg['low_volume_flag'].sum()} low-volume)"
    )
    return agg


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1]; returns 0.5 if all values are identical."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _compute_quality_scores(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalisation + weighted sum to produce a quality_score (0–100).

    For 'low' direction metrics (fraud_rate), we invert after normalising
    so that higher normalised value = lower fraud = better score.
    """
    df = metrics_df.copy()
    weighted_sum = pd.Series(0.0, index=df.index)

    for metric, cfg in METRIC_CONFIG.items():
        norm = _minmax_normalize(df[metric])
        if cfg["direction"] == "low":
            norm = 1.0 - norm          # invert: low raw value → high normalised score
        weighted_sum += norm * cfg["weight"]

    df["quality_score"] = (weighted_sum * 100).round(2)
    df["quality_tier"] = pd.cut(
        df["quality_score"],
        bins   = [0, 33, 66, 100],
        labels = ["Low", "Medium", "High"],
        include_lowest=True,
    )
    return df.sort_values("quality_score", ascending=False).reset_index(drop=True)


def score_entities(
    df: pd.DataFrame,
    group_col: Literal["publisher_id", "exchange_id"],
) -> pd.DataFrame:
    """
    Public entry point: compute quality scores for one entity dimension.

    Args:
        df:         Raw or full ad auction DataFrame (must include fraud_flag).
        group_col:  Which dimension to score.

    Returns:
        Scored DataFrame sorted by quality_score descending.
    """
    metrics = _compute_entity_metrics(df, group_col)
    scored  = _compute_quality_scores(metrics)

    logger.info(
        f"[{group_col}] scoring complete — "
        f"score range [{scored['quality_score'].min():.1f}, "
        f"{scored['quality_score'].max():.1f}]"
    )
    return scored


def run_scoring(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score both publishers and exchanges.

    Returns:
        (publisher_scores, exchange_scores) — each sorted by quality_score desc.
    """
    logger.info("Starting supply quality scoring...")
    pub_scores  = score_entities(df, "publisher_id")
    exch_scores = score_entities(df, "exchange_id")
    logger.info("Supply quality scoring complete.")
    return pub_scores, exch_scores
