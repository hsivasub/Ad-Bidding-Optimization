"""
Supply Quality Report Generator — Phase 6.

Saves scored DataFrames to CSV and produces charts:
  - Horizontal bar chart: publisher quality scores (ranked)
  - Horizontal bar chart: exchange quality scores
  - Scatter plot: CTR vs Fraud Rate (coloured by quality tier)
  - Metrics heatmap: normalised metrics across publishers
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.config.settings import settings

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(settings.BASE_DIR, "reports")
SUPPLY_QUALITY_DIR = os.path.join(REPORTS_DIR, "supply_quality")


def _ensure_dir() -> str:
    os.makedirs(SUPPLY_QUALITY_DIR, exist_ok=True)
    return SUPPLY_QUALITY_DIR


def save_scores_csv(pub_scores: pd.DataFrame, exch_scores: pd.DataFrame) -> dict:
    """Write scored DataFrames to CSV for downstream use (dashboard, DAGs)."""
    _ensure_dir()
    pub_path  = os.path.join(SUPPLY_QUALITY_DIR, "publisher_scores.csv")
    exch_path = os.path.join(SUPPLY_QUALITY_DIR, "exchange_scores.csv")

    pub_scores.to_csv(pub_path, index=False)
    exch_scores.to_csv(exch_path, index=False)

    logger.info(f"Publisher scores saved  → {pub_path}")
    logger.info(f"Exchange scores saved   → {exch_path}")
    return {"publisher": pub_path, "exchange": exch_path}


def plot_publisher_ranking(pub_scores: pd.DataFrame, top_n: int = 20) -> str:
    """Ranked horizontal bar chart coloured by quality tier."""
    _ensure_dir()
    df = pub_scores.head(top_n).copy()

    tier_colors = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
    colors = df["quality_tier"].astype(object).map(tier_colors).fillna("#95a5a6")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df["entity_id"][::-1], df["quality_score"][::-1], color=colors[::-1])

    ax.set_xlabel("Quality Score (0–100)", fontsize=12)
    ax.set_title(f"Publisher Quality Ranking (Top {top_n})", fontsize=14, fontweight="bold")
    ax.axvline(x=66, color="green",  linestyle="--", alpha=0.5, label="High threshold (66)")
    ax.axvline(x=33, color="orange", linestyle="--", alpha=0.5, label="Low threshold (33)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(SUPPLY_QUALITY_DIR, "publisher_ranking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Publisher ranking chart → {out}")
    return out


def plot_exchange_ranking(exch_scores: pd.DataFrame) -> str:
    """Bar chart for exchange-level quality scores."""
    _ensure_dir()
    tier_colors = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}
    colors = exch_scores["quality_tier"].astype(object).map(tier_colors).fillna("#95a5a6")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(exch_scores["entity_id"][::-1], exch_scores["quality_score"][::-1],
            color=colors[::-1])
    ax.set_xlabel("Quality Score (0–100)", fontsize=12)
    ax.set_title("Exchange Quality Scores", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(SUPPLY_QUALITY_DIR, "exchange_ranking.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Exchange ranking chart  → {out}")
    return out


def plot_ctr_vs_fraud(pub_scores: pd.DataFrame) -> str:
    """
    Scatter: CTR (x) vs Fraud Rate (y), sized by impressions,
    coloured by quality tier.  Low-quality publishers cluster top-left.
    """
    _ensure_dir()
    tier_num = pub_scores["quality_tier"].map({"High": 2, "Medium": 1, "Low": 0})
    cmap = plt.get_cmap("RdYlGn")
    sizes = (pub_scores["impressions"] / pub_scores["impressions"].max() * 400).clip(lower=30)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        pub_scores["ctr"],
        pub_scores["fraud_rate"],
        c=tier_num,
        s=sizes,
        cmap=cmap,
        alpha=0.75,
        edgecolors="grey",
        linewidths=0.4,
    )
    # Annotate top/bottom publishers
    for _, row in pub_scores.head(3).iterrows():
        ax.annotate(row["entity_id"], (row["ctr"], row["fraud_rate"]),
                    fontsize=8, ha="left", va="bottom")
    for _, row in pub_scores.tail(3).iterrows():
        ax.annotate(row["entity_id"], (row["ctr"], row["fraud_rate"]),
                    fontsize=8, ha="left", va="top", color="red")

    cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Low", "Medium", "High"])
    cbar.set_label("Quality Tier")

    ax.set_xlabel("CTR", fontsize=12)
    ax.set_ylabel("Fraud Rate", fontsize=12)
    ax.set_title("Publisher CTR vs Fraud Rate (size ∝ impressions)", fontsize=13, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(SUPPLY_QUALITY_DIR, "ctr_vs_fraud.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CTR vs Fraud scatter    → {out}")
    return out


def plot_metrics_heatmap(pub_scores: pd.DataFrame, top_n: int = 15) -> str:
    """
    Heatmap of normalised raw metrics across top publishers.
    Helps identify which specific metric drives a publisher's score up/down.
    """
    _ensure_dir()
    metrics = ["ctr", "cvr", "fraud_rate", "avg_roi", "revenue_per_imp"]
    df = pub_scores.head(top_n).set_index("entity_id")[metrics].copy()

    # Min-max normalise each column for display (independent of scorer normalisations)
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-9)
    # Fraud rate: invert for visual consistency (green = good = low fraud)
    df_norm["fraud_rate"] = 1.0 - df_norm["fraud_rate"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(df_norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(
        ["CTR", "CVR", "Fraud\n(inverted)", "Avg ROI", "Rev/Imp"],
        fontsize=10,
    )
    ax.set_yticks(range(len(df_norm)))
    ax.set_yticklabels(df_norm.index, fontsize=9)
    ax.set_title(f"Supply Quality Metric Heatmap — Top {top_n} Publishers",
                 fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalised value (green = better)")
    plt.tight_layout()

    out = os.path.join(SUPPLY_QUALITY_DIR, "metrics_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metrics heatmap         → {out}")
    return out


def generate_full_report(pub_scores: pd.DataFrame, exch_scores: pd.DataFrame) -> dict:
    """Run all report outputs and return paths dict."""
    paths = save_scores_csv(pub_scores, exch_scores)
    paths["publisher_ranking"] = plot_publisher_ranking(pub_scores)
    paths["exchange_ranking"]  = plot_exchange_ranking(exch_scores)
    paths["ctr_vs_fraud"]      = plot_ctr_vs_fraud(pub_scores)
    paths["metrics_heatmap"]   = plot_metrics_heatmap(pub_scores)
    logger.info("Full supply quality report generated.")
    return paths
