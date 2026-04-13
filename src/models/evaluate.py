"""
Model evaluation utilities for comparing CTR classifiers.

Produces:
  - Side-by-side AUC / log-loss / avg-precision comparison table
  - Precision-Recall curves saved to disk
  - Feature importance bar chart (XGBoost)
  - Calibration curve to assess probability quality
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/server use
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "reports"
)


def ensure_reports_dir() -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return REPORTS_DIR


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> dict:
    """Return a flat metrics dict for one model."""
    return {
        "model": model_name,
        "auc_roc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Pretty-print and return a comparison DataFrame from a list of metrics dicts.

    Args:
        results: list of dicts produced by compute_metrics()
    """
    df = pd.DataFrame(results).set_index("model")
    logger.info("\n" + "=" * 50)
    logger.info("Model Comparison")
    logger.info("=" * 50)
    logger.info(
        df.to_string(float_format=lambda x: f"{x:.4f}")
    )
    return df


def plot_precision_recall_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    filename: str = "pr_curves.png",
) -> str:
    """
    Plot precision-recall curves for each model in `models` dict.

    Args:
        models: {model_name: fitted_model}
        X_test: feature matrix
        y_test: ground truth labels
        filename: output filename in reports/

    Returns:
        Absolute path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — CTR Models")
    ax.legend()
    ax.grid(True)

    out_path = os.path.join(ensure_reports_dir(), filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curves saved to {out_path}")
    return out_path


def plot_feature_importance(
    model,
    feature_cols: list,
    top_n: int = 20,
    filename: str = "feature_importance.png",
) -> str:
    """
    Plot horizontal bar chart of XGBoost feature importances.

    Returns:
        Absolute path to saved figure.
    """
    importances = model.feature_importances_
    feat_imp = (
        pd.Series(importances, index=feature_cols)
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    feat_imp[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    out_path = os.path.join(ensure_reports_dir(), filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance chart saved to {out_path}")
    return out_path


def plot_calibration_curve(
    models: dict,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_bins: int = 10,
    filename: str = "calibration_curve.png",
) -> str:
    """
    Plot reliability (calibration) diagrams for each model.
    A well-calibrated model should lie close to the diagonal.

    Returns:
        Absolute path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — CTR Models")
    ax.legend()
    ax.grid(True)

    out_path = os.path.join(ensure_reports_dir(), filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Calibration curve saved to {out_path}")
    return out_path
