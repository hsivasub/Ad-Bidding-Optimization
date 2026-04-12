"""
Inference module: loads trained CTR model artifacts and generates predictions.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.features.feature_engineering import build_feature_matrix

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_DIR = os.path.join(settings.BASE_DIR, "models", "artifacts")


def load_artifact(filename: str):
    path = os.path.join(MODEL_ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}. Run train_ctr.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_ctr(df: pd.DataFrame, model_name: str = "xgb") -> np.ndarray:
    """
    Generate CTR predictions for a batch of impression requests.

    Args:
        df: DataFrame containing raw impression features (un-encoded).
        model_name: 'xgb' (default) or 'lr' for the logistic regression baseline.

    Returns:
        Array of predicted CTR probabilities, shape (n_samples,).
    """
    encoders = load_artifact("feature_encoders.pkl")
    feature_cols = load_artifact("feature_cols.pkl")

    model_file = "xgb_ctr_model.pkl" if model_name == "xgb" else "lr_ctr_model.pkl"
    model = load_artifact(model_file)

    # Build feature matrix — pass existing encoders to avoid refitting
    # Drop target col if present (inference mode)
    if "actual_click" not in df.columns:
        df = df.copy()
        df["actual_click"] = 0  # placeholder, not used in features

    X, _, _ = build_feature_matrix(df, encoders=encoders)

    # Align columns to what model was trained on
    X = X[feature_cols]

    probs = model.predict_proba(X)[:, 1]
    logger.info(f"Generated {len(probs)} CTR predictions (mean={probs.mean():.4f})")
    return probs
