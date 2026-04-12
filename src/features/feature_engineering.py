"""
Feature engineering for the CTR training pipeline.
Handles encoding, scaling, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

# Categorical columns to be label-encoded
CATEGORICAL_COLS = [
    "publisher_id",
    "exchange_id",
    "device_type",
    "os",
    "country",
    "campaign_id",
    "ad_id",
]

# Numeric features to pass through directly
NUMERIC_COLS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "floor_price",
    "bid_price",
]

TARGET_COL = "actual_click"


def encode_categoricals(df: pd.DataFrame, encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical columns. If encoders dict is not provided,
    fit new encoders. Otherwise, apply existing encoders (for inference/test sets).

    Returns:
        df_encoded: DataFrame with encoded columns replacing originals
        encoders: dict of {col_name: fitted LabelEncoder}
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
        fit_mode = True
    else:
        fit_mode = False

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in dataframe — skipping encoding.")
            continue
        le = encoders.get(col, LabelEncoder())
        if fit_mode:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            # Handle unseen labels gracefully by mapping to -1
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known_classes else -1
            )

    return df, encoders


def build_feature_matrix(df: pd.DataFrame, encoders: dict = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Full feature engineering pipeline that produces a feature matrix X and labels y.

    Returns:
        X: DataFrame of model-ready features
        y: numpy array of binary click labels
        encoders: dict of label encoders (for serialization)
    """
    required_cols = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_enc, encoders = encode_categoricals(df, encoders)

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df_enc[feature_cols]
    y = df_enc[TARGET_COL].values

    logger.info(f"Feature matrix built: shape={X.shape}, positive_rate={y.mean():.4f}")
    return X, y, encoders
