"""
Smoke tests for the CTR training pipeline and inference module.
"""

import os
import numpy as np
import pytest
import pandas as pd

from src.data.generate_synthetic_data import generate_ad_data
from src.features.feature_engineering import build_feature_matrix, CATEGORICAL_COLS, NUMERIC_COLS


def test_feature_matrix_shape():
    """Verify feature matrix is built with correct dimensionality."""
    df = generate_ad_data(num_samples=200)
    # preprocess.py adds is_weekend and roi; simulate that here
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['roi'] = 0.0

    X, y, encoders = build_feature_matrix(df)

    expected_cols = len(CATEGORICAL_COLS) + len(NUMERIC_COLS)
    assert X.shape[1] == expected_cols, f"Expected {expected_cols} features, got {X.shape[1]}"
    assert len(y) == 200
    assert set(np.unique(y)).issubset({0, 1})


def test_encoder_reuse():
    """Ensure encoders from training can be reused for unseen test batches."""
    df_train = generate_ad_data(num_samples=300)
    df_train['is_weekend'] = df_train['day_of_week'].isin([5, 6]).astype(int)
    df_train['roi'] = 0.0

    _, _, encoders = build_feature_matrix(df_train)

    df_test = generate_ad_data(num_samples=50)
    df_test['is_weekend'] = df_test['day_of_week'].isin([5, 6]).astype(int)
    df_test['roi'] = 0.0

    # Apply saved encoders — should not raise on known categories
    X_test, y_test, _ = build_feature_matrix(df_test, encoders=encoders)
    assert X_test.shape[0] == 50


def test_artifacts_exist_after_training():
    """Verify model artifacts are saved to disk after training pipeline runs."""
    artifact_dir = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
    for fname in ["xgb_ctr_model.pkl", "lr_ctr_model.pkl", "feature_encoders.pkl", "feature_cols.pkl"]:
        fpath = os.path.join(artifact_dir, fname)
        assert os.path.exists(fpath), (
            f"Missing artifact: {fname}. "
            "Run `python -m src.models.train_ctr` before running tests."
        )
