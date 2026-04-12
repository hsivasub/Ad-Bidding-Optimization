"""
Baseline CTR Model Training Pipeline.

Trains a Logistic Regression baseline and an XGBoost model for click-through rate
prediction. Evaluates on a held-out test set and saves model artifacts to disk.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
import xgboost as xgb

from src.config.settings import settings
from src.features.feature_engineering import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model artifact output directory
MODEL_OUTPUT_DIR = os.path.join(settings.BASE_DIR, "models", "artifacts")

RANDOM_STATE = 42


def load_processed_data() -> pd.DataFrame:
    path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Please run src/data/generate_synthetic_data.py and src/data/preprocess.py first."
        )
    logger.info(f"Loading processed dataset from {path}")
    return pd.read_csv(path)


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray, model_name: str) -> dict:
    """
    Compute AUC, log_loss, and average precision for a fitted model.
    Returns a metrics dict suitable for logging or MLflow tracking.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    logger.info(f"\n{'='*40}")
    logger.info(f"Model: {model_name}")
    logger.info(f"  AUC-ROC:          {auc:.4f}")
    logger.info(f"  Log Loss:         {ll:.4f}")
    logger.info(f"  Avg Precision:    {ap:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return {"model": model_name, "auc": auc, "log_loss": ll, "avg_precision": ap}


def save_artifact(obj, filename: str) -> str:
    """Persist a Python object to disk using pickle."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(MODEL_OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved artifact: {path}")
    return path


def train():
    """
    End-to-end training pipeline:
    1. Load processed data
    2. Feature engineering
    3. Train/val/test split (60/20/20)
    4. Train Logistic Regression baseline
    5. Train XGBoost model
    6. Evaluate both on test set
    7. Save models and encoders to disk
    """
    df = load_processed_data()
    X, y, encoders = build_feature_matrix(df)

    # 60/20/20 split — stratify on target to preserve class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # -----------------------------------------------------------------
    # 1. Logistic Regression (Baseline)
    # -----------------------------------------------------------------
    logger.info("Training Logistic Regression baseline...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")

    # -----------------------------------------------------------------
    # 2. XGBoost
    # -----------------------------------------------------------------
    logger.info("Training XGBoost classifier...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # -----------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------
    save_artifact(lr_model, "lr_ctr_model.pkl")
    save_artifact(xgb_model, "xgb_ctr_model.pkl")
    save_artifact(encoders, "feature_encoders.pkl")

    # Save feature column list for inference
    feature_cols = list(X.columns)
    save_artifact(feature_cols, "feature_cols.pkl")

    logger.info("\nTraining complete. Artifacts saved to models/artifacts/")
    return {"lr": lr_metrics, "xgb": xgb_metrics}


if __name__ == "__main__":
    results = train()
