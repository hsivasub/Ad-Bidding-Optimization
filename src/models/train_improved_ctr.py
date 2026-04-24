"""
Improved CTR Training Pipeline (Phase 5).

Enhancements over baseline (Phase 4):
  - Advanced feature engineering:  bid-to-floor ratio, frequency encoding,
    time buckets, publisher mean CTR, publisher × device interaction
  - XGBoost v2 with early stopping on validation AUC
  - Side-by-side model comparison (LR, XGB-baseline, XGB-improved)
  - PR curves, calibration curve, and feature importance charts saved to reports/
  - Artifacts clearly versioned with "_v2" suffix to avoid overwriting baseline
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.config.settings import settings
from src.features.feature_engineering import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    encode_categoricals,
)
from src.features.advanced_features import (
    build_advanced_features,
    ADVANCED_NUMERIC_COLS,
    ADVANCED_CATEGORICAL_COLS,
)
from src.models.evaluate import (
    compute_metrics,
    compare_models,
    plot_precision_recall_curves,
    plot_feature_importance,
    plot_calibration_curve,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_OUTPUT_DIR = os.path.join(settings.BASE_DIR, "models", "artifacts")
RANDOM_STATE = 42


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_processed_data() -> pd.DataFrame:
    path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at {path}. "
            "Run generate_synthetic_data.py and preprocess.py first."
        )
    logger.info(f"Loading processed dataset from {path}")
    return pd.read_csv(path)


def save_artifact(obj, filename: str) -> str:
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(MODEL_OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved artifact → {path}")
    return path


def build_improved_feature_matrix(
    df: pd.DataFrame,
    label_encoders: dict = None,
    adv_state: dict = None,
) -> tuple[pd.DataFrame, np.ndarray, dict, dict]:
    """
    Full improved feature pipeline.

    Steps:
      1. Advanced transforms (bid ratio, freq encoding, publisher CTR, etc.)
      2. Label-encode all categoricals (base + new interaction col)
      3. Select final feature columns

    Returns:
        X, y, label_encoders, adv_state
    """
    # 1. Advanced transforms
    df_adv, adv_state = build_advanced_features(df, state=adv_state)

    # 2. Combine column lists
    all_cat_cols = CATEGORICAL_COLS + ADVANCED_CATEGORICAL_COLS
    all_num_cols = NUMERIC_COLS + ADVANCED_NUMERIC_COLS

    # 3. Label-encode (base + new interaction cols)
    df_enc, label_encoders = encode_categoricals(
        df_adv, encoders=label_encoders  # type: ignore
    )
    # encode_categoricals only covers CATEGORICAL_COLS; handle ADVANCED_CATEGORICAL_COLS
    if label_encoders is None:
        label_encoders = {}
    fit_mode = "pub_device" not in label_encoders
    for col in ADVANCED_CATEGORICAL_COLS:
        if col not in df_enc.columns:
            continue
        le = label_encoders.get(col, LabelEncoder())
        if fit_mode:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            label_encoders[col] = le
        else:
            known = set(le.classes_)
            df_enc[col] = df_enc[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    feature_cols = all_cat_cols + all_num_cols
    # Drop any columns that didn't get produced (safety check)
    feature_cols = [c for c in feature_cols if c in df_enc.columns]

    X = df_enc[feature_cols]
    y = df_enc[TARGET_COL].values

    logger.info(
        f"Improved feature matrix: shape={X.shape}, "
        f"positive_rate={y.mean():.4f}"
    )
    return X, y, label_encoders, adv_state


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train():
    # Phase 11: Setup MLflow tracking UI
    # MLflow interprets C:\ as a scheme. Using a relative local path bypasses uri strictness.
    mlruns_path = "logs/mlruns"
    os.makedirs(mlruns_path, exist_ok=True)
    
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("Ad_Bidding_CTR_Optimization")
    logger.info(f"MLflow tracking configured -> {mlruns_path}")

    df = load_processed_data()

    X, y, label_encoders, adv_state = build_improved_feature_matrix(df)

    # 60 / 20 / 20 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    logger.info(f"Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    feature_cols = list(X.columns)

    # ------------------------------------------------------------------
    # 1. Logistic Regression (improved features, same algorithm as baseline)
    # ------------------------------------------------------------------
    logger.info("Training Logistic Regression (improved features)...")
    lr_params = {
        "max_iter": 3000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "solver": "saga",   # saga handles larger datasets / more features better than lbfgs
    }
    
    with mlflow.start_run(run_name="LogisticRegression_v2"):
        mlflow.log_params(lr_params)
        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(X_train, y_train)
        
        y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
        res_lr = compute_metrics(y_test, y_prob_lr, "LogisticRegression_v2")
        mlflow.log_metrics({k: v for k, v in res_lr.items() if k != "model"})
        mlflow.sklearn.log_model(lr_model, "lr_classifier_model")

    # ------------------------------------------------------------------
    # 2. XGBoost improved — with early stopping
    # ------------------------------------------------------------------
    logger.info("Training XGBoost (improved features + early stopping)...")
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    xgb_params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 30,
    }
    
    with mlflow.start_run(run_name="XGBoost_v2"):
        mlflow.log_params(xgb_params)
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_n = float(xgb_model.best_iteration)
        logger.info(f"XGBoost best iteration (early stopping): {best_n}")
        mlflow.log_metric("best_iteration", best_n)
        
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        res_xgb = compute_metrics(y_test, y_prob_xgb, "XGBoost_v2")
        mlflow.log_metrics({k: v for k, v in res_xgb.items() if k != "model"})
        mlflow.xgboost.log_model(xgb_model, "xgb_classifier_model")

    # ------------------------------------------------------------------
    # Evaluation & comparison
    # ------------------------------------------------------------------
    models_dict = {
        "LogisticRegression_v2": lr_model,
        "XGBoost_v2": xgb_model,
    }

    comparison_df = compare_models([res_lr, res_xgb])

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------
    plot_precision_recall_curves(models_dict, X_test, y_test, "pr_curves_v2.png")
    plot_feature_importance(xgb_model, feature_cols, top_n=20, filename="feature_importance_v2.png")
    plot_calibration_curve(models_dict, X_test, y_test, filename="calibration_v2.png")

    # ------------------------------------------------------------------
    # Save improved artifacts (v2 suffix to preserve baseline)
    # ------------------------------------------------------------------
    save_artifact(lr_model, "lr_ctr_model_v2.pkl")
    save_artifact(xgb_model, "xgb_ctr_model_v2.pkl")
    save_artifact(label_encoders, "feature_encoders_v2.pkl")
    save_artifact(adv_state, "advanced_feature_state_v2.pkl")
    save_artifact(feature_cols, "feature_cols_v2.pkl")

    logger.info("Phase 5 training complete — artifacts and plots saved.")
    return {"comparison": comparison_df, "models": models_dict}


if __name__ == "__main__":
    train()
