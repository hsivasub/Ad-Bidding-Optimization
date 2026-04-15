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
        model_name: 'xgb' (default), 'lr', 'xgb_v2' or 'lr_v2'.

    Returns:
        Array of predicted CTR probabilities, shape (n_samples,).
    """
    is_v2 = "_v2" in model_name
    
    encoders_file = "feature_encoders_v2.pkl" if is_v2 else "feature_encoders.pkl"
    cols_file = "feature_cols_v2.pkl" if is_v2 else "feature_cols.pkl"
    model_file = f"{model_name}_ctr_model.pkl" if not model_name.endswith("_ctr_model") else f"{model_name}.pkl"
    
    if model_name == "xgb":
        model_file = "xgb_ctr_model.pkl"
    elif model_name == "lr":
        model_file = "lr_ctr_model.pkl"
    elif model_name == "xgb_v2":
        model_file = "xgb_ctr_model_v2.pkl"
    elif model_name == "lr_v2":
        model_file = "lr_ctr_model_v2.pkl"

    encoders = load_artifact(encoders_file)
    feature_cols = load_artifact(cols_file)
    model = load_artifact(model_file)

    # Drop target col if present (inference mode)
    if "actual_click" not in df.columns:
        df = df.copy()
        df["actual_click"] = 0  # placeholder, not used in features

    if is_v2:
        # Load advanced feature state for V2 transforms
        from src.features.advanced_features import build_advanced_features
        from src.features.feature_engineering import encode_categoricals
        from src.features.advanced_features import ADVANCED_CATEGORICAL_COLS
        from sklearn.preprocessing import LabelEncoder
        
        adv_state = load_artifact("advanced_feature_state_v2.pkl")
        df_adv, _ = build_advanced_features(df, state=adv_state)
        
        df_enc, _ = encode_categoricals(df_adv, encoders=encoders)
        
        # Handle the custom interactions created in advanced features
        for col in ADVANCED_CATEGORICAL_COLS:
            if col in df_enc.columns:
                le = encoders.get(col)
                if isinstance(le, LabelEncoder):
                    known = set(le.classes_)
                    df_enc[col] = df_enc[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in known else -1
                    )
        X = df_enc[feature_cols]
    else:
        X, _, _ = build_feature_matrix(df, encoders=encoders)
        X = X[feature_cols]

    probs = model.predict_proba(X)[:, 1]
    logger.info(f"Generated {len(probs)} CTR predictions using {model_name} (mean={probs.mean():.4f})")
    return probs
