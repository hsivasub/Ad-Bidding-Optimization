"""
Advanced feature engineering for improved CTR model (Phase 5).

Adds:
  - Frequency encoding: replaces high-cardinality categoricals with count-based encodings
  - Bid-to-floor ratio: captures how aggressively we bid relative to the floor
  - Hour-of-day bucketing: morning / afternoon / evening / night
  - Publisher × Device interaction feature
  - Rolling/historical mean CTR per publisher (computed on training data only)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def add_bid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive auction pricing features from raw bid and floor price columns."""
    df = df.copy()
    # How aggressively we bid relative to the floor
    df["bid_to_floor_ratio"] = df["bid_price"] / (df["floor_price"] + 1e-6)
    # Absolute bid surplus above floor
    df["bid_surplus"] = df["bid_price"] - df["floor_price"]
    return df


def add_time_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket hour_of_day into four named periods:
      0-5   → night (0)
      6-11  → morning (1)
      12-17 → afternoon (2)
      18-23 → evening (3)
    """
    df = df.copy()
    conditions = [
        df["hour_of_day"].between(0, 5),
        df["hour_of_day"].between(6, 11),
        df["hour_of_day"].between(12, 17),
        df["hour_of_day"].between(18, 23),
    ]
    df["time_bucket"] = np.select(conditions, [0, 1, 2, 3], default=0)
    return df


def add_frequency_encoding(
    df: pd.DataFrame,
    freq_maps: Optional[dict] = None,
    high_card_cols: Optional[list] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Replace raw categorical IDs with their frequency (count) across the dataset.
    Useful for high-cardinality columns like publisher_id and ad_id.

    In inference mode, pass `freq_maps` built from training data.
    Unseen categories are filled with 1 (single occurrence / cold-start proxy).

    Returns:
        df with new *_freq columns added
        freq_maps dict (serialisable for inference reuse)
    """
    df = df.copy()
    if high_card_cols is None:
        high_card_cols = ["publisher_id", "ad_id", "campaign_id"]

    fit_mode = freq_maps is None
    if fit_mode:
        freq_maps = {}

    for col in high_card_cols:
        if col not in df.columns:
            continue
        freq_col = f"{col}_freq"
        if fit_mode:
            freq_map = df[col].value_counts().to_dict()
            freq_maps[col] = freq_map
        else:
            freq_map = freq_maps.get(col, {})

        df[freq_col] = df[col].map(freq_map).fillna(1).astype(float)

    return df, freq_maps


def add_publisher_mean_ctr(
    df: pd.DataFrame,
    pub_ctr_map: Optional[dict] = None,
    global_ctr: Optional[float] = None,
) -> tuple[pd.DataFrame, dict, float]:
    """
    Add historical mean CTR per publisher as a feature.

    In training mode (pub_ctr_map=None), compute from the dataset.
    In inference mode, apply the training map. Unknown publishers fall back to
    the global CTR computed during training.

    Returns:
        df with publisher_mean_ctr column
        pub_ctr_map dict
        global_ctr float
    """
    df = df.copy()
    if pub_ctr_map is None:
        pub_ctr_map = df.groupby("publisher_id")["actual_click"].mean().to_dict()
        global_ctr = df["actual_click"].mean()

    df["publisher_mean_ctr"] = (
        df["publisher_id"].map(pub_ctr_map).fillna(global_ctr)
    )
    return df, pub_ctr_map, global_ctr


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple string-concat interaction features which tree models can exploit
    even after label-encoding.
    """
    df = df.copy()
    # Publisher × Device type combination
    df["pub_device"] = (
        df["publisher_id"].astype(str) + "_" + df["device_type"].astype(str)
    )
    return df


# ------------------------------------------------------------------
# Columns produced by this module (appended to base feature set)
# ------------------------------------------------------------------
ADVANCED_NUMERIC_COLS = [
    "bid_to_floor_ratio",
    "bid_surplus",
    "time_bucket",
    "publisher_id_freq",
    "ad_id_freq",
    "campaign_id_freq",
    "publisher_mean_ctr",
]

ADVANCED_CATEGORICAL_COLS = ["pub_device"]


def build_advanced_features(
    df: pd.DataFrame,
    state: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Master function: apply all advanced transforms in order.

    `state` is a dict carrying fitted transform parameters so they can be
    applied consistently at inference time:
        state = {
            "freq_maps": {...},
            "pub_ctr_map": {...},
            "global_ctr": float,
        }

    Returns:
        df_enriched with all new columns
        state dict (for serialisation / reuse at inference)
    """
    fit_mode = state is None
    if fit_mode:
        state = {}

    df = add_bid_features(df)
    df = add_time_bucket(df)

    df, freq_maps = add_frequency_encoding(
        df, freq_maps=state.get("freq_maps") if not fit_mode else None
    )
    state["freq_maps"] = freq_maps

    # publisher_mean_ctr requires actual_click in df; skip during pure inference
    if "actual_click" in df.columns:
        df, pub_ctr_map, global_ctr = add_publisher_mean_ctr(
            df,
            pub_ctr_map=state.get("pub_ctr_map") if not fit_mode else None,
            global_ctr=state.get("global_ctr") if not fit_mode else None,
        )
        state["pub_ctr_map"] = pub_ctr_map
        state["global_ctr"] = global_ctr
    else:
        # Inference: apply saved map or fill with global CTR
        pub_ctr_map = state.get("pub_ctr_map", {})
        global_ctr = state.get("global_ctr", 0.06)
        df["publisher_mean_ctr"] = df["publisher_id"].map(pub_ctr_map).fillna(global_ctr)

    df = add_interaction_features(df)

    logger.info(
        f"Advanced features built — shape: {df.shape}, "
        f"new cols: {ADVANCED_NUMERIC_COLS + ADVANCED_CATEGORICAL_COLS}"
    )
    return df, state
