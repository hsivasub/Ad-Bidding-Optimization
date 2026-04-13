"""
Tests for advanced feature engineering (Phase 5).
"""

import numpy as np
import pytest
import pandas as pd

from src.data.generate_synthetic_data import generate_ad_data
from src.features.advanced_features import (
    add_bid_features,
    add_time_bucket,
    add_frequency_encoding,
    add_publisher_mean_ctr,
    add_interaction_features,
    build_advanced_features,
)


@pytest.fixture
def sample_df():
    df = generate_ad_data(num_samples=300)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["roi"] = 0.0
    return df


def test_bid_features_values(sample_df):
    df = add_bid_features(sample_df)
    assert "bid_to_floor_ratio" in df.columns
    assert "bid_surplus" in df.columns
    # bid_surplus should match manual calculation
    expected_surplus = sample_df["bid_price"] - sample_df["floor_price"]
    np.testing.assert_allclose(df["bid_surplus"].values, expected_surplus.values, rtol=1e-5)


def test_time_bucket_range(sample_df):
    df = add_time_bucket(sample_df)
    assert "time_bucket" in df.columns
    # All values must be in {0, 1, 2, 3}
    assert set(df["time_bucket"].unique()).issubset({0, 1, 2, 3})


def test_frequency_encoding_fit_apply(sample_df):
    df_train, freq_maps = add_frequency_encoding(sample_df)
    assert "publisher_id_freq" in df_train.columns
    assert isinstance(freq_maps, dict)
    assert "publisher_id" in freq_maps

    # Apply to new batch — should not raise
    df_test = generate_ad_data(num_samples=50)
    df_test["is_weekend"] = df_test["day_of_week"].isin([5, 6]).astype(int)
    df_test["roi"] = 0.0
    df_test_enc, _ = add_frequency_encoding(df_test, freq_maps=freq_maps)
    assert df_test_enc["publisher_id_freq"].notna().all()


def test_publisher_mean_ctr_fallback(sample_df):
    df, pub_ctr_map, global_ctr = add_publisher_mean_ctr(sample_df)
    assert "publisher_mean_ctr" in df.columns
    # Create a row with an unknown publisher
    df_unknown = sample_df.copy()
    df_unknown["publisher_id"] = "pub_UNKNOWN_9999"
    df_unknown, _, _ = add_publisher_mean_ctr(
        df_unknown, pub_ctr_map=pub_ctr_map, global_ctr=global_ctr
    )
    # All unknown publishers should fall back to global_ctr
    assert (df_unknown["publisher_mean_ctr"] == global_ctr).all()


def test_interaction_feature(sample_df):
    df = add_interaction_features(sample_df)
    assert "pub_device" in df.columns
    # pub_device should be a combination of publisher_id and device_type
    first_row = df.iloc[0]
    assert first_row["pub_device"] == f"{first_row['publisher_id']}_{first_row['device_type']}"


def test_build_advanced_features_stateful(sample_df):
    """State returned from training can be reused for inference without re-fitting."""
    df_out, state = build_advanced_features(sample_df)
    assert "freq_maps" in state
    assert "pub_ctr_map" in state

    df_new = generate_ad_data(num_samples=50)
    df_new["is_weekend"] = df_new["day_of_week"].isin([5, 6]).astype(int)
    df_new["roi"] = 0.0
    df_new_out, _ = build_advanced_features(df_new, state=state)
    assert "bid_to_floor_ratio" in df_new_out.columns
    assert "publisher_mean_ctr" in df_new_out.columns
