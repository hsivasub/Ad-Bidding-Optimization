"""
Experimentation and A/B Testing module — Phase 8

Provides statistical machinery to assess significance between
bidding algorithms or configurations simulated atop deterministic traffic splits.
"""

import hashlib
import numpy as np
import pandas as pd
import logging
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

logger = logging.getLogger(__name__)

def deterministic_hash(identifier: str, salt: str = "ab_test_salt") -> float:
    """
    Generate a deterministic float [0.0, 1.0) from a string identifier.
    Used for consistent user/request level test assignments.
    """
    h = hashlib.md5((str(identifier) + salt).encode()).hexdigest()
    return int(h, 16) / (16 ** 32)


def assign_variants(df: pd.DataFrame, id_col: str, split_ratio: float = 0.5, salt: str = "exp_1") -> pd.Series:
    """
    Assign DataFrame rows to Control or Treatment deterministically.
    
    Args:
        df: Input DataFrame.
        id_col: Column name containing the unique routing key (e.g., request_id).
        split_ratio: Fraction of traffic allocated to Treatment (e.g., 0.5 for 50/50).
        salt: Salt applied to hashing to prevent correlated assignments across tests.
        
    Returns:
        pd.Series populated with "Control" and "Treatment" labels.
    """
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in DataFrame.")
        
    hashes = df[id_col].astype(str).apply(lambda x: deterministic_hash(x, salt))
    return np.where(hashes < split_ratio, "Treatment", "Control")


def analyze_continuous_metric(control_vals: np.ndarray, treatment_vals: np.ndarray, metric_name: str) -> dict:
    """
    Execute Welch's T-Test for continuous metrics (Revenue, Profit, CPA).
    """
    c_m, t_m = np.mean(control_vals), np.mean(treatment_vals)
    lift = ((t_m - c_m) / c_m * 100) if c_m != 0 else 0.0
    
    # Handle edge case where variances are zero or empty arrays
    if len(control_vals) < 2 or len(treatment_vals) < 2:
        return _empty_test_result(metric_name, c_m, t_m, lift)
        
    t_stat, p_val = stats.ttest_ind(treatment_vals, control_vals, equal_var=False)
    
    return {
        "metric": metric_name,
        "test_type": "Welch T-Test",
        "control_mean": c_m,
        "treatment_mean": t_m,
        "absolute_diff": t_m - c_m,
        "relative_lift_pct": lift,
        "p_value": p_val,
        "significant": bool(p_val < 0.05)
    }


def analyze_conversion_metric(c_success: int, c_total: int, t_success: int, t_total: int, metric_name: str) -> dict:
    """
    Execute Two-Proportion Z-Test for binomial metrics (Win Rate, CTR, CVR).
    """
    c_rate = c_success / max(c_total, 1)
    t_rate = t_success / max(t_total, 1)
    lift = ((t_rate - c_rate) / c_rate * 100) if c_rate > 0 else 0.0
    
    if c_total == 0 or t_total == 0:
        return _empty_test_result(metric_name, c_rate, t_rate, lift)
        
    counts = np.array([t_success, c_success])
    nobs = np.array([t_total, c_total])
    
    try:
        z_stat, p_val = proportions_ztest(counts, nobs)
    except Exception as e:
        logger.warning(f"Z-test failed for {metric_name}: {e}")
        return _empty_test_result(metric_name, c_rate, t_rate, lift)
        
    return {
        "metric": metric_name,
        "test_type": "Proportions Z-Test",
        "control_mean": c_rate,
        "treatment_mean": t_rate,
        "absolute_diff": t_rate - c_rate,
        "relative_lift_pct": lift,
        "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
        "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False
    }


def _empty_test_result(metric: str, c_m: float, t_m: float, lift: float) -> dict:
    return {
        "metric": metric,
        "test_type": "None",
        "control_mean": c_m,
        "treatment_mean": t_m,
        "absolute_diff": t_m - c_m,
        "relative_lift_pct": lift,
        "p_value": 1.0,
        "significant": False
    }
