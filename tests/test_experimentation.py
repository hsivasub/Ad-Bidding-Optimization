"""
Unit testing for experimentation hashing logic and statistical analysis.
"""

import pytest
import numpy as np
import pandas as pd
from src.experimentation.ab_testing import (
    deterministic_hash,
    assign_variants,
    analyze_continuous_metric,
    analyze_conversion_metric
)

def test_deterministic_hash():
    # Calling hash on same ID should yield exactly same float
    h1 = deterministic_hash("user_A")
    h2 = deterministic_hash("user_A")
    assert h1 == h2
    assert 0.0 <= h1 < 1.0
    
    # Salt changes hash
    h3 = deterministic_hash("user_A", salt="diff")
    assert h1 != h3

def test_assign_variants():
    df = pd.DataFrame({
        "req_id": [f"user_{i}" for i in range(1000)]
    })
    
    variants = assign_variants(df, "req_id", split_ratio=0.5)
    
    # Checking approximate split and distinct classes
    assert set(variants) == {"Control", "Treatment"}
    t_count = (variants == "Treatment").sum()
    
    # Law of large numbers check for 50/50 split hashing (loose bound)
    assert 400 < t_count < 600

def test_analyze_continuous_metric():
    # Significant difference
    np.random.seed(42)
    control = np.random.normal(10, 2, 1000)
    treatment = np.random.normal(12, 2, 1000)
    
    res = analyze_continuous_metric(control, treatment, "Revenue")
    
    assert res["metric"] == "Revenue"
    assert res["significant"] is True
    assert res["p_value"] < 0.01
    assert res["relative_lift_pct"] > 0
    assert "test_type" in res

def test_analyze_conversion_metric():
    # Test binomial significance for proportions
    c_conv, c_tot = 100, 1000  # 10%
    t_conv, t_tot = 150, 1000  # 15%
    
    res = analyze_conversion_metric(c_conv, c_tot, t_conv, t_tot, "CVR")
    
    assert res["metric"] == "CVR"
    assert res["significant"] is True
    assert res["p_value"] < 0.05
    assert np.isclose(res["relative_lift_pct"], 50.0)  # 10 to 15 is a 50% relative lift
    
    # Test insignificant difference
    c_conv, c_tot = 100, 1000  # 10%
    t_conv, t_tot = 101, 1000  # 10.1%
    res_insig = analyze_conversion_metric(c_conv, c_tot, t_conv, t_tot, "CVR")
    assert res_insig["significant"] is False
