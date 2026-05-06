"""
Phase 14: Final Streamlit Dashboard for Ad Bidding Optimization

Visualizes:
1. Publisher Supply Quality Scores
2. A/B Testing Results (Traffic Splits & Lift)
3. Model Evaluation Metrics from MLflow (if available locally) or static report
4. Raw Data Explorer
"""

import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.settings import settings

st.set_page_config(page_title="Ad Bidding Dashboard", layout="wide", page_icon="📈")

# ------------------------------------------------------------------
# Data Loading Helpers
# ------------------------------------------------------------------
@st.cache_data
def load_processed_data() -> pd.DataFrame:
    path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_publisher_scores() -> pd.DataFrame:
    path = os.path.join(settings.BASE_DIR, "reports", "publisher_quality_scores.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_ab_test_results() -> dict:
    path = os.path.join(settings.BASE_DIR, "reports", "ab_test_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# ------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------
st.sidebar.title("Ad Bidding Hub 🎯")
page = st.sidebar.radio(
    "Navigation", 
    ["Overview & Raw Data", "Supply Quality Scoring", "A/B Testing Engine"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Roadmap: Phase 14**\n\n"
    "End-to-End Real Time Bidding (RTB) Optimizer."
)

df = load_processed_data()
pub_df = load_publisher_scores()
ab_results = load_ab_test_results()

# ------------------------------------------------------------------
# Page 1: Overview
# ------------------------------------------------------------------
if page == "Overview & Raw Data":
    st.title("Dataset Overview & Features")
    
    if df.empty:
        st.warning("Processed dataset not found! Run `src/data/generate_synthetic_data.py`")
    else:
        st.write("### Data Snapshot")
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Auctions Evaluated", f"{len(df):,}")
        col2.metric("Overall CTR", f"{(df['actual_click'].mean() * 100):.2f}%")
        col3.metric("Total Advertising Spend", f"${df['cost'].sum():,.2f}")
        
        st.markdown("---")
        st.write("### Clicks by Device Type")
        fig1 = px.histogram(df, x="device_type", color="actual_click", barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------
# Page 2: Supply Quality
# ------------------------------------------------------------------
elif page == "Supply Quality Scoring":
    st.title("Publisher Supply Quality 🛡️")
    st.markdown("Trust & Safety scores computed dynamically based on historical CTR, viewability, and fraud proxy metrics.")
    
    if pub_df.empty:
        st.warning("Quality scores not found! Run `scripts/run_supply_quality.py`")
    else:
        # Display top tier vs bottom tier
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Top Premium Publishers")
            st.dataframe(pub_df.head(10)[["publisher_id", "composite_score", "trust_tier"]], use_container_width=True)
        with col2:
            st.write("#### Blocklist Candidates")
            st.dataframe(pub_df.tail(10)[["publisher_id", "composite_score", "trust_tier"]], use_container_width=True)
            
        fig2 = px.box(pub_df, x="trust_tier", y="composite_score", color="trust_tier", title="Score Distribution by Tier")
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = px.scatter(pub_df, x="historical_ctr", y="composite_score", color="trust_tier", hover_data=["publisher_id"], title="CTR vs Composite Score")
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# Page 3: A/B Testing
# ------------------------------------------------------------------
elif page == "A/B Testing Engine":
    st.title("Experimentation Platform 🔬")
    st.markdown("Results from our deterministic hashing traffic splitter and statistical validation tests.")
    
    if not ab_results:
        st.warning("A/B test results not found! Run `scripts/run_ab_experiment.py`")
    else:
        st.write(f"**Experiment:** {ab_results.get('experiment_id', 'N/A')}")
        
        control = ab_results.get("groups", {}).get("control", {})
        treatment = ab_results.get("groups", {}).get("treatment", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("🟦 **Control Group (Legacy Strategy)**")
            st.write(f"- Users: {control.get('count', 0):,}")
            st.write(f"- Conversions: {control.get('conversions', 0):,}")
            st.write(f"- Conversion Rate: {control.get('conversion_rate', 0)*100:.2f}%")
            
        with col2:
            st.success("🟩 **Treatment Group (ML Optimized)**")
            st.write(f"- Users: {treatment.get('count', 0):,}")
            st.write(f"- Conversions: {treatment.get('conversions', 0):,}")
            st.write(f"- Conversion Rate: {treatment.get('conversion_rate', 0)*100:.2f}%")
            
        st.markdown("---")
        st.write("### Statistical Significance")
        
        stats = ab_results.get("statistics", {})
        st.write(f"- **Relative Lift:** {stats.get('relative_lift_pct', 0):.2f}%")
        st.write(f"- **P-Value:** {stats.get('p_value', 1.0):.6f}")
        st.write(f"- **Confidence Interval (95%):** {stats.get('confidence_interval_95_pct', [])}")
        
        if stats.get("significant"):
            st.balloons()
            st.success("✅ **Result:** The Treatment strategy outperformed Control with statistical significance!")
        else:
            st.warning("❌ **Result:** No statistically significant difference detected.")
