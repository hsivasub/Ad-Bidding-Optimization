import streamlit as st
import pandas as pd
import os
from PIL import Image

# Setup Streamlit page configuration
st.set_page_config(
    page_title="Ad Bidding Optimization Dashboard",
    page_icon="📈",
    layout="wide",
)

# Constants for Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Helper function to load data
@st.cache_data
def load_data(path, num_rows=1000):
    if os.path.exists(path):
        return pd.read_csv(path, nrows=num_rows)
    return None

# Helper function to load images
def load_image(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

def main():
    st.title("📈 Ad Bidding Optimization System")
    st.markdown("An end-to-end ad-tech simulation covering Data Generation, ML CTR Prediction, Supply Quality, and Bidding Strategies.")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    tabs = [
        "System Overview",
        "Data Exploration",
        "Model Performance",
        "Supply Quality",
        "Auction Simulation",
    ]
    selection = st.sidebar.radio("Go to", tabs)

    if selection == "System Overview":
        st.header("System Overview")
        st.markdown("""
        ### Welcome to the Ad Bidding Dashboard
        This application visualizes the output of the various phases of the ad-tech pipeline:
        
        1. **Data Exploration**: Review synthetic auction data simulating users, publishers, and bids.
        2. **Model Performance**: Analyze the performance of our Click-Through-Rate (CTR) prediction models (Phase 4/5).
        3. **Supply Quality**: See how publishers and exchanges are scored based on metrics like CTR, ROI, and Fraud Rate (Phase 6).
        4. **Auction Simulation**: Evaluate different bidding strategies (Static, CTR-based, Value-based) using a second-price auction simulator (Phase 7/8).
        """)
        st.info("Use the sidebar to navigate through the different sections of the pipeline.")

    elif selection == "Data Exploration":
        st.header("Data Exploration")
        st.markdown("Visualizing a sample of the raw synthetic auction dataset.")
        
        raw_data_path = os.path.join(DATA_DIR, "raw", "ad_auction_data.csv")
        df = load_data(raw_data_path, num_rows=10000)
        
        if df is not None:
            st.subheader("Raw Auction Data (Sample)")
            st.dataframe(df.head(100))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Bid Price Distribution")
                st.bar_chart(df["bid_price"].value_counts(bins=20).sort_index())
            with col2:
                st.subheader("Click-Through Rate (CTR)")
                ctr = df["actual_click"].mean()
                st.metric("Overall CTR", f"{ctr:.2%}")
                st.bar_chart(df["actual_click"].value_counts())
        else:
            st.warning("Raw data not found. Please run the data generation pipeline first.")

    elif selection == "Model Performance":
        st.header("CTR Model Performance")
        st.markdown("Evaluation metrics and diagnostic plots for the XGBoost CTR prediction model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Precision-Recall Curve")
            pr_img = load_image("pr_curves_v2.png")
            if pr_img:
                st.image(pr_img, use_container_width=True)
            else:
                st.info("PR curve not found in reports.")
                
            st.subheader("Calibration Curve")
            cal_img = load_image("calibration_v2.png")
            if cal_img:
                st.image(cal_img, use_container_width=True)
            else:
                st.info("Calibration curve not found in reports.")
                
        with col2:
            st.subheader("Feature Importance")
            feat_img = load_image("feature_importance_v2.png")
            if feat_img:
                st.image(feat_img, use_container_width=True)
            else:
                st.info("Feature importance chart not found in reports.")

    elif selection == "Supply Quality":
        st.header("Supply Quality Analysis")
        st.markdown("Scoring publishers and exchanges based on composite metrics (CTR, CVR, Fraud Rate, ROI).")
        
        pub_scores_path = os.path.join(REPORTS_DIR, "supply_quality", "publisher_scores.csv")
        pub_df = load_data(pub_scores_path)
        
        if pub_df is not None:
            st.subheader("Top Publishers")
            st.dataframe(pub_df.head(20))
        else:
            st.warning("Publisher scores CSV not found.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Publisher Ranking")
            rank_img = load_image(os.path.join("supply_quality", "publisher_ranking.png"))
            if rank_img:
                st.image(rank_img, use_container_width=True)
                
            st.subheader("CTR vs Fraud Scatter")
            scatter_img = load_image(os.path.join("supply_quality", "ctr_vs_fraud.png"))
            if scatter_img:
                st.image(scatter_img, use_container_width=True)
                
        with col2:
            st.subheader("Metrics Heatmap")
            heat_img = load_image(os.path.join("supply_quality", "metrics_heatmap.png"))
            if heat_img:
                st.image(heat_img, use_container_width=True)
                
            st.subheader("Exchange Ranking")
            exch_img = load_image(os.path.join("supply_quality", "exchange_ranking.png"))
            if exch_img:
                st.image(exch_img, use_container_width=True)

    elif selection == "Auction Simulation":
        st.header("Auction Simulation & Bidding Strategies")
        st.markdown("Evaluating the financial performance of different bidding strategies.")
        
        bid_results_path = os.path.join(REPORTS_DIR, "simulation", "bidding_results.csv")
        bid_df = load_data(bid_results_path)
        
        if bid_df is not None:
            st.subheader("Bidding Strategy Performance")
            st.dataframe(bid_df)
            
            # Simple bar chart comparing ROI
            st.subheader("ROI Comparison")
            bid_df_indexed = bid_df.set_index("strategy")
            if "roi_pct" in bid_df_indexed.columns:
                st.bar_chart(bid_df_indexed["roi_pct"])
        else:
            st.warning("Bidding simulation results not found. Please run the simulation pipeline.")
            
        ab_test_path = os.path.join(REPORTS_DIR, "simulation", "ab_test_report.csv")
        ab_df = load_data(ab_test_path)
        
        if ab_df is not None:
            st.subheader("A/B Testing Results")
            st.dataframe(ab_df)

if __name__ == "__main__":
    main()
