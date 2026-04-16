"""
Experimentation Pipeline — Phase 8

Simulates an A/B test by deterministically splitting auction requests
into Control (Static Bidding) and Treatment (Value-Based Bidding).
Outputs a statistical significance report on business metrics.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import settings
from src.bidding.strategies import StaticBidder, ValueBasedBidder
from src.bidding.simulator import AuctionSimulator
from src.models.predict import predict_ctr
from src.supply_quality.scorer import score_entities
from src.experimentation.ab_testing import assign_variants, analyze_continuous_metric, analyze_conversion_metric

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    df_path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    df = pd.read_csv(df_path)
    
    # Take a validation holdout to run an experiment (simulate active traffic)
    df_exp = df.sample(frac=0.3, random_state=123).copy()
    
    logger.info(f"Simulating live traffic experimentation on {len(df_exp)} incoming user requests.")
    
    # 1. Deterministic A/B Split (50/50)
    # Ensure there is an identifier, using request_id or index if no ID.
    if "request_id" not in df_exp.columns:
        df_exp["request_id"] = ["req_" + str(i) for i in range(len(df_exp))]
        
    df_exp["variant"] = assign_variants(df_exp, "request_id", split_ratio=0.5, salt="phase8_exp")
    
    control_n = (df_exp["variant"] == "Control").sum()
    treatment_n = (df_exp["variant"] == "Treatment").sum()
    logger.info(f"Traffic Split -> Control (Static Baseline): {control_n} | Treatment (Value-Based AI): {treatment_n}")
    
    # 2. Get AI Inference features (Only practically required for Treatment, but computed for all for simplicity in batch)
    predicted_ctr = predict_ctr(df_exp, model_name="xgb_v2")
    pub_scores = score_entities(df_exp, group_col="publisher_id")
    pub_map = dict(zip(pub_scores["entity_id"], pub_scores["quality_score"]))
    quality_scores = df_exp["publisher_id"].map(pub_map).fillna(50.0).values
    
    # 3. Predict bids for both arms
    st_bidder = StaticBidder(default_bid=1.5)
    vb_bidder = ValueBasedBidder(cpa_goal=20.0, avg_cvr=0.15)
    
    control_bids = st_bidder.calculate_bids(df_exp)
    treatment_bids = vb_bidder.calculate_bids(df_exp, predicted_ctr=predicted_ctr, quality_scores=quality_scores)
    
    # 4. Multiplex bids based on assigned variant
    df_exp["final_bid"] = np.where(df_exp["variant"] == "Treatment", treatment_bids, control_bids)
    
    # 5. Run Auction Simulator
    simulator = AuctionSimulator()
    # We modify simulator flow slightly to assess per impression profit distributions for statistical testing
    df_exp["won_auction"] = df_exp["final_bid"] >= df_exp["bid_price"]
    df_exp["cost"] = np.where(df_exp["won_auction"], df_exp["bid_price"] + 0.01, 0.0)
    df_exp["profit"] = np.where(df_exp["won_auction"], df_exp["revenue"] - df_exp["cost"], 0.0)
    df_exp["conversion_event"] = np.where(df_exp["won_auction"], df_exp["conversion"], 0)
    df_exp["click_event"] = np.where(df_exp["won_auction"], df_exp["actual_click"], 0)
    
    logger.info("Executing auction clearance simulation...")

    c_mask = df_exp["variant"] == "Control"
    t_mask = df_exp["variant"] == "Treatment"

    # 6. Statistical Engine Metrics Computation
    results = []
    
    # Metric A: Win Rate (Proportion)
    c_wins, t_wins = df_exp[c_mask]["won_auction"].sum(), df_exp[t_mask]["won_auction"].sum()
    res_win = analyze_conversion_metric(c_wins, control_n, t_wins, treatment_n, "Win Rate")
    results.append(res_win)
    
    # Metric B: ROI (Profit continuous distribution)
    # Using profit per participated auction to align with generic value lift.
    c_profit = df_exp[c_mask]["profit"].values
    t_profit = df_exp[t_mask]["profit"].values
    res_profit = analyze_continuous_metric(c_profit, t_profit, "Profit Per Request")
    results.append(res_profit)
    
    # Metric C: Conversion Rate (Conversions / requests)
    c_convs, t_convs = df_exp[c_mask]["conversion_event"].sum(), df_exp[t_mask]["conversion_event"].sum()
    res_conv = analyze_conversion_metric(c_convs, control_n, t_convs, treatment_n, "Conversion Rate")
    results.append(res_conv)

    # Output formatting
    eval_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*85)
    logger.info("A/B Experiment Report — Incremental Value Assessment")
    logger.info("="*85)
    logger.info("\n" + eval_df.to_string(index=False))
    logger.info("="*85)
    
    # Flag successful experiment launch
    if res_profit["significant"] and res_profit["relative_lift_pct"] > 0:
        logger.info("\nSUCCESS: Treatment variant generated statistically significant positive profit lift!")
    elif res_profit["significant"] and res_profit["relative_lift_pct"] < 0:
        logger.warning("\nWARNING: Treatment variant generated statistically significant negative profit lift (Rollback recommended).")
    else:
        logger.info("\nNEUTRAL: Iteration required. No statistically significant profit shift detected.")
        
    out_path = os.path.join(settings.BASE_DIR, "reports", "simulation", "ab_test_report.csv")
    eval_df.to_csv(out_path, index=False)
    logger.info(f"Statistical readout exported to {out_path}")

if __name__ == "__main__":
    main()
