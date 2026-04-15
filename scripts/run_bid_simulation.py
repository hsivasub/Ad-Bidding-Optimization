"""
Standalone script to execute bidirectional bidding simulations.
Generates baseline vs ML-optimized strategy impact stats.

Usage:
    python scripts/run_bid_simulation.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.settings import settings
from src.bidding.strategies import StaticBidder, CTRBidder, ValueBasedBidder
from src.bidding.simulator import AuctionSimulator
from src.models.predict import predict_ctr
from src.supply_quality.scorer import score_entities

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    df_path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing {df_path}. Run generate_synthetic_data.py first.")
        
    df = pd.read_csv(df_path)
    
    # 20% validation sample for bid tuning
    df_sim = df.sample(frac=0.2, random_state=42).copy()
    logger.info(f"Loaded {len(df_sim)} target auctions for bid simulation replay.")
    
    # 1. Prediction mapping
    logger.info("Generating internal CTR probability inferences (XGBoost v2)...")
    predicted_ctr = predict_ctr(df_sim, model_name="xgb_v2")
    
    # 2. Quality Domain Extraction
    logger.info("Computing publisher composite quality multipliers...")
    pub_scores = score_entities(df_sim, group_col="publisher_id")
    pub_map = dict(zip(pub_scores["entity_id"], pub_scores["quality_score"]))
    quality_scores = df_sim["publisher_id"].map(pub_map).fillna(50.0).values
    
    # 3. Attach strategies
    strategies = [
        StaticBidder(default_bid=1.5),
        CTRBidder(base_bid=1.2, target_ctr=0.06), # Our baseline CTR threshold
        ValueBasedBidder(cpa_goal=20.0, avg_cvr=0.15)
    ]
    
    simulator = AuctionSimulator()
    results = []
    
    logger.info("Executing algorithmic bidding simulations...")
    for strategy in strategies:
        logger.info(f"  -> Simulating [{strategy.name}]...")
        bids = strategy.calculate_bids(df_sim, predicted_ctr=predicted_ctr, quality_scores=quality_scores)
        res = simulator.simulate(df_sim, bids, strategy.name)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    
    logger.info("\n" + "="*80)
    logger.info("Bidding Strategy Simulation Outcomes")
    logger.info("="*80)
    logger.info("\n" + res_df.to_string(index=False))
    logger.info("="*80)
    
    out_path = os.path.join(settings.BASE_DIR, "reports", "simulation", "bidding_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res_df.to_csv(out_path, index=False)
    logger.info(f"\nFinal numerical simulation outcomes saved to {out_path}")

if __name__ == "__main__":
    main()
