"""
FastAPI Inference Application — Phase 9

Loads ML artifacts into memory on startup and serves the /bid route
returning real-time ValueBasedBidder ad-auction outputs driven by XGBoost predictions.
"""

import time
import logging
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.api.schemas import BidRequest, BidResponse
from src.models.predict import predict_ctr
from src.bidding.strategies import ValueBasedBidder
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Global memory state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts and configuration on startup."""
    logger.info("Initializing ML models for inference service...")
    
    # Pre-instantiate Bidder mapping standard ML performance targets
    app_state["bidder"] = ValueBasedBidder(cpa_goal=20.0, avg_cvr=0.15)
    
    # Load Publisher Supply Quality Cache (Phase 6 artifacts)
    pub_path = os.path.join(settings.BASE_DIR, "reports", "supply_quality", "publisher_scores.csv")
    quality_map = {}
    if os.path.exists(pub_path):
        try:
            df_pub = pd.read_csv(pub_path)
            quality_map = dict(zip(df_pub["entity_id"], df_pub["quality_score"]))
            logger.info(f"Loaded {len(quality_map)} publisher quality scores into cache.")
        except Exception as e:
            logger.error(f"Error loading quality scores: {e}")
            
    app_state["quality_map"] = quality_map
    
    # Pre-flight warm up standardizing XGBoost models into RAM
    # Generate dummy request just to trigger all the pickle loads.
    try:
        dummy = pd.DataFrame([{
            "request_id": "warmup", "campaign_id": "camp_X", "ad_id": "ad_X", "user_id": "u",
            "publisher_id": "pub_X", "exchange_id": "ex_X", "device_type": "mobile",
            "os_family": "android", "country": "US", "hour_of_day": 12, "floor_price": 1.0,
            "bid_price": 1.0
        }])
        _ = predict_ctr(dummy, model_name="xgb_v2")
        logger.info("Prediction pipelines warmed up successfully.")
    except Exception as e:
        logger.error(f"Warmup warning (ignorable if artifacts exist): {e}")

    yield
    app_state.clear()


app = FastAPI(
    title="Ad Bidding Optimizer API",
    description="Real-time ML bidding and CTR prediction inference service.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "bidding-optimization-inference"}

@app.post("/bid", response_model=BidResponse)
def compute_bid(request: BidRequest):
    """
    Main Real Time Bidding endpoint route.
    Transforms incoming payload into features, predicts probability of click,
    maps domain multiplier, and mathematically structures second-price auction bid.
    """
    start_time = time.time()
    
    try:
        # Convert incoming JSON payload to DataFrame
        req_dict = request.model_dump()
        
        # Inject standard proxy values required for the historical Feature Engineering states (Phase 5)
        # Because we haven't bid yet, the "bid_surplus" state requires a dummy bid to normalize calculations
        req_dict["bid_price"] = req_dict["floor_price"] * 1.1 
        
        # In pydantic alias we took os_family and mapped it to os. The dict handles this properly via by_alias.
        req_dict = request.model_dump()
        req_dict["bid_price"] = req_dict["floor_price"] * 1.1

        input_df = pd.DataFrame([req_dict])
        
        # Calculate dynamic date/weekend bounds
        input_df['is_weekend'] = input_df['day_of_week'].isin([5, 6]).astype(int)
        
        # 1. High Velocity XGBoost CTR Probability extraction
        ctr_array = predict_ctr(input_df, model_name="xgb_v2")
        p_ctr = float(ctr_array[0])
        
        # 2. Extract Publisher domain safety multipliers
        q_score = app_state["quality_map"].get(request.publisher_id, 50.0)
        
        # 3. Calculate final bounded EV bid logic
        bidder = app_state["bidder"]
        bids = bidder.calculate_bids(
            input_df, 
            predicted_ctr=np.array([p_ctr]), 
            quality_scores=np.array([q_score])
        )
        final_bid = float(bids[0])
        
        # 4. Check if bid actually meets Exchange floor parameters
        bid_placed = final_bid >= request.floor_price
        
        proc_time = (time.time() - start_time) * 1000
        
        return BidResponse(
            request_id=request.request_id,
            predicted_ctr=round(p_ctr, 5),
            bid_price=round(final_bid, 3), 
            bid_placed=bid_placed,
            strategy_used="ValueBasedBidder_XGB_v2",
            processing_time_ms=round(proc_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Inference error executing logic chain: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during bid generation.")
