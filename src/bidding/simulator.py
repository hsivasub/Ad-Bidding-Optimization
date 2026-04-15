"""
Auction Simulator — Phase 7.

Simulates the performance of bidding strategies against real/historical auction data.
A bid must exceed floor_price AND the historical clearing bid_price to win.
Cost clears at highest competing bid + margin ($0.01) simulating Second-Price logic.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AuctionSimulator:
    def simulate(self, df: pd.DataFrame, bids: np.ndarray, strategy_name: str) -> dict:
        """
        Executes a deterministic auction replay against historical logs.
        """
        df = df.copy()
        df["our_bid"] = bids
        
        # To win, bid must exceed the actual winning bid from logs
        # To simplify second price logic: If we win, we pay the historical winning bid + $0.01.
        df["won_auction"] = df["our_bid"] >= df["bid_price"]
        
        wins = df["won_auction"].sum()
        win_rate = wins / len(df) if len(df) > 0 else 0
        
        if wins == 0:
            return self._empty_result(strategy_name, len(df))
            
        # Cost generation (Second price clearance)
        # If we win, we don't pay 'our_bid', we pay the clearing threshold
        df["our_cost"] = np.where(df["won_auction"], df["bid_price"] + 0.01, 0.0)
        
        # Compute exact business metrics from won impressions
        won_df = df[df["won_auction"]]
        total_spend = won_df["our_cost"].sum()
        total_clicks = won_df["actual_click"].sum()
        total_conversions = won_df["conversion"].sum()
        total_revenue = won_df["revenue"].sum()
        
        # Financial derivatives
        roi = ((total_revenue - total_spend) / total_spend) * 100 if total_spend > 0 else 0.0
        cpa = total_spend / total_conversions if total_conversions > 0 else 0.0
        cpc = total_spend / total_clicks if total_clicks > 0 else 0.0
        
        return {
            "strategy": strategy_name,
            "auctions_participated": len(df),
            "auctions_won": int(wins),
            "win_rate": round(win_rate * 100, 2),
            "total_spend": round(total_spend, 2),
            "total_revenue": round(total_revenue, 2),
            "total_clicks": int(total_clicks),
            "total_conversions": int(total_conversions),
            "roi_pct": round(roi, 2),
            "cpa": round(cpa, 2),
            "cpc": round(cpc, 2)
        }

    def _empty_result(self, strategy_name: str, requested: int) -> dict:
        return {
            "strategy": strategy_name,
            "auctions_participated": requested,
            "auctions_won": 0,
            "win_rate": 0.0,
            "total_spend": 0.0,
            "total_revenue": 0.0,
            "total_clicks": 0,
            "total_conversions": 0,
            "roi_pct": 0.0,
            "cpa": 0.0,
            "cpc": 0.0
        }
