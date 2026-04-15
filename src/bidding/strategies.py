"""
Bidding Strategy Definitions — Phase 7.

Implements three programmatic bid generation strategies:
 1. StaticBidder: Fixed bid logic for baseline.
 2. CTRBidder: Bids proportional to predicted CTR probability.
 3. ValueBasedBidder: Modulates Expected Value (CPA * P(Conv|Click) * P(Click)) 
    with dynamic Supply Quality multipliers.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BiddingStrategy:
    def __init__(self, name: str):
        self.name = name

    def calculate_bids(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement calculate_bids()")


class StaticBidder(BiddingStrategy):
    """Bids a constant amount everywhere, providing a naive baseline."""
    def __init__(self, default_bid: float = 1.0):
        super().__init__("Static")
        self.default_bid = default_bid

    def calculate_bids(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        return np.full(len(df), self.default_bid)


class CTRBidder(BiddingStrategy):
    """Bids proportional to the predicted CTR."""
    def __init__(self, base_bid: float = 1.0, target_ctr: float = 0.05):
        super().__init__("CTR-Based")
        self.base_bid = base_bid
        self.target_ctr = target_ctr

    def calculate_bids(self, df: pd.DataFrame, predicted_ctr: np.ndarray = None, **kwargs) -> np.ndarray:
        if predicted_ctr is None:
            raise ValueError("CTRBidder requires predicted_ctr parameter.")
            
        # Scale bid based on how much better the predicted CTR is compared to threshold
        # Clip max bid multiplier at 5x to avoid catastrophic over-spending
        multiplier = np.clip(predicted_ctr / self.target_ctr, 0.0, 5.0)
        return self.base_bid * multiplier


class ValueBasedBidder(BiddingStrategy):
    """
    Bids estimated Expected Value (EV) modulated by publisher quality domain tier.
    EV = CPA_Goal * Avg_CVR * P(Click)
    """
    def __init__(self, cpa_goal: float = 25.0, avg_cvr: float = 0.15):
        super().__init__("Value-Based")
        self.cpa_goal = cpa_goal
        self.avg_cvr = avg_cvr

    def calculate_bids(self, df: pd.DataFrame, predicted_ctr: np.ndarray = None, quality_scores: np.ndarray = None, **kwargs) -> np.ndarray:
        if predicted_ctr is None or quality_scores is None:
            raise ValueError("ValueBasedBidder requires predicted_ctr and quality_scores parameters.")
            
        # Expected value
        base_ev = self.cpa_goal * self.avg_cvr * predicted_ctr
        
        # Modulate EV bid based on publisher quality score (0-100 scales to 0.5x -> 1.5x Multiplier)
        # 100 quality -> 1.5x base EV
        # 0 quality -> 0.5x base EV (or less, but clipped safely)
        quality_multiplier = 0.5 + (quality_scores / 100.0)
        final_bids = base_ev * quality_multiplier
        
        # Guardrails
        return np.clip(final_bids, 0.01, self.cpa_goal)
