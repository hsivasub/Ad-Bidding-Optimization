"""
Unit logic validation for Bid Generation and Simulator processing.
"""

import pandas as pd
import numpy as np
import pytest
from src.bidding.strategies import StaticBidder, CTRBidder, ValueBasedBidder
from src.bidding.simulator import AuctionSimulator

@pytest.fixture
def mock_auction_data():
    return pd.DataFrame({
        "bid_price": [1.0, 2.0, 3.0, 1.5],
        "floor_price": [0.5, 1.0, 2.5, 1.0],
        "actual_click": [0, 1, 1, 0],
        "conversion": [0, 0, 1, 0],
        "revenue": [0.0, 0.0, 50.0, 0.0]
    })

def test_static_bidder():
    bidder = StaticBidder(default_bid=2.5)
    df = pd.DataFrame({"a": [1, 2, 3]})
    bids = bidder.calculate_bids(df)
    np.testing.assert_array_equal(bids, np.array([2.5, 2.5, 2.5]))

def test_ctr_bidder():
    bidder = CTRBidder(base_bid=2.0, target_ctr=0.05)
    df = pd.DataFrame({"a": [1, 2, 3]})
    pctr = np.array([0.10, 0.025, 0.30])
    # [ (0.10/0.05)*2=4.0, (0.025/0.05)*2=1.0, (0.30/0.05)*2=12.0 -> capped by np.clip max multiplier at 5x => 10.0 ]
    bids = bidder.calculate_bids(df, predicted_ctr=pctr)
    np.testing.assert_allclose(bids, np.array([4.0, 1.0, 10.0]))

def test_value_based_bidder():
    bidder = ValueBasedBidder(cpa_goal=20.0, avg_cvr=0.15)
    df = pd.DataFrame({"a": [1, 2]})
    pctr = np.array([0.10, 0.05])
    qual = np.array([100.0, 50.0])
    
    # row 1: base_EV = 20 * 0.15 * 0.1 = 0.3. Quality mult = 0.5 + (100/100) = 1.5 => 0.45
    # row 2: base_EV = 20 * 0.15 * 0.05 = 0.15. Quality mult = 0.5 + (50/100) = 1.0 => 0.15
    bids = bidder.calculate_bids(df, predicted_ctr=pctr, quality_scores=qual)
    np.testing.assert_allclose(bids, np.array([0.45, 0.15]))

def test_simulator_win_logic(mock_auction_data):
    sim = AuctionSimulator()
    # Let's bid varying amounts against the mock data
    # bids = [0.5 (lose context 1), 2.5 (wins against 2.0), 3.5 (wins against 3.0), 1.0 (loses against 1.5)]
    bids = np.array([0.5, 2.5, 3.5, 1.0])
    
    res = sim.simulate(mock_auction_data, bids, "TestBidder")
    
    assert res["auctions_won"] == 2
    assert res["total_clicks"] == 2
    assert res["total_conversions"] == 1
    
    # We won the 2.0 auction and 3.0 auction. Second price clearance = 2.0+0.01 + 3.0+0.01 = 5.02
    assert np.isclose(res["total_spend"], 5.02)
    assert res["total_revenue"] == 50.0
    
    roi = ((50.0 - 5.02) / 5.02) * 100
    assert np.isclose(res["roi_pct"], round(roi, 2))
