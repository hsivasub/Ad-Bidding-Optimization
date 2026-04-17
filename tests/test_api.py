"""
Tests for the FastAPI Inference Service (Phase 9)
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "bidding-optimization-inference"}

def test_bid_endpoint_success():
    payload = {
        "request_id": "req_pytest_123",
        "campaign_id": "camp_5",
        "ad_id": "ad_99",
        "user_id": "user_xyz",
        "publisher_id": "pub_18",  # Should trigger High quality multiplier
        "exchange_id": "ex_appnexus",
        "device_type": "mobile",
        "os_family": "ios",
        "country": "US",
        "time_of_day": "08:00",
        "hour_of_day": 8,
        "day_of_week": 2,
        "floor_price": 0.50
    }
    
    with TestClient(app) as client:
        response = client.post("/bid", json=payload)
    
    # We should get 200 OK because the models are cached properly
    assert response.status_code == 200
    
    data = response.json()
    assert data["request_id"] == "req_pytest_123"
    assert "predicted_ctr" in data
    assert "bid_price" in data
    assert "bid_placed" in data
    assert data["strategy_used"] == "ValueBasedBidder_XGB_v2"
    assert data["processing_time_ms"] > 0
    
    # Check bounding logic mathematically works
    assert data["predicted_ctr"] >= 0.0

def test_bid_endpoint_validation_error():
    payload = {
        "request_id": "req_invalid",
        # Missing floor_price and other required parameters
    }
    with TestClient(app) as client:
        response = client.post("/bid", json=payload)
        assert response.status_code == 422
