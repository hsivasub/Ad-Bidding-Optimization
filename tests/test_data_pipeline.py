import os
from src.data.generate_synthetic_data import generate_ad_data
from src.config.settings import settings

def test_generate_ad_data():
    """Verify synthetic data generator produces correct shape and columns"""
    df = generate_ad_data(num_samples=100)
    
    # Check length
    assert len(df) == 100
    
    # Check required columns
    expected_cols = [
        'request_id', 'timestamp', 'user_id', 'ad_id', 'campaign_id', 
        'publisher_id', 'exchange_id', 'device_type', 'os', 'country', 
        'hour_of_day', 'day_of_week', 'floor_price', 'bid_price', 
        'actual_click', 'conversion', 'cost', 'revenue', 'fraud_flag'
    ]
    for col in expected_cols:
        assert col in df.columns
        
    # Check logical relationships
    # Conversions should only happen if click occurred
    assert df[df['actual_click'] == 0]['conversion'].sum() == 0
