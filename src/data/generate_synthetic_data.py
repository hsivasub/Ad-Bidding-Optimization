import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import logging
import os
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_ad_data(num_samples=10000) -> pd.DataFrame:
    """
    Generates a realistic synthetic ad auction dataset with underlying correlations 
    between devices, publishers, time of day, and CTR/Conversion events.
    """
    logger.info(f"Generating {num_samples} synthetic ad auction records...")
    
    np.random.seed(42)
    
    # Base dimensional mapping
    publishers = [f"pub_{i}" for i in range(1, 21)]  # 20 publishers
    exchanges = ["ex_appnexus", "ex_google", "ex_rubicon", "ex_pubmatic"]
    devices = ["mobile", "desktop", "tablet"]
    oses = ["ios", "android", "windows", "macos"]
    countries = ["US", "UK", "CA", "IN", "DE"]
    campaigns = [f"cmp_{i}" for i in range(1, 11)] # 10 campaigns
    ads = [f"ad_{i}" for i in range(1, 51)] # 50 ads

    # Generate dates over the last 14 days
    end_date = datetime.now()
    dates = [end_date - timedelta(days=np.random.randint(0, 14), 
                                 minutes=np.random.randint(0, 24*60)) 
             for _ in range(num_samples)]
    
    df = pd.DataFrame({
        "request_id": [str(uuid.uuid4()) for _ in range(num_samples)],
        "timestamp": dates,
        "user_id": [f"u_{np.random.randint(1000, 9999)}" for _ in range(num_samples)],
        "ad_id": np.random.choice(ads, num_samples),
        "campaign_id": np.random.choice(campaigns, num_samples),
        "publisher_id": np.random.choice(publishers, num_samples),
        "exchange_id": np.random.choice(exchanges, num_samples),
        "device_type": np.random.choice(devices, num_samples, p=[0.6, 0.3, 0.1]),
        "os": np.random.choice(oses, num_samples),
        "country": np.random.choice(countries, num_samples, p=[0.5, 0.15, 0.1, 0.15, 0.1]),
    })
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Financials (Auctions)
    df['floor_price'] = np.random.uniform(0.1, 3.0, num_samples)
    # Bid is usually above floor
    df['bid_price'] = df['floor_price'] + np.random.uniform(0.1, 2.0, num_samples)
    
    # Introduce CTR correlations for realistic dataset
    base_ctr = 0.05
    df['simulated_ctr_prob'] = base_ctr
    
    # Modifiers based on patterns
    df.loc[df['device_type'] == 'mobile', 'simulated_ctr_prob'] *= 1.2
    df.loc[df['device_type'] == 'desktop', 'simulated_ctr_prob'] *= 0.8
    df.loc[df['country'] == 'US', 'simulated_ctr_prob'] *= 1.2
    df.loc[df['publisher_id'] == 'pub_1', 'simulated_ctr_prob'] *= 1.5 # high quality pub
    df.loc[df['publisher_id'] == 'pub_20', 'simulated_ctr_prob'] *= 0.3 # low quality pub
    df.loc[df['hour_of_day'].between(18, 22), 'simulated_ctr_prob'] *= 1.3 # prime time
    
    df['simulated_ctr_prob'] = df['simulated_ctr_prob'].clip(0, 1)
    
    # Simulate actual click
    df['actual_click'] = (np.random.rand(num_samples) < df['simulated_ctr_prob']).astype(int)
    
    # Conversions only happen if there's a click. ~15% conversion rate on clicks
    df['conversion'] = (df['actual_click'] & (np.random.rand(num_samples) < 0.15)).astype(int)
    
    # Cost = what we pay on impression
    df['cost'] = df['bid_price'] * np.random.uniform(0.7, 0.95, num_samples)
    df.loc[df['actual_click'] == 0, 'cost'] = df['bid_price'] * 0.1 # basic impression cost
    
    # Revenue = Payout on conversion
    df['revenue'] = df['conversion'] * np.random.uniform(20.0, 50.0, num_samples)
    
    # Fraud flag (3% base chance, higher on specific exchanges/publishers to simulate bad traffic)
    fraud_prob = 0.03 * np.ones(num_samples)
    fraud_prob[df['exchange_id'] == 'ex_rubicon'] *= 2.0
    fraud_prob[df['publisher_id'] == 'pub_20'] *= 5.0
    df['fraud_flag'] = (np.random.rand(num_samples) < fraud_prob).astype(int)
    
    # Cleanup pseudo-prob column used for generation
    df = df.drop(columns=['simulated_ctr_prob'])
    
    return df

def main():
    os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
    df = generate_ad_data(10000)
    output_file = os.path.join(settings.RAW_DATA_PATH, "ad_auction_data.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Successfully saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    main()
