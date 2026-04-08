import pandas as pd
import os
import logging
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess() -> pd.DataFrame:
    """
    Reads raw data, applies basic feature engineering, filters fraud traffic, 
    and saves output to processed storage.
    """
    raw_path = os.path.join(settings.RAW_DATA_PATH, "ad_auction_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run generate_synthetic_data.py first.")
    
    logger.info(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, parse_dates=['timestamp'])
    
    # Basic Feature Engineering
    logger.info("Applying basic feature engineering (temporal and target variables)...")
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Filter out fraud traffic before feeding into ML models downstream
    fraud_count = df['fraud_flag'].sum()
    logger.info(f"Filtering out {fraud_count} anomalous/fraudulent inventory sources.")
    df_clean = df[df['fraud_flag'] == 0].copy()
    
    # Target value metric (Useful for ROI/Bid calculations later)
    # Avoid zero division
    df_clean['roi'] = (df_clean['revenue'] - df_clean['cost']) / (df_clean['cost'] + 1e-5)
    
    # Optional: fill any missing ad identifiers with a default category if any exist
    df_clean.fillna({'ad_id': 'unknown', 'campaign_id': 'unknown'}, inplace=True)

    # Ensure processed directory exists and save
    os.makedirs(settings.PROCESSED_DATA_PATH, exist_ok=True)
    processed_path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Processed dataset saved to {processed_path} containing {len(df_clean)} rows.")
    
    return df_clean

if __name__ == "__main__":
    load_and_preprocess()
