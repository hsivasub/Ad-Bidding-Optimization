"""
Phase 13: Model Drift and Data Quality Monitoring using Evidently

Extracts chronological splits from the dataset to simulate "Reference" (Training) 
vs "Current" (Production Inference) data distributions.
Generates interactive HTML drift reports tracing distribution shifts in features.
"""

import os
import sys
import logging
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_drift_report():
    data_path = os.path.join(settings.PROCESSED_DATA_PATH, "ad_auction_processed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at {data_path}")
        
    logger.info("Loading processed data for drift evaluation...")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    
    # Sort chronologically to simulate a time-series deployment timeline
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Split: first 70% as reference (training data), last 30% as current (production data)
    split_idx = int(len(df) * 0.7)
    reference_data = df.iloc[:split_idx].copy()
    current_data = df.iloc[split_idx:].copy()
    
    logger.info(f"Reference Data (Training): {len(reference_data)} rows")
    logger.info(f"Current Data (Production): {len(current_data)} rows")
    
    # Setup Evidently Report with Presets
    logger.info("Generating Evidently Drift & Data Quality Report (Auto-Infer Mapping)...")
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    snapshot = drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Save Report
    output_dir = os.path.join(settings.BASE_DIR, "reports", "monitoring")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "drift_report.html")
    
    snapshot.save_html(report_path)
    logger.info(f"Drift report generated successfully: {report_path}")

if __name__ == "__main__":
    generate_drift_report()
