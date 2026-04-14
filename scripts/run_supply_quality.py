"""
Standalone script to run the supply quality scoring pipeline.

Usage:
    python scripts/run_supply_quality.py

Reads the raw auction data (to preserve fraud_flag), scores publishers
and exchanges, saves CSVs and charts to reports/supply_quality/.
"""

import os
import sys
import logging

# Ensure project root is on path when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.config.settings import settings
from src.supply_quality.scorer import run_scoring
from src.supply_quality.report import generate_full_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Load the RAW data so fraud_flag is intact for fraud_rate calculation
    raw_path = os.path.join(settings.RAW_DATA_PATH, "ad_auction_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Run `python -m src.data.generate_synthetic_data` first."
        )

    logger.info(f"Loading raw auction data from {raw_path}")
    df = pd.read_csv(raw_path)

    pub_scores, exch_scores = run_scoring(df)

    logger.info("\n=== Top 10 Publishers by Quality Score ===")
    logger.info(
        pub_scores[["entity_id", "impressions", "ctr", "fraud_rate",
                    "avg_roi", "quality_score", "quality_tier"]]
        .head(10)
        .to_string(index=False)
    )

    logger.info("\n=== Exchange Quality Scores ===")
    logger.info(
        exch_scores[["entity_id", "impressions", "ctr", "fraud_rate",
                     "quality_score", "quality_tier"]]
        .to_string(index=False)
    )

    report_paths = generate_full_report(pub_scores, exch_scores)

    logger.info("\nGenerated output files:")
    for label, path in report_paths.items():
        logger.info(f"  {label:25s} → {path}")


if __name__ == "__main__":
    main()
