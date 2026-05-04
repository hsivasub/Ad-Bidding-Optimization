"""
Phase 12: Airflow DAG for Automated Ad Bidding Optimization Pipeline

This DAG orchestrates the entire ML lifecycle:
1. Synthetic Data Generation (Simulation of daily ingestion)
2. Preprocessing & Fraud Filtering
3. Supply Quality Entity Scoring
4. Model Training & MLflow Artifact Logging
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Because the DAG might run from Airflow's home directory, we specify the project root
# In a real production system, this would be an absolute path or passed via Airflow Variables.
PROJECT_ROOT = os.environ.get("AD_BIDDING_HOME", "/opt/airflow/dags/ad_bidding")

default_args = {
    "owner": "ml_engineering_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ad_bidding_optimization_pipeline",
    default_args=default_args,
    description="End-to-End CTR Model Training & Publisher Scoring Pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ad-tech", "ml-pipeline"],
) as dag:

    # Task 1: Data Ingestion Simulation
    ingest_data = BashOperator(
        task_id="ingest_daily_auction_data",
        bash_command=f"cd {PROJECT_ROOT} && python src/data/generate_synthetic_data.py",
    )

    # Task 2: Preprocess & Filter Fraud
    preprocess_data = BashOperator(
        task_id="preprocess_and_filter_fraud",
        bash_command=f"cd {PROJECT_ROOT} && python src/data/preprocess.py",
    )

    # Task 3: Score Publisher Quality (Independent path, but relies on processed data)
    score_publishers = BashOperator(
        task_id="update_publisher_supply_quality",
        bash_command=f"cd {PROJECT_ROOT} && python scripts/run_supply_quality.py",
    )

    # Task 4: Train Improved CTR Model with MLflow
    train_ctr_model = BashOperator(
        task_id="train_ctr_model_mlflow",
        bash_command=f"cd {PROJECT_ROOT} && python -m src.models.train_improved_ctr",
    )

    # Define DAG structure/dependencies
    ingest_data >> preprocess_data
    preprocess_data >> score_publishers
    preprocess_data >> train_ctr_model
