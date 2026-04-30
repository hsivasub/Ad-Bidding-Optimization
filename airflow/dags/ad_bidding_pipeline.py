from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

# Get the project root directory
# Assuming AIRFLOW_HOME is set, or we default to the current working directory structure
# A robust way is to define an environment variable or simply assume the bash scripts are 
# run from the root of the Ad Bidding Optimization directory.
# For this DAG, we'll configure bash commands to change directory to the project root first,
# or we can rely on Airflow being run from the project root.
PROJECT_ROOT = os.environ.get("AD_BIDDING_PROJECT_ROOT", "/opt/airflow/project") 
# We'll use a dynamic approach by running bash commands. To make it work regardless 
# of where Airflow starts, we can CD to the expected directory or just run python modules.
# Since we know the user is running this locally from their project root:
# we will just use standard python -m commands.

default_args = {
    "owner": "ad_optimization_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ad_bidding_pipeline",
    default_args=default_args,
    description="End-to-End Ad Bidding Optimization Pipeline",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ad-tech", "ml", "simulation"],
) as dag:

    # 1. Generate Synthetic Data
    generate_data = BashOperator(
        task_id="generate_data",
        bash_command="python -m src.data.generate_synthetic_data",
    )

    # 2. Preprocess Data
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="python -m src.data.preprocess",
    )

    # 3. Train CTR Model
    train_ctr_model = BashOperator(
        task_id="train_ctr_model",
        bash_command="python -m src.models.train_improved_ctr",
    )

    # 4. Supply Quality Scoring
    score_supply_quality = BashOperator(
        task_id="score_supply_quality",
        bash_command="python scripts/run_supply_quality.py",
    )

    # 5. Bidding Simulation
    run_simulation = BashOperator(
        task_id="run_simulation",
        bash_command="python scripts/run_bid_simulation.py",
    )

    # 6. A/B Testing
    run_ab_experiment = BashOperator(
        task_id="run_ab_experiment",
        bash_command="python scripts/run_ab_experiment.py",
    )

    # Define DAG dependencies
    generate_data >> preprocess_data >> train_ctr_model
    train_ctr_model >> score_supply_quality
    train_ctr_model >> run_simulation
    [score_supply_quality, run_simulation] >> run_ab_experiment
