# Ad Bidding Optimization & Experimentation System

An end-to-end ads analytics and machine learning system that simulates a real ad-tech optimization workflow.

## Business Problem
Ad-tech systems rely on predicting ad clicks (CTR) and conversions to optimize bidding strategies across multiple publishers and exchanges. This project simulates an end-to-end flow to ingest data, train predictive models, score supply quality, run bidding strategies, and evaluate them through A/B testing frameworks.

## Project Structure
- **data/**: Directory for raw, processed, and synthetic datasets.
- **src/data**: Ingestion and preprocessing pipelines.
- **src/features**: Feature engineering workflows.
- **src/models**: Training and prediction logic.
- **src/bidding**: Auction simulation and bid strategies.
- **src/experimentation**: A/B testing and lift analysis.
- **src/api**: FastAPI inference endpoints.
- **dashboard**: Streamlit application.
- **airflow**: Airflow DAGs.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Unix/Git Bash:
   source venv/Scripts/activate
   ```
2. Install dependencies:
   ```bash
   make install
   # Or using pip directly: pip install -r requirements.txt
   ```


