"""
Global configuration for the Ad Bidding Optimization system.
Config-driven design using Pydantic.
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Ad Bidding Optimization System"
    VERSION: str = "0.1.0"
    
    # Base paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Data paths
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_DATA_PATH: str = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_PATH: str = os.path.join(DATA_DIR, "processed")
    SYNTHETIC_DATA_PATH: str = os.path.join(DATA_DIR, "synthetic")
    
    # Model configuration
    MODEL_DIR: str = os.path.join(BASE_DIR, "mlruns")
    
    # Simulation settings
    DEFAULT_BID: float = 1.5
    
    class Config:
        env_file = ".env"

# Instantiate for global imports
settings = Settings()
