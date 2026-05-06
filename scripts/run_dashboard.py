"""
Script to launch the Phase 14 Streamlit Dashboard.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.settings import settings

def main():
    dashboard_path = os.path.join(settings.BASE_DIR, "src", "dashboard", "app.py")
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
        
    print(f"Starting Streamlit dashboard at {dashboard_path}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

if __name__ == "__main__":
    main()
