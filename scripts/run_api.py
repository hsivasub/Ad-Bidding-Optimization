"""
Entrypoint to run the FastAPI inference service via Uvicorn.

Usage:
    python scripts/run_api.py
"""

import uvicorn
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
