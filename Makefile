.PHONY: setup install test run-api run-dashboard clean

setup:
	python -m venv venv
	@echo "Run '.\venv\Scripts\Activate.ps1' on Windows PowerShell, then run 'make install'"

install:
	pip install -r requirements.txt

test:
	pytest tests/

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
