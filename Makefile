.PHONY: setup install test run-api run-dashboard run-airflow setup-airflow clean

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

setup-airflow:
	set AIRFLOW_HOME=%cd%\airflow
	airflow db init
	airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

run-airflow:
	set AIRFLOW_HOME=%cd%\airflow
	start airflow webserver -p 8080
	start airflow scheduler

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
