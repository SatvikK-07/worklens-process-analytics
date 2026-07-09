MPLCONFIGDIR ?= /tmp/worklens-matplotlib
LOKY_MAX_CPU_COUNT ?= 1
export MPLCONFIGDIR
export LOKY_MAX_CPU_COUNT

.PHONY: setup dependency-audit verify check-hygiene data generate-data generate-sample-data sample-data external-data run-real-eventlog-sample run-real-eventlog-full db train train-sample run run-streamlit api run-api test coverage lint reports clean demo all

setup:
	python -m pip install -r requirements.txt -r requirements-dev.txt

dependency-audit:
	python -m pip check

verify: dependency-audit
	python scripts/verify_clean_setup.py

check-hygiene:
	python scripts/check_repo_hygiene.py

data:
	python -m src.data_generation.generate_claims_event_log

generate-data: data

generate-sample-data:
	python -m src.data_generation.generate_claims_event_log --cases 250 --output-dir data/sample/synthetic

sample-data: generate-sample-data

external-data:
	python scripts/download_external_log.py
	python scripts/validate_external_log.py

run-real-eventlog-sample:
	python real_eventlog_experiments/scripts/run_all_real_experiments.py --sample

run-real-eventlog-full:
	python real_eventlog_experiments/scripts/run_all_real_experiments.py --input data/real/raw/sepsis_cases.xes.gz

db:
	python -m src.ingestion.load_to_db

train:
	python -m src.ml.train_all

train-sample: generate-sample-data
	WORKLENS_DATA_DIR=data/sample/synthetic WORKLENS_MODEL_DIR=models/sample python -m src.ml.train_all

run:
	python -m streamlit run app/main.py

run-streamlit: run

api:
	python -m uvicorn backend.main:app --reload --port 8000

run-api: api

test:
	python -m pytest --cov=src --cov=backend --cov=real_eventlog_experiments --cov-report=term-missing -q

coverage: test

lint:
	python -m ruff check .
	python -m ruff format --check .

reports:
	python scripts/generate_model_reports.py

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name .ipynb_checkpoints -o -name __MACOSX \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' -o -name '*.db' -o -name '*.db-shm' -o -name '*.db-wal' \) -delete
	find data/synthetic -maxdepth 1 -type f -name '*.csv' -delete
	find models -type f \( -name '*.pkl' -o -name '*.joblib' \) -delete
	find models/sample -type f ! -name '.gitkeep' -delete
	find data/real/raw -type f ! -name '.gitkeep' -delete
	find . -type f \( -name '*.tmp' -o -name '*.bak' \) -delete
	rm -f reports/figures/roc_curve.png reports/figures/pr_curve.png reports/figures/calibration_curve.png reports/figures/confusion_matrix.png reports/figures/feature_importance.png reports/figures/regression_residuals.png reports/figures/error_by_bucket.png
	rm -f .coverage

demo: data db train test

all: check-hygiene verify lint demo coverage
