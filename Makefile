.PHONY: install etl build-graph train-baseline train-graph-baseline train-gnn train-gnn-rte train-hybrid check-timestamps check-graph check-density data-quality-report debug all seon compare-all api-start api-demo api-test retrain-production optimize-window optimize-hyperparams staged-hyperopt mlflow-ui mlflow-compare mlflow-compare-models mlflow-deployment-recommendation mlflow-drift-summary

install:
	pip install -r requirements.txt

env:
	python3 -m venv .venv
	@echo "Virtual environment created. To activate:"
	@echo "  source .venv/bin/activate"
	.venv/bin/pip install -r requirements.txt

etl:
	python src/cli.py extract-data

build-graph:
	python src/cli.py build-graph

train-baseline:
	python src/cli.py train-baseline

train-graph-baseline:
	python src/cli.py train-graph-baseline

train-sage:
	python src/cli.py train-hybrid-sage

train-hgt:
	python src/cli.py train-hybrid-hgt

# -------------------

check-density:
	python src/cli.py check-density

data-quality-report:
	@echo "Generating data quality report..."
	python src/cli.py data-quality-report

# Run complete pipeline: ETL → Build Graph → All Training Scripts
all: etl build-graph train-baseline train-sage train-hgt

# --- Seon Evaluation ---

seon:
	@echo "Evaluating Seon (production baseline)..."
	python src/cli.py evaluate-seon --evaluation-start-days 90 --step-days 14

# --- MLflow Model Management ---

mlflow-compare-models:
	@echo "Compare two MLflow runs..."
	@echo "Usage: make mlflow-compare-models PROD_RUN=<id> CAND_RUN=<id>"
	python src/cli.py mlflow-compare-models --production-run-id $(PROD_RUN) --candidate-run-id $(CAND_RUN)

mlflow-deployment-recommendation:
	@echo "Get deployment recommendation from Model Registry..."
	@echo "Usage: make mlflow-deployment-recommendation CAND_RUN=<id>"
	python src/cli.py mlflow-deployment-recommendation --candidate-run-id $(CAND_RUN)

mlflow-drift-summary:
	@echo "Analyze model drift from Model Registry..."
	python src/cli.py mlflow-drift-summary

mlflow-compare:
	@echo "Comparing MLflow runs..."
	python src/cli.py mlflow-compare --top-n 10