.PHONY: install etl build-graph train-baseline train-graph-baseline train-gnn train-gnn-rte train-hybrid graph-features advanced-features time-weighted-features interaction-features all-features check-timestamps check-graph check-density debug all seon compare-all api-start api-demo api-test retrain-production optimize-window optimize-hyperparams staged-hyperopt mlflow-ui mlflow-compare mlflow-compare-models mlflow-deployment-recommendation mlflow-drift-summary

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

graph-features:
	python src/cli.py graph-features

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

# Run complete pipeline: ETL → Build Graph → All Features → All Training Scripts
# Note: Training scripts run sequentially to avoid MLflow conflicts and GPU/memory contention
all: etl build-graph all-features train-baseline train-sage train-hgt

# --- Advanced Features ---

advanced-features:
	@echo "Generating advanced graph features..."
	python src/cli.py advanced-graph-features

time-weighted-features:
	@echo "Generating time-weighted graph features (Experiment 9)..."
	python src/cli.py time-weighted-features

interaction-features:
	@echo "Generating interaction features (Experiment 10)..."
	python src/cli.py interaction-features

# Generate all feature types
all-features: graph-features advanced-features time-weighted-features interaction-features

# --- Production Continuous Learning ---

retrain-production:
	@echo "Retraining production model with all historical + production data..."
	python src/cli.py retrain-production

# --- Baseline Optimization Experiments ---

optimize-window:
	@echo "Testing graph feature window sizes (Experiment 8)..."
	python src/cli.py optimize-graph-window

optimize-hyperparams:
	@echo "Running hyperparameter optimization (Experiment 11)..."
	python src/cli.py optimize-hyperparams --n-trials 100 --n-windows 5

staged-hyperopt:
	@echo "Running staged hyperparameter optimization (Experiment 11b)..."
	python src/cli.py staged-hyperopt --n-windows 5

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

# --- MLflow ---
# Note: All training commands (train-baseline, train-sage, train-hgt) 
# automatically use MLflow tracking and register models to the Model Registry.

mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5000..."
	python src/cli.py mlflow-ui

mlflow-compare:
	@echo "Comparing MLflow runs..."
	python src/cli.py mlflow-compare --top-n 10