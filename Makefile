.PHONY: install etl build-graph train-baseline train-graph-baseline train-gnn train-gnn-rte train-hybrid graph-features advanced-features time-weighted-features interaction-features all-features check-timestamps check-graph check-density debug all seon compare-all api-start api-demo api-test train-baseline-expanding train-hgt-expanding train-gat-expanding train-sage-expanding expanding-all retrain-production optimize-window optimize-hyperparams staged-hyperopt analyze-adaptation adaptation-report train-mlflow mlflow-ui mlflow-compare

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

# --- Experiments ---

train-gat:
	python src/cli.py train-embeddings --model gat
	python src/cli.py train-hybrid --model gat

train-sage:
	python src/cli.py train-embeddings --model sage
	python src/cli.py train-hybrid --model sage

train-hgt:
	python src/cli.py train-embeddings --model hgt
	python src/cli.py train-hybrid --model hgt

train-hgt-rte:
	python src/cli.py train-embeddings --model hgt_rte
	python src/cli.py train-hybrid --model hgt_rte

# -------------------

check-timestamps:
	python src/cli.py check-timestamps

check-graph:
	python src/cli.py check-graph-timestamps

check-density:
	python src/cli.py check-density

debug:
	python src/cli.py debug-polars

# Run complete pipeline: ETL → Build Graph → Graph Features → Advanced Features → All Training Scripts
all: etl build-graph graph-features advanced-features train-baseline train-graph-baseline train-gat train-sage train-hgt train-hgt-rte

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

# --- Expanding Window Experiments (Production-Realistic) ---

train-baseline-expanding:
	@echo "Training Baseline with expanding window (accumulating data)..."
	python src/cli.py train-baseline-expanding --window-days 180 --step-days 7

train-hgt-expanding:
	@echo "Training HGT with expanding window (complete graph)..."
	python src/cli.py train-hybrid-expanding --model hgt --window-days 180 --step-days 7

train-gat-expanding:
	@echo "Training GAT with expanding window (complete graph)..."
	python src/cli.py train-hybrid-expanding --model gat --window-days 180 --step-days 7

train-sage-expanding:
	@echo "Training SAGE with expanding window (complete graph)..."
	python src/cli.py train-hybrid-expanding --model sage --window-days 180 --step-days 7

# Run all expanding window experiments (with prerequisites)
expanding-all: etl build-graph graph-features advanced-features train-baseline-expanding train-hgt-expanding train-gat-expanding train-sage-expanding

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
	python src/cli.py evaluate-seon --window-days 90 --step-days 7

compare-all:
	@echo "Comparing all models (XGBoost vs Hybrid vs Seon)..."
	python src/cli.py compare-all-models

# --- Production API ---

api-start:
	@echo "Starting Fraud Detection API on http://localhost:8000"
	uvicorn src.api.main:app --reload --port 8000

api-demo:
	@echo "Running API demo script..."
	python scripts/demo_api.py

api-test:
	@echo "Running API tests..."
	pytest tests/features/test_temporal_validation.py -v

# --- Adaptation Analysis ---

analyze-adaptation:
	@echo "Running adaptation analysis on baseline model..."
	python src/cli.py analyze-adaptation --model-type baseline

analyze-adaptation-all:
	@echo "Running adaptation analysis on all windows..."
	python src/cli.py analyze-adaptation --model-type baseline --all-windows

adaptation-report:
	@echo "Generating adaptation summary report..."
	python src/cli.py generate-adaptation-report --model-type baseline --n-windows 15

# --- MLflow ---

train-mlflow:
	@echo "Training with MLflow tracking..."
	python src/cli.py train-mlflow --model-type baseline_graph

train-mlflow-register:
	@echo "Training with MLflow and registering model..."
	python src/cli.py train-mlflow --model-type baseline_graph --register-model

mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5000..."
	python src/cli.py mlflow-ui

mlflow-compare:
	@echo "Comparing MLflow runs..."
	python src/cli.py mlflow-compare --top-n 10

# --- Continuous Pipeline ---



# --- Full Continuous Pipeline Setup ---
