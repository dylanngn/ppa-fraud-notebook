.PHONY: install etl build-graph train-baseline train-graph-baseline train-gnn train-gnn-rte train-hybrid graph-features advanced-features check-timestamps check-graph check-density debug all seon compare-all api-start api-demo api-test train-baseline-expanding train-hgt-expanding train-gat-expanding train-sage-expanding expanding-all retrain-production optimize-window

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
