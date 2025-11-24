.PHONY: install etl build-graph train-baseline train-gnn train-gnn-rte train-hybrid graph-features check-timestamps check-graph check-density debug all

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

# Run complete pipeline: ETL → Build Graph → Graph Features → All Training Scripts
all: etl build-graph graph-features train-baseline train-graph-baseline train-gat train-sage train-hgt train-hgt-rte

# --- Seon Evaluation ---

seon:
	@echo "Evaluating Seon (production baseline)..."
	python src/cli.py evaluate-seon --window-days 90 --step-days 7

compare-all:
	@echo "Comparing all models (XGBoost vs Hybrid vs Seon)..."
	python src/cli.py compare-all-models
