.PHONY: install env etl build-graph train-baseline train-sage train-hgt \
	optimize-xgboost optimize-pytorch check-timestamps check-density \
	data-quality-report evaluate-seon mlflow-compare-models \
	mlflow-deployment-recommendation mlflow-drift-summary help

# Default target
help:
	@echo "Fraud Detection Pipeline - Available Commands:"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make install          - Install dependencies"
	@echo "  make env              - Create virtual environment"
	@echo "  make etl              - Run ETL pipeline"
	@echo "  make build-graph       - Build PyTorch Geometric graph"
	@echo ""
	@echo "Model Training:"
	@echo "  make train-baseline   - Train baseline XGBoost model"
	@echo "  make train-sage       - Train SAGE hybrid model"
	@echo "  make train-hgt        - Train HGT hybrid model"
	@echo ""
	@echo "Hyperparameter Optimization:"
	@echo "  make optimize-xgboost - Optimize XGBoost hyperparameters"
	@echo "  make optimize-pytorch - Optimize PyTorch GNN hyperparameters"
	@echo ""
	@echo "Utilities:"
	@echo "  make check-density    - Check fraud density in time windows"
	@echo "  make data-quality-report - Generate data quality report"
	@echo "  make evaluate-seon    - Evaluate Seon baseline"
	@echo ""
	@echo "MLflow Management:"
	@echo "  make mlflow-compare-models - Compare two MLflow runs"
	@echo "  make mlflow-deployment-recommendation - Get deployment recommendation"
	@echo "  make mlflow-drift-summary - Analyze model drift"

# Installation
install:
	pip install -r requirements.txt

env:
	python3 -m venv .venv
	@echo "Virtual environment created. To activate:"
	@echo "  source .venv/bin/activate"
	.venv/bin/pip install -r requirements.txt

# Data Pipeline
etl:
	python src/cli.py extract-data

build-graph:
	python src/cli.py build-graph

# Model Training
train-baseline:
	python src/cli.py train-baseline

train-sage:
	python src/cli.py train-hybrid-sage

train-hgt:
	python src/cli.py train-hybrid-hgt

# Hyperparameter Optimization
optimize-xgboost:
	python src/cli.py optimize-xgboost

optimize-pytorch:
	@echo "Usage: make optimize-pytorch MODEL_TYPE=hgt|sage"
	@if [ -z "$(MODEL_TYPE)" ]; then \
		echo "Error: MODEL_TYPE is required (hgt or sage)"; \
		exit 1; \
	fi
	python src/cli.py optimize-pytorch --model-type $(MODEL_TYPE)

# Utilities
check-density:
	python src/cli.py check-density

data-quality-report:
	python src/cli.py data-quality-report

evaluate-seon:
	python src/cli.py evaluate-seon
