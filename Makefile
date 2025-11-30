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
	python -m src.data.etl

build-graph:
	python -m src.data.create_graph_artifacts

# Model Training
train-baseline:
	python -m src.models.train_baseline

train-sage:
	python -m src.models.train_hybrid_sage

train-hgt:
	python -m src.models.train_hybrid_hgt

# Hyperparameter Optimization
optimize-xgboost:
	python -m src.models.hyperopt_xgboost

optimize-pytorch:
	@echo "Usage: make optimize-pytorch MODEL_TYPE=hgt|sage"
	@if [ -z "$(MODEL_TYPE)" ]; then \
		echo "Error: MODEL_TYPE is required (hgt or sage)"; \
		exit 1; \
	fi
	python -m src.models.hyperopt_pytorch --model-type $(MODEL_TYPE)

# Utilities
check-density:
	python -m src.utils.check_window_density

data-quality-report:
	python -m src.utils.data_quality_report

evaluate-seon:
	python -m src.utils.evaluate_seon

# MLflow Management
mlflow-compare-models:
	@echo "Usage: make mlflow-compare-models MODEL_NAME=... CANDIDATE_RUN_ID=..."
	@if [ -z "$(MODEL_NAME)" ] || [ -z "$(CANDIDATE_RUN_ID)" ]; then \
		echo "Error: MODEL_NAME and CANDIDATE_RUN_ID are required"; \
		exit 1; \
	fi
	python -m src.utils.mlflow_model_comparison compare --model-name $(MODEL_NAME) --candidate-run-id $(CANDIDATE_RUN_ID)

mlflow-deployment-recommendation:
	@echo "Usage: make mlflow-deployment-recommendation MODEL_NAME=... CANDIDATE_RUN_ID=..."
	@if [ -z "$(MODEL_NAME)" ] || [ -z "$(CANDIDATE_RUN_ID)" ]; then \
		echo "Error: MODEL_NAME and CANDIDATE_RUN_ID are required"; \
		exit 1; \
	fi
	python -m src.utils.mlflow_model_comparison recommend --model-name $(MODEL_NAME) --candidate-run-id $(CANDIDATE_RUN_ID)

mlflow-drift-summary:
	@echo "Usage: make mlflow-drift-summary MODEL_NAME=..."
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "Error: MODEL_NAME is required"; \
		exit 1; \
	fi
	python -m src.utils.mlflow_model_comparison drift --model-name $(MODEL_NAME)
