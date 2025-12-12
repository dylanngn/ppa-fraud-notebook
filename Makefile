
# Makefile for Hybrid GNN-XGBoost Fraud Detection Project

PYTHON := python
TRAINER := src.training.trainer
DATA_START := 2024-01-01
DATA_END := 2025-11-02

.PHONY: help install train-vanilla train-handcrafted train-gnn mlflow clean

help:
	@echo "Available commands:"
	@echo "  make install               - Install dependencies"
	@echo "  make train-vanilla         - Train Baseline XGBoost"
	@echo "  make train-handcrafted     - Train XGBoost + Handcrafted Graph Features"
	@echo "  make train-gnn             - Train XGBoost + GraphSAGE Embeddings (Hybrid)"
	@echo "  make mlflow                - Start MLflow UI"
	@echo "  make clean                 - Remove artifacts"

install:
	pip install -r requirements.txt

# --- Training Commands ---

train-vanilla:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Vanilla_XGBoost" \
		model.variant="vanilla_xgboost" \
		training.initial_train_months=12 \
		training.accumulation_frequency="weekly" \
		training.prediction_window_days=30 \
		data.start_date=$(DATA_START) \
		data.end_date=$(DATA_END)

train-handcrafted:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Handcrafted_XGBoost" \
		model.variant="handcrafted_xgboost" \
		data.start_date=$(DATA_START) \
		data.end_date=$(DATA_END)

train-gnn:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Hybrid_GraphSAGE" \
		model.variant="graphsage_xgboost" \
		model.gnn.epochs=100 \
		model.gnn.device="mps" \
		data.start_date=$(DATA_START) \
		data.end_date=$(DATA_END)

# --- Utilities ---

mlflow:
	mlflow ui --backend-store-uri sqlite:///ppa-fraud-detection-mlflow.db

clean:
	rm -rf tmp/
	rm -rf outputs/
	rm -rf multirun/
	rm -rf mlruns/
	find . -type d -name "__pycache__" -exec rm -rf {} +
