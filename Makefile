# Makefile for Hybrid GNN-XGBoost Fraud Detection Project

PYTHON := python
TRAINER := src.training.trainer

# Default train/test split dates
TRAIN_END := 2024-06-01
TEST_END := 2024-09-01

.PHONY: help install train-vanilla train-handcrafted train-gnn compare-all hpo hpo-quick mlflow clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Training (single model):"
	@echo "    make train-vanilla         - Train Baseline XGBoost"
	@echo "    make train-handcrafted     - Train XGBoost + Handcrafted Graph Features"
	@echo "    make train-gnn             - Train XGBoost + GraphSAGE Embeddings"
	@echo ""
	@echo "  Comparison (all variants on same test set):"
	@echo "    make compare-all           - Train all 3 variants for A/B comparison"
	@echo ""
	@echo "  Hyperparameter Optimization:"
	@echo "    make hpo                   - Run XGBoost HPO (50 trials)"
	@echo "    make hpo-quick             - Run XGBoost HPO quick test (10 trials)"
	@echo ""
	@echo "  Utilities:"
	@echo "    make mlflow                - Start MLflow UI"
	@echo "    make clean                 - Remove artifacts"
	@echo ""
	@echo "  Custom dates:"
	@echo "    make train-vanilla TRAIN_END=2024-03-01 TEST_END=2024-06-01"

install:
	pip install -r requirements.txt

# --- Training Commands (Single Model) ---

train-vanilla:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Vanilla_XGBoost" \
		model.variant="vanilla_xgboost" \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-handcrafted:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Handcrafted_XGBoost" \
		model.variant="handcrafted_xgboost" \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-gnn:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Hybrid_GraphSAGE" \
		model.variant="graphsage_xgboost" \
		model.gnn.epochs=100 \
		model.gnn.device="mps" \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# --- A/B Comparison (all variants, same test set) ---

compare-all:
	@echo "=== Training all variants for A/B comparison ==="
	@echo "Train end: $(TRAIN_END), Test end: $(TEST_END)"
	@echo ""
	@echo "--- Vanilla XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="AB_Comparison" \
		model.variant="vanilla_xgboost" \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- Handcrafted XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="AB_Comparison" \
		model.variant="handcrafted_xgboost" \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- GraphSAGE + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="AB_Comparison" \
		model.variant="graphsage_xgboost" \
		model.gnn.epochs=100 \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "=== Comparison complete! Check MLflow for results ==="

# --- Hyperparameter Optimization ---

hpo:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

hpo-quick:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		hydra.sweeper.n_trials=10 \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# HPO with custom trials count: make hpo-custom TRIALS=100
hpo-custom:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		hydra.sweeper.n_trials=$(TRIALS) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# --- Temporal Robustness Check ---
# Run same model across different time periods to verify stability

temporal-check:
	@echo "=== Temporal Robustness Check ==="
	@echo "Training vanilla_xgboost across 3 time periods"
	@echo ""
	@echo "--- Period 1: Early (train→2024-03-01, test→2024-06-01) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Temporal_Check" \
		model.variant="vanilla_xgboost" \
		data.train_end_date="2024-03-01" \
		data.test_end_date="2024-06-01"
	@echo ""
	@echo "--- Period 2: Mid (train→2024-06-01, test→2024-09-01) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Temporal_Check" \
		model.variant="vanilla_xgboost" \
		data.train_end_date="2024-06-01" \
		data.test_end_date="2024-09-01"
	@echo ""
	@echo "--- Period 3: Recent (train→2024-09-01, test→2024-12-01) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Temporal_Check" \
		model.variant="vanilla_xgboost" \
		data.train_end_date="2024-09-01" \
		data.test_end_date="2024-12-01"
	@echo ""
	@echo "=== Temporal check complete! Compare AUC-PR across periods ==="

# --- Utilities ---

mlflow:
	mlflow ui --backend-store-uri sqlite:///ppa-fraud-detection-mlflow.db

clean:
	rm -rf tmp/
	rm -rf outputs/
	rm -rf multirun/
	find . -type d -name "__pycache__" -exec rm -rf {} +
