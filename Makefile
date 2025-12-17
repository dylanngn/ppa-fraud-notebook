# Makefile for Hybrid GNN-XGBoost Fraud Detection Project

PYTHON := python
TRAINER := src.training.trainer

# Default train/test split dates
# Note: SEON data starts 2024-11-16, use 2024-12-01 as safe start
TRAIN_START := 2024-12-01
TRAIN_END := 2025-06-01
TEST_END := 2025-07-01

.PHONY: help install etl train-vanilla train-gnn compare-all hpo hpo-quick expanding-vanilla expanding-gnn mlflow clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Data Pipeline:"
	@echo "    make etl                   - Run ETL pipeline (CSV → Parquet)"
	@echo ""
	@echo "  Training (single split):"
	@echo "    make train-vanilla         - Train Baseline XGBoost"
	@echo "    make train-gnn             - Train XGBoost + GraphSAGE Embeddings"
	@echo ""
	@echo "  Expanding Window (concept drift / RQ3):"
	@echo "    make expanding-vanilla     - Expanding window with Vanilla XGBoost"
	@echo "    make expanding-gnn         - Expanding window with GNN+XGBoost"
	@echo ""
	@echo "  Comparison (all variants on same test set):"
	@echo "    make compare-all           - Train both variants for A/B comparison"
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
	@echo "    make train-vanilla TRAIN_START=2024-12-01 TRAIN_END=2025-03-01 TEST_END=2025-04-01"

install:
	pip install -r requirements.txt

# --- Training Commands (Single Model) ---

train-vanilla:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Vanilla_XGBoost" \
		model.variant="vanilla_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-gnn:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_GNN_XGBoost" \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=50 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# --- A/B Comparison (all variants, same test set) ---

compare-all:
	@echo "=== Training all variants for A/B comparison ==="
	@echo "Train: $(TRAIN_START) → $(TRAIN_END), Test end: $(TEST_END)"
	@echo ""
	@echo "--- Vanilla XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="AB_Comparison" \
		model.variant="vanilla_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- GNN + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="AB_Comparison" \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=50 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "=== Comparison complete! Check MLflow for results ==="

# --- Hyperparameter Optimization ---

hpo:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		experiment.name="HPO_XGBoost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

hpo-quick:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		hydra.sweeper.n_trials=10 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# HPO with custom trials count: make hpo-custom TRIALS=100
hpo-custom:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		hydra.sweeper.n_trials=$(TRIALS) \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# --- Expanding Window (Concept Drift / RQ3) ---

# Default: 30-day windows, end at TEST_END
WINDOW_DAYS := 30

expanding-vanilla:
	@echo "=== Expanding Window Training: Vanilla XGBoost ==="
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Expanding_Window" \
		model.variant="vanilla_xgboost" \
		training.mode="expanding" \
		data.train_start_date=$(TRAIN_START) \
		data.test_end_date=$(TEST_END) \
		expanding_window.window_days=$(WINDOW_DAYS) \
		expanding_window.min_train_windows=2

expanding-gnn:
	@echo "=== Expanding Window Training: GNN + XGBoost ==="
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Expanding_Window" \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=50 \
		training.mode="expanding" \
		data.train_start_date=$(TRAIN_START) \
		data.test_end_date=$(TEST_END) \
		expanding_window.window_days=$(WINDOW_DAYS) \
		expanding_window.min_train_windows=2

# --- Temporal Robustness Check ---

temporal-check:
	@echo "=== Temporal Robustness Check ==="
	@echo "Training vanilla_xgboost across 3 time periods"
	@echo ""
	@echo "--- Period 1: Early (2024-12 → 2025-03, test → 2025-04) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Temporal_Check" \
		model.variant="vanilla_xgboost" \
		data.train_start_date="2024-12-01" \
		data.train_end_date="2025-03-01" \
		data.test_end_date="2025-04-01"
	@echo ""
	@echo "--- Period 2: Mid (2024-12 → 2025-06, test → 2025-07) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Temporal_Check" \
		model.variant="vanilla_xgboost" \
		data.train_start_date="2024-12-01" \
		data.train_end_date="2025-06-01" \
		data.test_end_date="2025-07-01"
	@echo ""
	@echo "--- Period 3: Recent (2024-12 → 2025-09, test → 2025-10) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Temporal_Check" \
		model.variant="vanilla_xgboost" \
		data.train_start_date="2024-12-01" \
		data.train_end_date="2025-09-01" \
		data.test_end_date="2025-10-01"
	@echo ""
	@echo "=== Temporal check complete! Compare AUC-PR across periods ==="

# --- Data Pipeline (ETL) ---

etl:
	$(PYTHON) -m src.data.etl

# --- Utilities ---

mlflow:
	mlflow ui --backend-store-uri sqlite:///ppa-fraud-detection-mlflow.db

clean:
	rm -rf tmp/
	rm -rf outputs/
	rm -rf multirun/
	find . -type d -name "__pycache__" -exec rm -rf {} +
