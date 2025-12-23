# Makefile for Hybrid GNN-XGBoost Fraud Detection Project

PYTHON := python
TRAINER := src.training.trainer

# Default train/test split dates
# Note: SEON data starts 2024-11-16, use 2024-12-01 as safe start
TRAIN_START := 2024-12-01
TRAIN_END := 2025-06-01
TEST_END := 2025-07-01

.PHONY: help install etl train-lr train-rf train-vanilla train-graph-features train-gnn train-hgt train-care train-vanilla-shap train-gnn-shap compare-all compare-gnn-encoders hpo hpo-quick hpo-high-recall hpo-gnn hpo-gnn-quick hpo-gnn-with-best-xgb hpo-hgt expanding-vanilla expanding-gnn api api-dev mlflow thesis thesis-clean clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Data Pipeline:"
	@echo "    make etl                   - Run ETL pipeline (CSV → Parquet)"
	@echo ""
	@echo "  Training (single split):"
	@echo "    make train-lr              - Train Logistic Regression (baseline)"
	@echo "    make train-rf              - Train Random Forest (baseline)"
	@echo "    make train-vanilla         - Train Vanilla XGBoost"
	@echo "    make train-graph-features  - Train XGBoost + Handcrafted Graph Features"
	@echo "    make train-gnn             - Train XGBoost + GraphSAGE Embeddings"
	@echo "    make train-hgt             - Train XGBoost + HGT Embeddings"
	@echo "    make train-care            - Train XGBoost + CARE-GNN (camouflage-resistant)"
	@echo "    make train-vanilla-shap    - Train Vanilla XGBoost + SHAP analysis"
	@echo "    make train-gnn-shap        - Train GNN+XGBoost + SHAP analysis"
	@echo ""
	@echo "  Expanding Window (concept drift / RQ3):"
	@echo "    make expanding-vanilla     - Expanding window with Vanilla XGBoost"
	@echo "    make expanding-gnn         - Expanding window with GNN+XGBoost"
	@echo ""
	@echo "  Comparison (all variants on same test set):"
	@echo "    make compare-all           - Train all 4 variants (LR, RF, XGB, GNN+XGB)"
	@echo "    make compare-gnn-encoders  - Compare GraphSAGE vs HGT encoders"
	@echo ""
	@echo "  Hyperparameter Optimization:"
	@echo "    make hpo                   - Run XGBoost HPO (50 trials)"
	@echo "    make hpo-quick             - Run XGBoost HPO quick test (10 trials)"
	@echo "    make hpo-high-recall       - HPO optimized for high recall"
	@echo "    make hpo-gnn               - GNN+XGBoost joint HPO (30 trials, slower)"
	@echo "    make hpo-gnn-quick         - GNN+XGBoost HPO quick test (10 trials)"
	@echo "    make hpo-gnn-with-best-xgb - GNN HPO using best XGBoost params (recommended)"
	@echo ""
	@echo "  API Service:"
	@echo "    make api                   - Start Fraud Detection API (port 8000)"
	@echo "    make api-dev               - Start API with hot reload"
	@echo ""
	@echo "  Utilities:"
	@echo "    make mlflow                - Start MLflow UI"
	@echo "    make thesis                - Build LaTeX thesis (requires latexmk)"
	@echo "    make thesis-clean          - Clean LaTeX build artifacts"
	@echo "    make clean                 - Remove code artifacts"
	@echo ""
	@echo "  Custom dates:"
	@echo "    make train-vanilla TRAIN_START=2024-12-01 TRAIN_END=2025-03-01 TEST_END=2025-04-01"

install:
	pip install -r requirements.txt

# --- Training Commands (Single Model) ---

train-lr:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Baselines" \
		model.variant="logistic_regression" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-rf:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Baselines" \
		model.variant="random_forest" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-vanilla:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Vanilla_XGBoost" \
		model.variant="vanilla_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-graph-features:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Graph_Features_XGBoost" \
		model.variant="graph_features_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-gnn:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_GNN_XGBoost" \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="graphsage" \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-hgt:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_HGT_XGBoost" \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="hgt" \
		model.gnn.num_heads=4 \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-care:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_CARE_GNN_XGBoost" \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="care" \
		model.gnn.similarity_dim=32 \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# Compare GraphSAGE vs HGT on same test set
compare-gnn-encoders:
	@echo "=== Comparing GNN Encoders ==="
	@echo "Train: $(TRAIN_START) → $(TRAIN_END), Test end: $(TEST_END)"
	@echo ""
	@echo "--- 1/2: GraphSAGE + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="GNN_Encoder_Comparison" \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="graphsage" \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 2/2: HGT + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="GNN_Encoder_Comparison" \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="hgt" \
		model.gnn.num_heads=4 \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "=== Done! Check MLflow for comparison ==="

# --- Training with SHAP Analysis (RQ4) ---

train-vanilla-shap:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_Vanilla_SHAP" \
		model.variant="vanilla_xgboost" \
		training.run_shap=true \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "SHAP analysis saved to artifacts/shap/"
	@echo "Check MLflow for shap artifacts"

train-gnn-shap:
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Exp_GNN_SHAP" \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=30 \
		training.run_shap=true \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "SHAP analysis saved to artifacts/shap/"
	@echo "Check MLflow for shap artifacts"

# --- A/B Comparison (all variants, same test set) ---

compare-all:
	@echo "=== Training all variants for comparison ==="
	@echo "Train: $(TRAIN_START) → $(TRAIN_END), Test end: $(TEST_END)"
	@echo ""
	@echo "--- 1/4: Logistic Regression (baseline) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Model_Comparison" \
		model.variant="logistic_regression" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 2/4: Random Forest (baseline) ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Model_Comparison" \
		model.variant="random_forest" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 3/4: Vanilla XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Model_Comparison" \
		model.variant="vanilla_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 4/4: GNN + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		experiment.name="Model_Comparison" \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "=== Comparison complete! Check MLflow for results ==="
	@echo "Run 'make mlflow' to view results"

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

# HPO for high recall (aggressive fraud catching)
hpo-high-recall:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_high_recall \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# HPO for GNN + XGBoost (joint tuning)
hpo-gnn:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_gnn_xgboost \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# HPO for GNN + XGBoost (quick, fewer trials)
hpo-gnn-quick:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_gnn_xgboost \
		hydra.sweeper.n_trials=10 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# HPO for GNN only (uses best XGBoost params from previous HPO)
hpo-gnn-with-best-xgb:
	$(PYTHON) scripts/run_gnn_hpo_with_best_xgb.py \
		--trials=20 \
		--train-start=$(TRAIN_START) \
		--train-end=$(TRAIN_END) \
		--test-end=$(TEST_END)

# HPO for HGT (Heterogeneous Graph Transformer) - 20 trials
hpo-hgt:
	@echo "=== HGT Hyperparameter Optimization (20 trials) ==="
	$(PYTHON) -m $(TRAINER) \
		--multirun \
		--config-name=hpo_hgt \
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

# --- API Service ---

api:
	@echo "Starting Fraud Detection API..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

api-dev:
	@echo "Starting Fraud Detection API (dev mode with reload)..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-docs:
	@echo "API documentation available at:"
	@echo "  - Swagger UI: http://localhost:8000/docs"
	@echo "  - ReDoc: http://localhost:8000/redoc"

# --- Utilities ---

mlflow:
	mlflow ui --backend-store-uri sqlite:///ppa-fraud-detection-mlflow.db

# --- Thesis ---

thesis:
	@echo "Compiling thesis..."
	@cd thesis && \
	(pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; true) && \
	(biber main > /dev/null 2>&1; true) && \
	(pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; true) && \
	(pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; true) && \
	echo "✅ Thesis compiled: thesis/main.pdf"

thesis-clean:
	cd thesis && latexmk -c

clean:
	rm -rf tmp/
	rm -rf outputs/
	rm -rf multirun/
	rm -rf artifacts/api/
	find . -type d -name "__pycache__" -exec rm -rf {} +
