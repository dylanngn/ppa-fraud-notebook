# Makefile for Hybrid GNN-XGBoost Fraud Detection Project

PYTHON := python
TRAINER := src.training.trainer

# Default train/test split dates
# SEON data range: 2025-03-21 → 2026-03-20
# Events query range: 2025-03-07 → 2026-03-21
# 7-day label-maturation gap between TRAIN_END and TEST_START
TRAIN_START := 2025-03-21
TRAIN_END   := 2025-12-01
TEST_END    := 2026-02-01

.PHONY: help install etl \
	train-lr train-rf train-vanilla train-graph-features \
	train-gnn train-hgt train-care \
	train-vanilla-shap train-gnn-shap \
	compare-all compare-gnn-encoders \
	expanding-vanilla expanding-gnn temporal-check \
	hpo hpo-quick hpo-high-recall \
	hpo-gnn hpo-gnn-quick hpo-gnn-with-best-xgb hpo-hgt \
	register-model api api-dev api-docs \
	nodered nodered-install demo \
	mlflow thesis thesis-clean clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Data Pipeline:"
	@echo "    make etl                   - Run ETL pipeline (CSV → Parquet)"
	@echo "    make csv                   - Export anonymized parquet → CSV (for council review)"
	@echo ""
	@echo "  Training (single split) — runs go to MLflow experiment: fraud-detection"
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
	@echo "  Expanding Window — runs go to experiment: fraud-detection-expanding"
	@echo "    make expanding-vanilla     - Expanding window with Vanilla XGBoost"
	@echo "    make expanding-gnn         - Expanding window with GNN+XGBoost"
	@echo ""
	@echo "  Comparison (all variants on same test set):"
	@echo "    make compare-all           - Train all 4 variants (LR, RF, XGB, GNN+XGB)"
	@echo "    make compare-gnn-encoders  - Compare GraphSAGE vs HGT encoders"
	@echo ""
	@echo "  Hyperparameter Optimization — runs go to experiment: fraud-detection-hpo"
	@echo "    make hpo                   - XGBoost HPO (50 trials)"
	@echo "    make hpo-quick             - XGBoost HPO quick test (10 trials)"
	@echo "    make hpo-high-recall       - HPO optimised for high recall"
	@echo "    make hpo-gnn               - GNN+XGBoost joint HPO (50 trials)"
	@echo "    make hpo-gnn-quick         - GNN+XGBoost HPO quick test (10 trials)"
	@echo "    make hpo-gnn-with-best-xgb - GNN HPO using best XGBoost params (recommended)"
	@echo "    make hpo-hgt               - HGT encoder HPO (20 trials)"
	@echo ""
	@echo "  Model Registry:"
	@echo "    make register-model        - Promote latest run to the 'production' alias"
	@echo "    make register-best-hpo     - Find best HPO run by AUC-PR and promote to 'production'"
	@echo ""
	@echo "  API Service:"
	@echo "    make api                   - Start Fraud Detection API (port 8000)"
	@echo "    make api-dev               - Start API with hot reload"
	@echo ""
	@echo "  Node-RED Demo:"
	@echo "    make nodered-install       - Install Node-RED dependencies"
	@echo "    make nodered               - Start Node-RED dashboard (port 1880)"
	@echo "    make demo                  - Print instructions to run full demo"
	@echo ""
	@echo "  Utilities:"
	@echo "    make mlflow                - Start MLflow UI"
	@echo "    make thesis                - Build LaTeX thesis (requires latexmk)"
	@echo "    make thesis-clean          - Clean LaTeX build artifacts"
	@echo "    make clean                 - Remove code artefacts"
	@echo ""
	@echo "  Custom dates:"
	@echo "    make train-vanilla TRAIN_START=2024-12-01 TRAIN_END=2025-03-01 TEST_END=2025-04-01"

install:
	pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Training Commands (Single Model)
# All runs land in the 'fraud-detection' MLflow experiment.
# Differentiate by tag: mlflow ui → filter by 'model.variant' tag.
# ---------------------------------------------------------------------------

train-lr:
	$(PYTHON) -m $(TRAINER) \
		model.variant="logistic_regression" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-rf:
	$(PYTHON) -m $(TRAINER) \
		model.variant="random_forest" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-vanilla:
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-graph-features:
	$(PYTHON) -m $(TRAINER) \
		model.variant="graph_features_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-gnn:
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="graphsage" \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-hgt:
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="hgt" \
		model.gnn.num_heads=4 \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-care:
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="care" \
		model.gnn.similarity_dim=32 \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# Compare GraphSAGE vs HGT on the same test set
compare-gnn-encoders:
	@echo "=== Comparing GNN Encoders ==="
	@echo "Train: $(TRAIN_START) → $(TRAIN_END), Test end: $(TEST_END)"
	@echo ""
	@echo "--- 1/2: GraphSAGE + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="graphsage" \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 2/2: HGT + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.encoder="hgt" \
		model.gnn.num_heads=4 \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "=== Done! Filter runs by tag 'model.encoder' in MLflow UI ==="

# Training with SHAP Analysis
train-vanilla-shap:
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		training.run_shap=true \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

train-gnn-shap:
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=30 \
		training.run_shap=true \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# A/B Comparison — all variants on the same split
compare-all:
	@echo "=== Training all variants for comparison ==="
	@echo "Train: $(TRAIN_START) → $(TRAIN_END), Test end: $(TEST_END)"
	@echo ""
	@echo "--- 1/4: Logistic Regression ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="logistic_regression" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 2/4: Random Forest ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="random_forest" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 3/4: Vanilla XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "--- 4/4: GNN + XGBoost ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=30 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)
	@echo ""
	@echo "=== Comparison complete! Run 'make mlflow' to view results ==="

# ---------------------------------------------------------------------------
# Expanding Window (Concept Drift / RQ3)
# Runs land in experiment: fraud-detection-expanding
# ---------------------------------------------------------------------------

WINDOW_DAYS := 30

expanding-vanilla:
	@echo "=== Expanding Window Training: Vanilla XGBoost ==="
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		training.mode="expanding" \
		data.train_start_date=$(TRAIN_START) \
		data.test_end_date=$(TEST_END) \
		expanding_window.window_days=$(WINDOW_DAYS) \
		expanding_window.min_train_windows=2

expanding-gnn:
	@echo "=== Expanding Window Training: GNN + XGBoost ==="
	$(PYTHON) -m $(TRAINER) \
		model.variant="gnn_xgboost" \
		model.gnn.epochs=50 \
		training.mode="expanding" \
		data.train_start_date=$(TRAIN_START) \
		data.test_end_date=$(TEST_END) \
		expanding_window.window_days=$(WINDOW_DAYS) \
		expanding_window.min_train_windows=2

# Temporal robustness: same model across three fixed periods
temporal-check:
	@echo "=== Temporal Robustness Check ==="
	@echo "--- Period 1: Early (Apr–Jun 2025) ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		data.train_start_date="2025-03-21" \
		data.train_end_date="2025-06-01" \
		data.test_end_date="2025-07-01"
	@echo "--- Period 2: Mid (Apr–Sep 2025) ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		data.train_start_date="2025-03-21" \
		data.train_end_date="2025-09-01" \
		data.test_end_date="2025-10-01"
	@echo "--- Period 3: Full (Apr–Dec 2025) ---"
	$(PYTHON) -m $(TRAINER) \
		model.variant="vanilla_xgboost" \
		data.train_start_date="2025-03-21" \
		data.train_end_date="2025-12-01" \
		data.test_end_date="2026-02-01"
	@echo "=== Temporal check complete! Compare AUC-PR across periods ==="

# ---------------------------------------------------------------------------
# Hyperparameter Optimisation
# Runs land in experiment: fraud-detection-hpo
# ---------------------------------------------------------------------------

hpo:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
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

hpo-custom:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_xgboost \
		hydra.sweeper.n_trials=$(TRIALS) \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

hpo-high-recall:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_high_recall \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

hpo-gnn:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_gnn_xgboost \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

hpo-gnn-quick:
	$(PYTHON) -m $(TRAINER) \
		--config-name=hpo_gnn_xgboost \
		hydra.sweeper.n_trials=10 \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

hpo-gnn-with-best-xgb:
	$(PYTHON) scripts/run_gnn_hpo_with_best_xgb.py \
		--trials=20 \
		--train-start=$(TRAIN_START) \
		--train-end=$(TRAIN_END) \
		--test-end=$(TEST_END)

hpo-hgt:
	@echo "=== HGT Hyperparameter Optimisation (20 trials) ==="
	$(PYTHON) -m $(TRAINER) \
		--multirun \
		--config-name=hpo_hgt \
		data.train_start_date=$(TRAIN_START) \
		data.train_end_date=$(TRAIN_END) \
		data.test_end_date=$(TEST_END)

# ---------------------------------------------------------------------------
# Model Registry — promote a trained model to production
# ---------------------------------------------------------------------------

# Set MODEL_VERSION to the specific version number, or leave unset to use latest.
# Example: make register-model MODEL_ALIAS=production MODEL_NAME=fraud-detection MODEL_VER=3
MODEL_ALIAS := production
MODEL_NAME  := fraud-detection
MODEL_VER   := latest

register-best-hpo:
	@echo "Finding best HPO run by AUC-PR and promoting to '$(MODEL_ALIAS)' …"
	$(PYTHON) -c "\
import mlflow; \
tracking_uri = 'sqlite:///ppa-fraud-detection-mlflow.db'; \
mlflow.set_tracking_uri(tracking_uri); \
client = mlflow.MlflowClient(tracking_uri); \
exp = client.get_experiment_by_name('fraud-detection-hpo'); \
runs = client.search_runs(exp.experiment_id, order_by=['metrics.auc_pr DESC'], max_results=1); \
best = runs[0]; \
print(f'Best run: {best.info.run_id[:8]} | AUC-PR: {best.data.metrics[\"auc_pr\"]:.4f} | AUC-ROC: {best.data.metrics.get(\"auc_roc\", 0):.4f}'); \
mv = mlflow.register_model(f'runs:/{best.info.run_id}/model', '$(MODEL_NAME)'); \
client.set_registered_model_alias('$(MODEL_NAME)', '$(MODEL_ALIAS)', mv.version); \
print(f'Registered version {mv.version} → alias @$(MODEL_ALIAS)')"

register-model:
	@echo "Promoting $(MODEL_NAME) version '$(MODEL_VER)' to alias '$(MODEL_ALIAS)' …"
	$(PYTHON) -c "\
import mlflow; \
client = mlflow.MlflowClient('sqlite:///ppa-fraud-detection-mlflow.db'); \
versions = client.get_latest_versions('$(MODEL_NAME)'); \
v = '$(MODEL_VER)' if '$(MODEL_VER)' != 'latest' else versions[-1].version; \
client.set_registered_model_alias('$(MODEL_NAME)', '$(MODEL_ALIAS)', v); \
print(f'Set alias $(MODEL_ALIAS) → version {v}')"

# ---------------------------------------------------------------------------
# Data Pipeline (ETL)
# ---------------------------------------------------------------------------

etl:
	$(PYTHON) -m src.data.etl

# Export anonymized parquet to CSV for council / stakeholder review.
# PII is already hashed by the ETL — safe to share.
PARQUET := artifacts/merged_events.parquet
CSV     := artifacts/merged_events_anonymized.csv

csv:
	@test -f $(PARQUET) || (echo "Run 'make etl' first — $(PARQUET) not found." && exit 1)
	$(PYTHON) -c "import polars as pl; pl.read_parquet('$(PARQUET)').write_csv('$(CSV)'); print(f'Saved: $(CSV)')"
	@echo "Rows: $$($(PYTHON) -c \"import polars as pl; print(f'{pl.read_parquet(\\\"$(PARQUET)\\\").height:,}')\")"

# ---------------------------------------------------------------------------
# API Service
# ---------------------------------------------------------------------------

api:
	@echo "Starting Fraud Detection API (production mode) …"
	@echo "Model Registry: $(MODEL_NAME) @ $(MODEL_ALIAS)"
	MLFLOW_MODEL_NAME=$(MODEL_NAME) MLFLOW_MODEL_ALIAS=$(MODEL_ALIAS) \
	THRESHOLD_HIGH=0.22 THRESHOLD_MEDIUM=0.16 \
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

api-dev:
	@echo "Starting Fraud Detection API (dev mode, hot-reload) …"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-docs:
	@echo "API documentation available at:"
	@echo "  Swagger UI : http://localhost:8000/docs"
	@echo "  ReDoc      : http://localhost:8000/redoc"

# ---------------------------------------------------------------------------
# Node-RED Demo
# ---------------------------------------------------------------------------

nodered-install:
	cd nodered && npm install

nodered:
	@echo "Starting Node-RED dashboard …"
	@echo "Dashboard : http://localhost:1880/dashboard"
	@echo "Flow editor: http://localhost:1880/"
	cd nodered && npm install --silent && npm start

demo:
	@echo ""
	@echo "=== PPA Fraud Detection Demo ==="
	@echo ""
	@echo "Terminal 1 — start the API:"
	@echo "    make api"
	@echo ""
	@echo "Terminal 2 — start Node-RED:"
	@echo "    make nodered"
	@echo ""
	@echo "Then open the dashboard:"
	@echo "    http://localhost:1880/dashboard"
	@echo ""
	@echo "API docs: http://localhost:8000/docs"
	@echo ""

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

mlflow:
	mlflow ui --backend-store-uri sqlite:///ppa-fraud-detection-mlflow.db

# ---------------------------------------------------------------------------
# Thesis
# ---------------------------------------------------------------------------

thesis:
	@echo "Compiling thesis …"
	@cd thesis && \
	(pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; true) && \
	(biber main > /dev/null 2>&1; true) && \
	(pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; true) && \
	(pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; true) && \
	echo "Thesis compiled: thesis/main.pdf"

thesis-clean:
	cd thesis && latexmk -c

clean:
	rm -rf tmp/
	rm -rf outputs/
	rm -rf multirun/
	rm -rf artifacts/api/
	find . -type d -name "__pycache__" -exec rm -rf {} +
