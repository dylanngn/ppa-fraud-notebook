.PHONY: help install etl build-graph seon-baseline \
	train train-quick train-sage train-hgt \
	optimize-xgboost data-quality-report \
	mlflow-compare mlflow-recommend mlflow-drift

# =============================================================================
# HELP
# =============================================================================
help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "                   FRAUD DETECTION PIPELINE                        "
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Data Pipeline (run once):"
	@echo "  make etl              Extract, transform, load data"
	@echo "  make build-graph      Build PyTorch Geometric graph"
	@echo "  make seon-baseline    Generate static Seon baseline"
	@echo ""
	@echo "Model Training:"
	@echo "  make train            XGBoost with production features"
	@echo "  make train-quick      XGBoost with core features (fast)"
	@echo "  make train-sage       SAGE GNN hybrid"
	@echo "  make train-hgt        HGT GNN hybrid"
	@echo ""
	@echo "Optimization:"
	@echo "  make optimize-xgboost Tune XGBoost hyperparameters"
	@echo ""
	@echo "Analysis:"
	@echo "  make data-quality-report  Generate data quality report"
	@echo ""
	@echo "MLflow (requires MODEL_NAME, CANDIDATE_RUN_ID):"
	@echo "  make mlflow-compare   Compare candidate vs production"
	@echo "  make mlflow-recommend Get deployment recommendation"
	@echo "  make mlflow-drift     Analyze model drift"
	@echo ""

# =============================================================================
# SETUP
# =============================================================================
install:
	pip install -r requirements.txt

# =============================================================================
# DATA PIPELINE
# =============================================================================
etl:
	python -m src.data.pipeline

build-graph:
	python -m src.data.graph.create_artifacts

seon-baseline:
	python -m src.utils.evaluate_seon

# =============================================================================
# MODEL TRAINING
# =============================================================================
train:
	python -m src.models.train features=production

train-quick:
	python -m src.models.train features=quick

train-sage:
	python -m src.models.gnn.sage

train-hgt:
	python -m src.models.gnn.hgt

# =============================================================================
# OPTIMIZATION
# =============================================================================
optimize-xgboost:
	python -m src.models.hyperopt.xgboost

# =============================================================================
# ANALYSIS
# =============================================================================
data-quality-report:
	python -m src.utils.data_quality_report

# =============================================================================
# MLFLOW MODEL MANAGEMENT
# =============================================================================
mlflow-compare:
ifndef MODEL_NAME
	$(error MODEL_NAME is required)
endif
ifndef CANDIDATE_RUN_ID
	$(error CANDIDATE_RUN_ID is required)
endif
	python -m src.utils.mlflow_model_comparison compare \
		--model-name $(MODEL_NAME) \
		--candidate-run-id $(CANDIDATE_RUN_ID)

mlflow-recommend:
ifndef MODEL_NAME
	$(error MODEL_NAME is required)
endif
ifndef CANDIDATE_RUN_ID
	$(error CANDIDATE_RUN_ID is required)
endif
	python -m src.utils.mlflow_model_comparison recommend \
		--model-name $(MODEL_NAME) \
		--candidate-run-id $(CANDIDATE_RUN_ID)

mlflow-drift:
ifndef MODEL_NAME
	$(error MODEL_NAME is required)
endif
	python -m src.utils.mlflow_model_comparison drift \
		--model-name $(MODEL_NAME)
