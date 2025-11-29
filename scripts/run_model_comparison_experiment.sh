#!/bin/bash
# Experiment 13: Baseline vs Hybrid Model Comparison
# This script runs all models with consistent parameters and MLflow tracking

set -e  # Exit on error

EXPERIMENT_NAME="exp13-baseline-vs-hybrid-365d"
INITIAL_WINDOW_DAYS=365
STEP_DAYS=42

# Feature categories to use (all features for fair comparison)
FEATURE_CATEGORIES="base,graph,advanced_graph,time_weighted"

echo "=========================================="
echo "Model Comparison Experiment"
echo "=========================================="
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Initial Window: $INITIAL_WINDOW_DAYS days (365d for GNN advantage)"
echo "Step Size: $STEP_DAYS days (14d for faster training)"
echo "Feature Categories: $FEATURE_CATEGORIES"
echo ""
echo "Expected Configuration:"
echo "  Data range: 2023-01-01 to 2025-11-02 (~2.8 years, 1036 days)"
echo "  Evaluation windows: ~16 windows"
echo "  Estimated time: Baseline ~8 min, Hybrids ~30 min each"
echo "  Total time: ~1.1 hours"
echo ""
echo "Note: 365-day window gives GNNs more historical data"
echo "      42-day step = ~16 windows (balanced: speed + statistical validity)"
echo ""

# Create results directory
mkdir -p artifacts/comparison_results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="artifacts/comparison_results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Step 1: Train Baseline XGBoost
echo "=========================================="
echo "Step 1/3: Training Baseline XGBoost"
echo "=========================================="
python src/cli.py train-baseline \
    --experiment-name "$EXPERIMENT_NAME" \
    --initial-window-days $INITIAL_WINDOW_DAYS \
    --step-days $STEP_DAYS \
    --feature-categories "$FEATURE_CATEGORIES" || {
    echo "ERROR: Baseline training failed"
    exit 1
}

BASELINE_RUN_ID=$(python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///fraud-detection-mlflow.db')
exp = mlflow.get_experiment_by_name('$EXPERIMENT_NAME')
if exp:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], 
                              filter_string=\"tags.model_type = 'baseline_graph'\",
                              max_results=1, order_by=['start_time DESC'])
    if not runs.empty:
        print(runs.iloc[0]['run_id'])
")

echo "Baseline Run ID: $BASELINE_RUN_ID"
echo "$BASELINE_RUN_ID" > "$RESULTS_DIR/baseline_run_id.txt"
echo ""

# Step 2: Train SAGE Hybrid
echo "=========================================="
echo "Step 2/3: Training SAGE Hybrid Model"
echo "=========================================="
python src/cli.py train-hybrid-sage \
    --experiment-name "$EXPERIMENT_NAME" \
    --initial-window-days $INITIAL_WINDOW_DAYS \
    --step-days $STEP_DAYS \
    --epochs 25 || {
    echo "ERROR: SAGE hybrid training failed"
    exit 1
}

SAGE_RUN_ID=$(python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///fraud-detection-mlflow.db')
exp = mlflow.get_experiment_by_name('$EXPERIMENT_NAME')
if exp:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], 
                              filter_string=\"tags.model_type = 'hybrid_sage'\",
                              max_results=1, order_by=['start_time DESC'])
    if not runs.empty:
        print(runs.iloc[0]['run_id'])
")

echo "SAGE Run ID: $SAGE_RUN_ID"
echo "$SAGE_RUN_ID" > "$RESULTS_DIR/sage_run_id.txt"
echo ""

# Step 3: Train HGT Hybrid
echo "=========================================="
echo "Step 3/3: Training HGT Hybrid Model"
echo "=========================================="
python src/cli.py train-hybrid-hgt \
    --experiment-name "$EXPERIMENT_NAME" \
    --initial-window-days $INITIAL_WINDOW_DAYS \
    --step-days $STEP_DAYS \
    --epochs 30 || {
    echo "ERROR: HGT hybrid training failed"
    exit 1
}

HGT_RUN_ID=$(python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///fraud-detection-mlflow.db')
exp = mlflow.get_experiment_by_name('$EXPERIMENT_NAME')
if exp:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], 
                              filter_string=\"tags.model_type = 'hybrid_hgt'\",
                              max_results=1, order_by=['start_time DESC'])
    if not runs.empty:
        print(runs.iloc[0]['run_id'])
")

echo "HGT Run ID: $HGT_RUN_ID"
echo "$HGT_RUN_ID" > "$RESULTS_DIR/hgt_run_id.txt"
echo ""

# Step 4: Validate Signatures
echo "=========================================="
echo "Step 4: Validating Model Signatures"
echo "=========================================="
python src/utils/signature_comparison.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --model-types baseline_graph hybrid_sage hybrid_hgt \
    > "$RESULTS_DIR/signature_comparison.txt"

echo "Signature validation complete. Results saved to: $RESULTS_DIR/signature_comparison.txt"
echo ""

# Step 5: Generate Comparison Report
echo "=========================================="
echo "Step 5: Generating Comparison Report"
echo "=========================================="
python scripts/compare_model_results.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --baseline-run-id "$BASELINE_RUN_ID" \
    --sage-run-id "$SAGE_RUN_ID" \
    --hgt-run-id "$HGT_RUN_ID" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view results:"
echo "1. MLflow UI: mlflow ui"
echo "2. Comparison report: cat $RESULTS_DIR/comparison_report.txt"
echo "3. Signature validation: cat $RESULTS_DIR/signature_comparison.txt"
echo ""

