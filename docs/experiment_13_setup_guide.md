# Experiment 13: Baseline vs Hybrid Comparison - Setup Guide

This guide provides step-by-step instructions for re-running the baseline vs hybrid model comparison with MLflow integration, signature validation, and clean feature setup.

## Prerequisites

1. **Data Pipeline**: Ensure all artifacts are generated
   ```bash
   make etl
   make build-graph
   ```

2. **Feature Generation**: Generate all feature categories
   ```bash
   python src/cli.py graph-features
   python src/cli.py advanced-graph-features
   python src/cli.py time-weighted-features
   # Optional: python src/cli.py interaction-features
   # Optional: python src/cli.py text-features
   ```

3. **MLflow**: Ensure MLflow tracking server is accessible
   ```bash
   # Local file-based tracking (default)
   # MLflow will use ./mlruns directory
   ```

## Experiment Configuration

### Parameters to Use

All models should use **identical parameters** for fair comparison:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `experiment_name` | `exp13-baseline-vs-hybrid-365d` | MLflow experiment name |
| `initial_window_days` | `365` | Initial training window (days) - gives GNNs more historical data |
| `step_days` | `42` | Step size between evaluation windows - balanced speed and statistical validity |
| `feature_categories` | `base,graph,advanced_graph,time_weighted` | Feature categories to include |

**Note**: We exclude `interaction` features based on Experiment 10 findings (they don't help tree models).

### Hybrid Model Specific Parameters

| Model | Epochs | Description |
|-------|--------|-------------|
| SAGE | `25` | Standard SAGE training epochs |
| HGT | `30` | HGT with RTE (requires more epochs) |

## Running the Experiment

### Option 1: Automated Script (Recommended)

Use the provided automation script:

```bash
bash scripts/run_model_comparison_experiment.sh
```

This script will:
1. Train baseline model
2. Train SAGE hybrid model
3. Train HGT hybrid model
4. Validate model signatures
5. Generate comparison report

**Estimated Time** (with 42-day step size):
- **Number of evaluation windows**: ~16 windows (data range: 2023-01-01 to 2025-11-02, ~2.8 years)
- Baseline: ~8 minutes (16 windows × ~0.5 min/window)
- SAGE Hybrid: ~30 minutes (GPU required, 16 windows × ~2 min/window)
- HGT Hybrid: ~30 minutes (GPU required, 16 windows × ~2 min/window)
- **Total**: ~1.1 hours (balanced: speed + statistical validity)

### Option 2: Manual Step-by-Step

If you prefer manual control or want to run models separately:

#### Step 1: Train Baseline

```bash
python src/cli.py train-baseline \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --initial-window-days 365 \
    --step-days 42 \
    --feature-categories "base,graph,advanced_graph,time_weighted"
```

**Save the run ID** from the output (you'll need it for comparison).

#### Step 2: Train SAGE Hybrid

```bash
python src/cli.py train-hybrid-sage \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --initial-window-days 365 \
    --step-days 42 \
    --epochs 25
```

**Save the run ID** from the output.

#### Step 3: Train HGT Hybrid

```bash
python src/cli.py train-hybrid-hgt \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --initial-window-days 365 \
    --step-days 42 \
    --epochs 30
```

**Save the run ID** from the output.

#### Step 4: Validate Signatures

```bash
python -m src.utils.signature_comparison \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --model-types baseline_graph hybrid_sage hybrid_hgt
```

This ensures all models have compatible signatures for fair comparison.

#### Step 5: Generate Comparison Report

```bash
python scripts/compare_model_results.py \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --baseline-run-id <YOUR_BASELINE_RUN_ID> \
    --sage-run-id <YOUR_SAGE_RUN_ID> \
    --hgt-run-id <YOUR_HGT_RUN_ID> \
    --output-dir artifacts/comparison_results/$(date +%Y%m%d_%H%M%S)
```

## Metrics to Monitor

### Primary Metrics

1. **Mean AUC-PR** (`mean_auc_pr`)
   - **Target**: Baseline > 0.70, Hybrid competitive
   - **Interpretation**: Overall ranking quality

2. **Best AUC-PR** (`best_auc_pr`)
   - **Target**: Highest single-window performance
   - **Interpretation**: Peak model capability

3. **Mean Precision@100** (`mean_p_at_100`)
   - **Target**: > 0.75
   - **Interpretation**: Production review efficiency

4. **Mean AUC-ROC** (`mean_auc_roc`)
   - **Target**: > 0.90
   - **Interpretation**: Overall discrimination ability

### Secondary Metrics

5. **Number of Windows** (`num_windows`)
   - **Expected**: ~120-150 windows (depends on data range)
   - **Interpretation**: Evaluation coverage

6. **Window-level AUC-PR** (`window_*_auc_pr`)
   - **Use**: Identify performance trends over time
   - **Interpretation**: Model stability

### Training Metrics

7. **Training Time** (logged as tag `training_time_minutes`)
   - **Baseline**: ~15 minutes
   - **Hybrid**: ~120-180 minutes
   - **Interpretation**: Cost comparison

8. **Feature Count** (logged as tag `feature_count`)
   - **Baseline**: ~50-70 features (base + graph)
   - **Hybrid**: ~114-134 features (base + graph + 64 embeddings)
   - **Interpretation**: Model complexity

## Analysis & Visualization

### Option 1: Jupyter Notebook (Recommended)

Use the provided notebook for interactive analysis:

```bash
jupyter notebook notebooks/model_comparison_analysis.ipynb
```

**Steps**:
1. Update run IDs in the first cell
2. Run all cells
3. Review visualizations and summary

**What you'll get**:
- Performance comparison charts
- Window-by-window trend analysis
- Feature importance visualization
- Summary statistics and recommendations

### Option 2: MLflow UI

View results in MLflow UI:

```bash
mlflow ui
```

Navigate to:
- **Experiments** → `exp13-baseline-vs-hybrid-365d`
- Compare runs side-by-side
- View metrics, parameters, artifacts

**Key Views**:
- **Metrics**: Compare `mean_auc_pr`, `mean_p_at_100`
- **Parameters**: Verify consistent configuration
- **Artifacts**: View SHAP plots, feature importance

### Option 3: Text Reports

Check generated reports:

```bash
# Comparison report
cat artifacts/comparison_results/<timestamp>/comparison_report.txt

# Signature validation
cat artifacts/comparison_results/<timestamp>/signature_comparison.txt

# CSV data
cat artifacts/comparison_results/<timestamp>/metrics_comparison.csv
```

## Expected Results

Based on previous experiments (Experiment 7):

| Model | Expected AUC-PR | Expected P@100 | Training Time |
|-------|----------------|----------------|---------------|
| **Baseline** | **0.64-0.70** | **0.75-0.78** | ~15 min |
| SAGE Hybrid | 0.63-0.64 | ~0.75 | ~2-3 hours |
| HGT Hybrid | 0.64-0.65 | ~0.75 | ~2-3 hours |

**Note**: With optimized hyperparameters (Experiment 11), baseline may achieve **0.70+ AUC-PR**.

## Decision Criteria

### If Baseline Wins (>5% improvement):
- ✅ **Continue with Baseline optimization** (Experiment 13A)
- Focus on feature engineering, hyperparameter tuning
- Document GNN limitations for research

### If Hybrid Wins (>2% improvement):
- ⚠️ **Investigate hybrid improvements** (Experiment 13B)
- Consider cost-benefit analysis
- May need GPU infrastructure for production

### If Models are Competitive (<2% difference):
- ⚠️ **Cost-benefit analysis**:
  - Baseline: Faster, simpler, interpretable
  - Hybrid: More complex, requires GPU, less interpretable
- **Recommendation**: Choose baseline for production (lower TCO)

## Troubleshooting

### Issue: Models have incompatible signatures

**Solution**:
```bash
# Re-run signature validation
python -m src.utils.signature_comparison \
    --experiment-name "exp13-baseline-vs-hybrid-365d"
```

If signatures are incompatible:
1. Check feature categories match across models
2. Verify all features are generated
3. Retrain models with consistent configuration

### Issue: MLflow run IDs not found

**Solution**:
```bash
# List all runs in experiment
mlflow runs search --experiment-names "exp13-baseline-vs-hybrid-365d" \
    --max-results 10

# Get latest run for each model type
mlflow runs search --experiment-names "exp13-baseline-vs-hybrid-365d" \
    --filter "tags.model_type = 'baseline_graph'" \
    --max-results 1 \
    --order-by "start_time DESC"
```

### Issue: GPU out of memory (Hybrid models)

**Solution**:
- Reduce batch size in GNN training
- Reduce embedding dimensions (64 → 32)
- Use gradient accumulation

### Issue: Training takes too long

**Solution**:
- Reduce `step_days` from 7 to 14 (fewer windows)
- Use `max_windows` parameter to limit evaluation
- Run models in parallel (separate machines)

## Next Steps

After completing the comparison:

1. **Document Results**: Update `docs/experiment_journal.md` with new results
2. **Choose Direction**: Based on results, follow Experiment 13A (Baseline) or 13B (Hybrid)
3. **Plan Experiments**: Review Experiment 13 plan for next steps

## Files Generated

After running the experiment, you'll have:

```
artifacts/comparison_results/
└── <timestamp>/
    ├── baseline_run_id.txt
    ├── sage_run_id.txt
    ├── hgt_run_id.txt
    ├── comparison_report.txt
    ├── signature_comparison.txt
    ├── metrics_comparison.csv
    └── model_metadata.csv
```

All results are also logged to MLflow for permanent tracking.

