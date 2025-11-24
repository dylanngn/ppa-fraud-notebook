# SHAP Integration Guide

## Overview

This guide explains how to use SHAP (SHapley Additive exPlanations) to understand and debug the XGBoost fraud detection models in this project.

**Key Question:** Why doesn't the hybrid model (GNN + XGBoost) outperform the baseline XGBoost model?

SHAP helps answer this by revealing:
- Which features drive predictions
- How much GNN embeddings actually contribute
- Patterns in false positives and false negatives
- Differences between baseline and hybrid models

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This includes `shap>=0.44.0` and other required packages.

### 2. Train Models with SHAP Support

To enable SHAP analysis, train models with the `--save-models` flag:

```bash
# Train baseline model
python -m src.cli train-baseline --save-models

# Train hybrid model
python -m src.cli train-hybrid --model hgt --save-models
```

This saves model artifacts to `artifacts/models/` for later analysis.

### 3. Generate SHAP Explanations

**Option A: Using the CLI**

```bash
# Explain a specific model
python -m src.cli explain-model --model-type baseline

# Compare baseline vs hybrid
python -m src.cli compare-baseline-hybrid

# Analyze false positives
python -m src.cli analyze-failures --model-type baseline --prediction-type FP
```

**Option B: Using the Interactive Notebook**

Open `notebooks/04_shap_analysis.ipynb` for interactive exploration.

## CLI Commands Reference

### `explain-model`

Generate comprehensive SHAP explanations for a trained model.

```bash
python -m src.cli explain-model \
    --model-type baseline \
    --window-idx -1 \
    --output-dir artifacts/shap/baseline/window_0
```

**Options:**
- `--model-type`: Model type (baseline, hybrid_hgt, hybrid_sage, etc.)
- `--window-idx`: Window to explain (-1 for latest, or specific index)
- `--output-dir`: Output directory (auto-generated if not specified)

**Outputs:**
- `summary_plot.png`: SHAP beeswarm plot
- `feature_importance.png`: Bar chart of feature importance
- `feature_importance.csv`: Tabular feature importance data
- `fp_explanations/`: Waterfall plots for false positives
- `tp_vs_fp_comparison.png`: Cohort comparison plot
- `dependence_*.png`: Dependence plots for top features

### `compare-models-shap`

Compare SHAP feature importance across multiple models.

```bash
python -m src.cli compare-models-shap \
    --models "baseline,hybrid_hgt,hybrid_sage" \
    --window-idx -1 \
    --output-path artifacts/shap/model_comparison.png
```

**Options:**
- `--models`: Comma-separated model types to compare
- `--window-idx`: Window index to use (-1 for latest)
- `--output-path`: Path for comparison plot

### `analyze-failures`

Deep dive into false positives, false negatives, or other prediction types.

```bash
python -m src.cli analyze-failures \
    --model-type baseline \
    --prediction-type FP \
    --top-k 20 \
    --window-idx -1
```

**Options:**
- `--model-type`: Model type to analyze
- `--prediction-type`: TP, FP, TN, or FN
- `--top-k`: Number of instances to explain
- `--window-idx`: Window to analyze (-1 for latest)

### `compare-baseline-hybrid`

**⭐ Most Important Command**

Generate a comprehensive comparison report answering why hybrid doesn't outperform baseline.

```bash
python -m src.cli compare-baseline-hybrid \
    --baseline baseline \
    --hybrid hybrid_hgt \
    --output-dir artifacts/shap/comparison
```

**Outputs:**
- `performance_comparison.csv`: Metrics across all windows
- `performance_over_time.png`: Performance trends
- `feature_importance_comparison.png`: Side-by-side feature importance
- Console report with embedding contribution analysis

## Python API Usage

### Basic SHAP Analysis

```python
from src.utils.explainability import ModelExplainer, load_saved_model

# Load a saved model
model_bundle = load_saved_model("artifacts/models/baseline/model_window_0.pkl")

# Create explainer
explainer = ModelExplainer(
    model=model_bundle['model'],
    feature_names=model_bundle['features'],
    X_test=model_bundle['X_test'],
    y_test=model_bundle['y_test'],
    y_pred=model_bundle['y_pred'],
    model_type="baseline",
    window_info=model_bundle['window_info']
)

# Generate summary plot
explainer.plot_summary(max_display=20, aggregate_embeddings=True)

# Get feature importance DataFrame
importance_df = explainer.get_feature_importance_df()
print(importance_df.head(10))
```

### Explain Individual Predictions

```python
# Explain a specific instance
explainer.plot_waterfall(instance_idx=42, aggregate_embeddings=True)

# Explain top false positives
explainer.explain_top_predictions(
    top_k=10,
    prediction_type='FP',
    output_dir='artifacts/shap/fp_analysis'
)
```

### Compare Models

```python
from src.utils.explainability import compare_models

# Load multiple models
baseline_bundle = load_saved_model("artifacts/models/baseline/model_window_0.pkl")
hybrid_bundle = load_saved_model("artifacts/models/hybrid_hgt/model_window_0.pkl")

# Create explainers
baseline_exp = ModelExplainer(
    model=baseline_bundle['model'],
    feature_names=baseline_bundle['features'],
    X_test=baseline_bundle['X_test'],
    y_test=baseline_bundle['y_test'],
    y_pred=baseline_bundle['y_pred'],
    model_type="baseline"
)

hybrid_exp = ModelExplainer(
    model=hybrid_bundle['model'],
    feature_names=hybrid_bundle['features'],
    X_test=hybrid_bundle['X_test'],
    y_test=hybrid_bundle['y_test'],
    y_pred=hybrid_bundle['y_pred'],
    model_type="hybrid_hgt"
)

# Compare
compare_models(
    {"baseline": baseline_exp, "hybrid_hgt": hybrid_exp},
    top_n_features=15,
    aggregate_embeddings=True,
    output_path="comparison.png"
)
```

### Embedding Contribution Analysis

```python
# Get embedding contribution statistics
embed_features = explainer._get_embedding_features()
print(f"Found {len(embed_features)} embedding features")

# Aggregate embeddings for cleaner visualization
shap_vals, features, X = explainer._aggregate_embedding_shap()
```

## Interpreting SHAP Visualizations

### Summary Plot (Beeswarm)

- **Vertical axis**: Features ranked by importance
- **Horizontal axis**: SHAP value (impact on prediction)
- **Color**: Feature value (red = high, blue = low)
- **Interpretation**: 
  - Features at the top are most important
  - Red dots on the right mean high feature values increase fraud probability
  - Blue dots on the left mean low feature values decrease fraud probability

### Feature Importance Bar Chart

- Simple ranking of features by mean absolute SHAP value
- Higher bars = more important features
- Use for quick comparison across models

### Waterfall Plot

- Shows how a single prediction was made
- **Base value**: Model's average prediction
- **Red bars**: Features pushing toward fraud (positive SHAP)
- **Blue bars**: Features pushing toward legitimate (negative SHAP)
- **Final value**: Actual prediction for this instance

### Dependence Plot

- X-axis: Feature value
- Y-axis: SHAP value (impact on prediction)
- Color: Interaction feature
- Use to understand:
  - Non-linear relationships
  - Feature interactions
  - Threshold effects

## Common Analysis Workflows

### Workflow 1: First-Time Model Analysis

```bash
# 1. Train with model saving
python -m src.cli train-baseline --save-models

# 2. Generate explanations
python -m src.cli explain-model --model-type baseline

# 3. Review outputs in artifacts/shap/baseline/
```

### Workflow 2: Why Doesn't Hybrid Work?

```bash
# 1. Train both models
python -m src.cli train-baseline --save-models
python -m src.cli train-hybrid --model hgt --save-models

# 2. Run comparison
python -m src.cli compare-baseline-hybrid

# 3. Check artifacts/shap/comparison/ for insights
```

**Look for:**
- Embedding contribution % (should be >10% to be meaningful)
- Performance differences (AUC-PR, P@100)
- Top features comparison (are embeddings even in top 10?)
- Agreement rate (if >95%, models are too similar)

### Workflow 3: Debug False Positives

```bash
# 1. Analyze FPs
python -m src.cli analyze-failures \
    --model-type baseline \
    --prediction-type FP \
    --top-k 20

# 2. Review waterfall plots in:
#    artifacts/shap/baseline/window_X/FP_analysis/

# 3. Look for common patterns:
#    - Which features consistently push toward fraud?
#    - Are there threshold issues with specific features?
```

### Workflow 4: Interactive Exploration

```bash
# Launch Jupyter
jupyter notebook notebooks/04_shap_analysis.ipynb

# Follow the notebook cells to:
# - Load models
# - Generate interactive plots
# - Analyze embedding contributions
# - Export custom visualizations
```

## Key Insights from SHAP Analysis

### Expected Findings

1. **`account_age_days` dominates**: New accounts are highly predictive of fraud
2. **Price-related features matter**: Abnormal pricing patterns signal fraud
3. **Graph features add value**: If PageRank, component size show up in top 10
4. **Embeddings might be weak**: If contribution <5%, GNN isn't helping

### What to Do Based on Results

**If embeddings contribute <5%:**
- Graph is too sparse or disconnected
- Try graph-derived features (PageRank, clustering) instead
- Investigate GNN architecture (try GAT, GraphSAGE)
- Consider removing hybrid approach

**If baseline and hybrid perform equally:**
- Embeddings are redundant with tabular features
- Focus on feature engineering instead
- Graph structure may not capture fraud patterns

**If many false positives share patterns:**
- Adjust feature thresholds
- Add interaction terms
- Consider post-hoc calibration

**If feature importance is unstable across windows:**
- Concept drift is occurring
- Need more robust features
- Consider adaptive thresholding

## Troubleshooting

### "No models found" Error

**Problem:** You forgot to train with `--save-models`

**Solution:**
```bash
python -m src.cli train-baseline --save-models
```

### SHAP Takes Too Long

**Problem:** SHAP computation is slow for large datasets

**Solutions:**
- Use a smaller window (fewer test instances)
- Reduce number of background samples
- Use `aggregate_embeddings=True` for hybrid models

### Plots Are Cluttered

**Problem:** Too many features or embedding dimensions

**Solutions:**
- Set `max_display=10` to show fewer features
- Use `aggregate_embeddings=True` to group embeddings
- Focus on specific features with `plot_dependence()`

### Can't Compare Models

**Problem:** Different test sets across models

**Solution:** Ensure you're comparing the same window index:
```bash
python -m src.cli compare-models-shap \
    --models "baseline,hybrid_hgt" \
    --window-idx 0  # Use same window
```

## Additional Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP GitHub](https://github.com/slundberg/shap)
- [Interpreting XGBoost with SHAP](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)

## Support

For questions or issues with SHAP integration, check:
1. `docs/knowledge_base.md` for project-specific context
2. `docs/experiment_journal.md` for modeling decisions
3. SHAP documentation for visualization questions

