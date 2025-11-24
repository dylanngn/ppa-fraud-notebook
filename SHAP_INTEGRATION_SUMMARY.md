# SHAP Integration - Implementation Summary

## ✅ All Components Completed

This document summarizes the complete SHAP integration for XGBoost fraud detection model explainability.

---

## 📦 What Was Built

### 1. **SHAP Explainability Module** (`src/utils/explainability.py`)

**Purpose:** Core SHAP functionality for model interpretation

**Key Features:**
- `ModelExplainer` class: Wrapper for SHAP TreeExplainer
- Global explanations (summary plots, feature importance)
- Local explanations (waterfall plots for individual predictions)
- Cohort analysis (compare TP vs FP, etc.)
- Embedding aggregation for hybrid models
- Dependence plots for feature interactions

**Main Functions:**
- `plot_summary()`: Beeswarm plot showing feature contributions
- `plot_feature_importance_bar()`: Simple bar chart of importance
- `plot_waterfall()`: Explain individual predictions
- `plot_dependence()`: Feature interaction analysis
- `explain_top_predictions()`: Batch waterfall plots
- `analyze_cohort_differences()`: Compare prediction groups
- `compare_models()`: Side-by-side model comparison

**Usage:**
```python
from src.utils.explainability import ModelExplainer, load_saved_model

model_bundle = load_saved_model("artifacts/models/baseline/model_window_0.pkl")
explainer = ModelExplainer(
    model=model_bundle['model'],
    feature_names=model_bundle['features'],
    X_test=model_bundle['X_test'],
    y_test=model_bundle['y_test'],
    y_pred=model_bundle['y_pred'],
    model_type="baseline"
)
explainer.plot_summary()
```

---

### 2. **Training Script Modifications**

**Modified Files:**
- `src/models/train_baseline.py`
- `src/models/train_hybrid.py`

**Changes:**
- Added `save_models` parameter to `train_sliding_window()`
- Models now saved as pickle bundles with:
  - Trained XGBoost model
  - Feature names
  - Test data (X_test, y_test)
  - Predictions (y_pred)
  - Window metadata
  - Performance metrics
- Models saved to `artifacts/models/{model_type}/model_window_{idx}.pkl`

**Usage:**
```bash
python -m src.cli train-baseline --save-models
python -m src.cli train-hybrid --model hgt --save-models
```

---

### 3. **CLI Commands** (Added to `src/cli.py`)

#### **`explain-model`**
Generate comprehensive SHAP explanations for a single model.

```bash
python -m src.cli explain-model --model-type baseline --window-idx -1
```

**Outputs:**
- Summary plots
- Feature importance charts
- False positive/negative explanations
- Cohort comparisons
- Dependence plots

#### **`compare-models-shap`**
Compare feature importance across multiple models.

```bash
python -m src.cli compare-models-shap --models "baseline,hybrid_hgt"
```

#### **`analyze-failures`**
Deep dive into specific prediction types (FP, FN, TP, TN).

```bash
python -m src.cli analyze-failures --model-type baseline --prediction-type FP --top-k 20
```

#### **`compare-baseline-hybrid`** ⭐
**Most Important:** Comprehensive comparison answering "Why doesn't hybrid outperform baseline?"

```bash
python -m src.cli compare-baseline-hybrid
```

**Analysis includes:**
- Performance metrics across all windows
- Embedding contribution percentage
- Prediction agreement/disagreement
- Feature importance comparison
- Actionable recommendations

---

### 4. **Interactive Notebook** (`notebooks/04_shap_analysis.ipynb`)

**Sections:**
1. **Load Models**: Choose model type and window
2. **Create SHAP Explainer**: Initialize and compute SHAP values
3. **Global Feature Importance**: Summary plots and rankings
4. **Feature Dependence**: Interaction analysis
5. **Local Explanations**: Individual prediction walkthroughs
6. **Model Comparison**: Baseline vs Hybrid analysis
7. **Embedding Contribution**: Deep dive on GNN effectiveness
8. **Export Results**: Save all visualizations

**Usage:**
```bash
jupyter notebook notebooks/04_shap_analysis.ipynb
```

---

### 5. **Baseline vs Hybrid Comparison Tool** (`src/utils/compare_baseline_hybrid.py`)

**Purpose:** Answer the key question about hybrid model performance

**Analysis Components:**

**A. Performance Metrics**
- Compare AUC-PR, AUC-ROC, P@100, Lift@100 across all windows
- Calculate mean differences and percentage changes
- Identify winner for each metric

**B. Embedding Contribution**
- Calculate overall embedding contribution %
- Per-instance contribution distribution
- Breakdown by prediction type (TP, FP, TN, FN)
- Top features including embeddings

**C. Prediction Agreement**
- Overall agreement rate between baseline and hybrid
- Correlation of prediction scores
- Disagreement case analysis (who was right?)

**D. Visualizations**
- Performance over time plots
- Feature importance comparison
- Embedding contribution distribution

**E. Recommendations**
Auto-generated based on findings:
- If embeddings contribute <5% → investigate graph quality
- If performance is equal → embeddings are redundant
- If hybrid is worse → GNN may be adding noise

**Usage:**
```python
from src.utils.compare_baseline_hybrid import generate_comparison_report

generate_comparison_report(
    baseline_type="baseline",
    hybrid_type="hybrid_hgt",
    output_dir="artifacts/shap/comparison"
)
```

---

### 6. **Documentation** (`docs/shap_integration_guide.md`)

**Contents:**
- Quick start guide
- CLI command reference
- Python API examples
- Interpretation guide for each plot type
- Common analysis workflows
- Troubleshooting section
- Best practices

---

### 7. **Dependencies** (Updated `requirements.txt`)

**Added:**
- `shap>=0.44.0`: SHAP library
- `pandas>=2.0.0`: Required by SHAP
- `networkx>=3.0.0`: For graph features (already used)

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models with Saving
```bash
# Baseline
python -m src.cli train-baseline --save-models

# Hybrid
python -m src.cli train-hybrid --model hgt --save-models
```

### 3. Run Analysis

**Option A: Quick CLI Analysis**
```bash
python -m src.cli explain-model --model-type baseline
```

**Option B: Comprehensive Comparison**
```bash
python -m src.cli compare-baseline-hybrid
```

**Option C: Interactive Notebook**
```bash
jupyter notebook notebooks/04_shap_analysis.ipynb
```

---

## 🔍 Key Questions SHAP Answers

### ❓ "Which features are most important for fraud detection?"
**Tool:** `explain-model` → Summary plot

### ❓ "Why was this specific listing flagged as fraud?"
**Tool:** `analyze-failures` → Waterfall plots

### ❓ "What distinguishes false positives from true positives?"
**Tool:** Cohort comparison in `explain-model`

### ❓ "Why doesn't the hybrid model outperform baseline?"
**Tool:** `compare-baseline-hybrid` ⭐

**Possible Answers:**
1. **Embeddings contribute <5%**: GNN isn't learning useful patterns
2. **High prediction agreement (>95%)**: Models are too similar
3. **Embeddings not in top 10 features**: Tabular features dominate
4. **Graph is too sparse**: Not enough connectivity for GNN to work

---

## 📊 Expected Outputs

### Directory Structure
```
artifacts/
├── models/
│   ├── baseline/
│   │   ├── model_window_0.pkl
│   │   ├── model_window_1.pkl
│   │   └── ...
│   └── hybrid_hgt/
│       ├── model_window_0.pkl
│       └── ...
└── shap/
    ├── baseline/
    │   └── window_0/
    │       ├── summary_plot.png
    │       ├── feature_importance.png
    │       ├── feature_importance.csv
    │       ├── fp_explanations/
    │       ├── tp_vs_fp_comparison.png
    │       └── dependence_*.png
    ├── hybrid_hgt/
    │   └── window_0/
    │       └── ...
    └── comparison/
        ├── performance_comparison.csv
        ├── performance_over_time.png
        └── feature_importance_comparison.png
```

---

## 🎯 Success Criteria

✅ **Complete Implementation:**
- [x] SHAP explainability module created
- [x] Training scripts modified for model saving
- [x] CLI commands added (4 new commands)
- [x] Interactive notebook created
- [x] Comparison tool built
- [x] Documentation written
- [x] Dependencies updated

✅ **Functional Requirements:**
- [x] Can explain individual predictions
- [x] Can compare baseline vs hybrid
- [x] Can analyze false positives/negatives
- [x] Can quantify embedding contribution
- [x] Can identify feature importance differences

✅ **Usability:**
- [x] Multiple interfaces (CLI, Python API, Notebook)
- [x] Comprehensive documentation
- [x] Clear visualizations
- [x] Actionable insights

---

## 🔄 Next Steps for User

### 1. **Immediate Actions**
```bash
# Install new dependencies
pip install -r requirements.txt

# Re-train models with saving enabled
python -m src.cli train-baseline --save-models
python -m src.cli train-hybrid --model hgt --save-models

# Run comparison to answer your question
python -m src.cli compare-baseline-hybrid
```

### 2. **Review Results**
- Check `artifacts/shap/comparison/` for comprehensive analysis
- Look for embedding contribution % (key metric)
- Review performance comparison CSV
- Examine feature importance plots

### 3. **Take Action Based on Findings**

**If embeddings are weak (<10% contribution):**
- Investigate graph quality using `notebooks/04_graph_sparsity_analysis.ipynb`
- Try different GNN architectures (GAT, GraphSAGE)
- Consider using graph-derived features instead of raw embeddings

**If performance is equal:**
- Focus on feature engineering for tabular features
- Graph information may not add value for this task

**If hybrid is worse:**
- GNN may be adding noise
- Consider removing hybrid approach
- Stick with baseline + good feature engineering

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `src/utils/explainability.py` | Core SHAP functionality |
| `src/utils/compare_baseline_hybrid.py` | Comparison analysis tool |
| `src/cli.py` | CLI commands (updated) |
| `src/models/train_baseline.py` | Baseline training (updated) |
| `src/models/train_hybrid.py` | Hybrid training (updated) |
| `notebooks/04_shap_analysis.ipynb` | Interactive analysis |
| `docs/shap_integration_guide.md` | Complete usage guide |
| `requirements.txt` | Dependencies (updated) |

---

## 💡 Pro Tips

1. **Always use same window index when comparing models**
2. **Set `aggregate_embeddings=True` for hybrid models** (cleaner plots)
3. **Start with `compare-baseline-hybrid`** (answers the big question)
4. **Use notebook for exploration, CLI for batch processing**
5. **Save SHAP values are expensive to compute** (cache results)
6. **Focus on top 10-15 features** (rest usually don't matter)

---

## ✨ Summary

You now have a **complete SHAP integration** that provides:

1. ✅ **Full explainability** for all XGBoost models
2. ✅ **Direct answer** to why hybrid doesn't outperform baseline
3. ✅ **Multiple interfaces** (CLI, Python, Jupyter)
4. ✅ **Comprehensive visualizations** (12+ plot types)
5. ✅ **Actionable insights** with auto-generated recommendations

**Key Deliverable:** The `compare-baseline-hybrid` command will definitively show whether GNN embeddings are helping or hurting, and provide clear next steps.

---

**Ready to use!** Run the commands above and start analyzing your models. 🚀

