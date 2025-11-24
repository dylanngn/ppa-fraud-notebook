# SHAP Quick Reference Card

## 🚀 Quickstart (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models with saving
python -m src.cli train-baseline --save-models
python -m src.cli train-hybrid --model hgt --save-models

# 3. Get your answer!
python -m src.cli compare-baseline-hybrid
```

---

## 📋 CLI Commands Cheatsheet

### Explain Single Model
```bash
python -m src.cli explain-model --model-type baseline
```
**Output:** Complete SHAP analysis in `artifacts/shap/baseline/`

### Compare Models
```bash
python -m src.cli compare-models-shap --models "baseline,hybrid_hgt"
```
**Output:** Side-by-side comparison plot

### Analyze Failures
```bash
python -m src.cli analyze-failures --model-type baseline --prediction-type FP
```
**Output:** Waterfall plots for false positives

### ⭐ Comprehensive Comparison
```bash
python -m src.cli compare-baseline-hybrid
```
**Output:** Full report answering "Why doesn't hybrid work?"

---

## 🐍 Python API Quick Examples

### Load and Explain
```python
from src.utils.explainability import ModelExplainer, load_saved_model

# Load
bundle = load_saved_model("artifacts/models/baseline/model_window_0.pkl")

# Explain
exp = ModelExplainer(
    model=bundle['model'],
    feature_names=bundle['features'],
    X_test=bundle['X_test'],
    y_test=bundle['y_test'],
    y_pred=bundle['y_pred'],
    model_type="baseline"
)

# Visualize
exp.plot_summary()
exp.plot_feature_importance_bar()
```

### Explain Single Instance
```python
# Why was listing #42 flagged?
exp.plot_waterfall(42)
```

### Compare Two Models
```python
from src.utils.explainability import compare_models

compare_models(
    {"baseline": baseline_exp, "hybrid": hybrid_exp},
    output_path="comparison.png"
)
```

---

## 📊 What Each Visualization Tells You

| Visualization | Purpose | Key Insight |
|---------------|---------|-------------|
| **Summary Plot** | Global importance | Which features matter most? |
| **Bar Chart** | Simple ranking | Quick comparison across models |
| **Waterfall** | Individual prediction | Why this specific case? |
| **Dependence** | Feature interactions | How do features interact? |
| **Cohort Comparison** | TP vs FP patterns | What distinguishes mistakes? |
| **Performance Over Time** | Temporal trends | Is performance stable? |

---

## 🎯 Decision Tree: What Should I Run?

```
Do you want to...
│
├─ Understand ONE model?
│  └─ Run: explain-model --model-type baseline
│     Output: Summary plots, feature importance, FP analysis
│
├─ Compare MULTIPLE models?
│  └─ Run: compare-models-shap --models "baseline,hybrid_hgt"
│     Output: Side-by-side feature importance
│
├─ Investigate FAILURES?
│  └─ Run: analyze-failures --prediction-type FP
│     Output: Waterfall plots for FPs/FNs
│
└─ Answer "WHY DOESN'T HYBRID WORK?"
   └─ Run: compare-baseline-hybrid ⭐
      Output: Comprehensive report with embedding analysis
```

---

## 🔍 Interpreting Results

### Embedding Contribution %

| Range | Interpretation | Action |
|-------|----------------|--------|
| **>15%** | 🟢 Embeddings helping | Keep hybrid approach |
| **10-15%** | 🟡 Marginal benefit | Consider cost/benefit |
| **5-10%** | 🟠 Weak contribution | Investigate graph quality |
| **<5%** | 🔴 Not helping | Remove hybrid, focus on features |

### Performance Comparison

| Metric | Baseline | Hybrid | Conclusion |
|--------|----------|--------|------------|
| AUC-PR | 0.65 | 0.67 | ✅ Hybrid better (+3%) |
| AUC-PR | 0.65 | 0.65 | ⚠️ No difference |
| AUC-PR | 0.65 | 0.63 | ❌ Hybrid worse |

### Prediction Agreement

| Agreement Rate | Interpretation |
|----------------|----------------|
| **>95%** | Models are nearly identical |
| **90-95%** | Similar but some differences |
| **<90%** | Models disagree significantly |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No models found" | Train with `--save-models` flag |
| SHAP is slow | Use fewer test instances or `aggregate_embeddings=True` |
| Plots are cluttered | Set `max_display=10` |
| Can't compare models | Use same `--window-idx` for both |
| Memory error | Process fewer instances at once |

---

## 📁 Output Files Map

```
artifacts/
└── shap/
    ├── baseline/window_0/
    │   ├── summary_plot.png          # Global feature importance
    │   ├── feature_importance.png    # Bar chart
    │   ├── feature_importance.csv    # Tabular data
    │   ├── fp_explanations/          # False positive waterfalls
    │   ├── tp_vs_fp_comparison.png   # Cohort comparison
    │   └── dependence_*.png          # Feature interactions
    │
    └── comparison/
        ├── performance_comparison.csv          # Metrics by window
        ├── performance_over_time.png           # Trend plot
        └── feature_importance_comparison.png   # Side-by-side
```

---

## 💡 Pro Tips

1. **Start with `compare-baseline-hybrid`** - answers the main question
2. **Use notebook for exploration** - interactive and flexible
3. **Use CLI for production** - automated and reproducible
4. **Always aggregate embeddings** - cleaner visualizations
5. **Compare same windows** - fair comparison
6. **Look at top 10 features** - rest don't usually matter
7. **Check embedding contribution FIRST** - tells you if hybrid can work

---

## 🎓 Learning Path

**Beginner:**
1. Run `explain-model` on baseline
2. Understand summary plot
3. Review top 3 features

**Intermediate:**
4. Run `compare-baseline-hybrid`
5. Analyze embedding contribution
6. Review false positive patterns

**Advanced:**
7. Use Python API for custom analysis
8. Create custom visualizations
9. Integrate insights into feature engineering

---

## 🆘 Need Help?

1. **Full Guide:** `docs/shap_integration_guide.md`
2. **Implementation Details:** `SHAP_INTEGRATION_SUMMARY.md`
3. **Interactive Tutorial:** `notebooks/04_shap_analysis.ipynb`
4. **Project Context:** `docs/knowledge_base.md`

---

**Remember:** The goal is to understand **why hybrid doesn't work**, not just confirm that it doesn't. SHAP gives you the "why"! 🎯

