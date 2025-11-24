# Seon Evaluation - Quick Reference

## 🚀 3-Command Quickstart

```bash
# 1. Evaluate Seon (production baseline)
python -m src.cli evaluate-seon

# 2. Train your models (if not done)
python -m src.cli train-baseline
python -m src.cli train-hybrid --model hgt

# 3. Compare all models
python -m src.cli compare-all-models
```

**Output**: `artifacts/results/comparison/` - Shows if you beat Seon!

---

## 📋 CLI Commands

### Evaluate Seon Performance
```bash
python -m src.cli evaluate-seon \
    --window-days 90 \
    --step-days 7 \
    --include-fallback true
```

### Compare All Models (XGBoost vs Hybrid vs Seon)
```bash
python -m src.cli compare-all-models
```

---

## 🧩 What is Seon?

| Aspect | Details |
|--------|---------|
| **What** | Production fraud detection system |
| **When** | Before listing publication |
| **Output** | Binary (approve/reject) |
| **Metric** | Precision, Recall, F1 |
| **Goal** | Beat this with research models |

---

## 🎯 Evaluation Logic

### Primary (when `seonApproved` available)

| Seon Decision | Fraud Flag | Result |
|---------------|------------|--------|
| Approved (T) | No fraud | **TN** ✅ |
| Approved (T) | Fraud | **FN** ❌ |
| Rejected (F) | Fraud | **TP** ✅ |
| Rejected (F) | No fraud | **FP** ❌ |

### Fallback (when Seon data missing)

| Published | Fraud Flag | Result |
|-----------|------------|--------|
| Yes, then flagged | Fraud | **FN** |
| No | Fraud | **TP** |
| Yes | No fraud | **TN** |

⚠️ **Limitation**: Cannot detect FP in fallback mode

---

## 📊 Understanding Results

### Coverage

| Range | Interpretation | Action |
|-------|----------------|--------|
| **>80%** | 🟢 Reliable | Trust results |
| **50-80%** | 🟡 Fair | Note fallback usage |
| **<50%** | 🔴 Unreliable | Use recent data only |

### Performance Benchmarks

| Metric | Good | Typical | Concern |
|--------|------|---------|---------|
| **Precision** | >0.80 | 0.60-0.80 | <0.60 |
| **Recall** | >0.70 | 0.50-0.70 | <0.50 |
| **F1** | >0.75 | 0.60-0.75 | <0.60 |

---

## 🏆 Who Wins?

### Seon Wins

**Signs:**
- F1 > 0.75
- Better balanced precision/recall
- Lower false alarm rate

**Action:** Keep Seon, improve research

### Research Wins

**Signs:**
- P@100 > Seon Precision
- Better top-K performance
- Lower FP at top of ranking

**Action:** Deploy research model!

### Hybrid Wins

**Signs:**
- Beats both Seon and Baseline
- GNN embeddings effective

**Action:** Deploy hybrid approach!

---

## ⚠️ Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Low coverage (<50%) | Seon data missing | Use recent data |
| Too good (>95% all) | Fallback inflating | Use `--include-fallback false` |
| Can't compare | Different metrics | Understand use cases |

---

## 🐍 Python API

```python
from src.models.evaluate_seon import run_seon_evaluation

# Evaluate Seon
results = run_seon_evaluation(window_days=90)

# Parse Seon decisions
from src.models.evaluate_seon import calculate_seon_labels
import polars as pl

df = pl.read_parquet("artifacts/nodes_listing.parquet")
df = calculate_seon_labels(df)
print(df.select(["seon_approved", "seon_prediction", "is_fraud"]))
```

---

## 📁 Output Files

```
artifacts/results/
├── seon_baseline_results.csv       # Seon per-window metrics
└── comparison/
    ├── all_models_comparison.csv   # Summary table
    ├── all_models_performance.png  # Bar charts
    └── all_models_trends.png       # Time series
```

---

## 🎯 Decision Tree

```
Do you want to...
│
├─ Evaluate SEON only?
│  └─ Run: evaluate-seon
│     Result: Seon performance metrics
│
├─ Compare ALL models?
│  └─ Run: compare-all-models
│     Result: Comprehensive comparison
│
└─ Parse SEON decisions?
   └─ Use: calculate_seon_labels(df)
      Result: DataFrame with Seon predictions
```

---

## 💡 Pro Tips

1. **Always check coverage %** first
2. **Use `--include-fallback false`** for pure Seon eval
3. **Compare recent periods** only (Seon might not exist in old data)
4. **Different metrics** ≠ direct comparison (binary vs probabilistic)
5. **Both have value**: Seon (automated) + Research (prioritization)

---

## 🔄 Typical Workflow

```bash
# Day 1: Establish baselines
python -m src.cli evaluate-seon              # Measure Seon
python -m src.cli train-baseline             # Train baseline

# Day 2: Try improvements
python -m src.cli train-hybrid --model hgt   # Train hybrid

# Day 3: Compare everything
python -m src.cli compare-all-models         # See who wins

# Day 4: Deep analysis
python -m src.cli compare-baseline-hybrid    # Why hybrid wins/loses?
python -m src.cli explain-model --model-type baseline  # SHAP analysis
```

---

## 📖 Full Guides

| Guide | Purpose |
|-------|---------|
| `docs/seon_evaluation_guide.md` | Complete reference |
| `SEON_INTEGRATION_SUMMARY.md` | Implementation details |
| `src/models/evaluate_seon.py` | Source code |

---

## ✅ Checklist

Before comparing:
- [ ] Seon evaluated (`evaluate-seon`)
- [ ] Baseline trained (`train-baseline`)
- [ ] Hybrid trained (`train-hybrid`)
- [ ] Comparison run (`compare-all-models`)
- [ ] Coverage checked (>50%?)
- [ ] Results reviewed

After comparing:
- [ ] Winner identified
- [ ] Coverage noted
- [ ] Action plan decided
- [ ] Next steps clear

---

## 🎯 Your Goal

**Beat Seon with research models while maintaining production quality!**

Good luck! 🚀

