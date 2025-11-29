# Experiment 13: Quick Reference

## Quick Start

### 1. Run the Experiment

```bash
bash scripts/run_model_comparison_experiment.sh
```

**Time**: ~1.1 hours
- **Windows**: ~16 evaluation windows (data: 2023-01-01 to 2025-11-02, ~2.8 years)
- Baseline: ~8 min
- Hybrids: ~30 min each

### 2. View Results

**Option A: Jupyter Notebook** (Recommended)
```bash
jupyter notebook notebooks/model_comparison_analysis.ipynb
```
- Update run IDs in first cell
- Run all cells
- Get visualizations and summary

**Option B: MLflow UI**
```bash
mlflow ui
```
- Navigate to experiment: `ppa-fraud-detection-comparison`
- Compare runs side-by-side

**Option C: Text Reports**
```bash
cat artifacts/comparison_results/<timestamp>/comparison_report.txt
```

## Key Metrics to Check

| Metric | Target | What It Means |
|--------|--------|--------------|
| **mean_auc_pr** | > 0.70 (Baseline) | Overall ranking quality |
| **mean_p_at_100** | > 0.75 | Production review efficiency |
| **mean_auc_roc** | > 0.90 | Overall discrimination ability |

## Expected Results

| Model | AUC-PR | P@100 | Training Time |
|-------|--------|-------|---------------|
| Baseline | 0.64-0.70 | 0.75-0.78 | ~8 min (16 windows) |
| SAGE Hybrid | 0.63-0.64 | ~0.75 | ~30 min (16 windows) |
| HGT Hybrid | 0.64-0.65 | ~0.75 | ~30 min (16 windows) |

**Note**: 
- 365-day initial window gives GNNs more historical data advantage
- 42-day step = ~16 evaluation windows (balanced: speed + statistical validity)
- Data range: 2023-01-01 to 2025-11-02 (~2.8 years, 1036 days)

## Decision Tree

```
Is Baseline > Hybrid by >5%?
├─ YES → Continue Baseline optimization (Experiment 13A)
└─ NO
   ├─ Is Hybrid > Baseline by >2%?
   │  ├─ YES → Investigate Hybrid (Experiment 13B)
   │  └─ NO → Cost-benefit analysis (choose Baseline)
   └─ Models competitive (<2% diff)?
      └─ Choose Baseline (lower TCO)
```

## Commands Reference

### Train Models

```bash
# Baseline
python src/cli.py train-baseline \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --initial-window-days 365 \
    --step-days 42 \
    --feature-categories "base,graph,advanced_graph,time_weighted"

# SAGE Hybrid
python src/cli.py train-hybrid-sage \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --initial-window-days 365 \
    --step-days 42 \
    --epochs 25

# HGT Hybrid
python src/cli.py train-hybrid-hgt \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --initial-window-days 365 \
    --step-days 42 \
    --epochs 30
```

### Compare Results

```bash
# Generate comparison report
python scripts/compare_model_results.py \
    --baseline-run-id <RUN_ID> \
    --sage-run-id <RUN_ID> \
    --hgt-run-id <RUN_ID> \
    --output-dir artifacts/comparison_results/<timestamp>

# Validate signatures
python src/utils/signature_comparison.py \
    --experiment-name "exp13-baseline-vs-hybrid-365d" \
    --model-types baseline_graph hybrid_sage hybrid_hgt
```

## Files Generated

```
artifacts/comparison_results/<timestamp>/
├── baseline_run_id.txt
├── sage_run_id.txt
├── hgt_run_id.txt
├── comparison_report.txt          # Main results
├── signature_comparison.txt       # Signature validation
├── metrics_comparison.csv        # Data for analysis
└── model_metadata.csv             # Model info
```

## Troubleshooting

**Models have incompatible signatures?**
→ Check feature categories match, regenerate features if needed

**Can't find run IDs?**
→ Use `mlflow runs search --experiment-names "exp13-baseline-vs-hybrid-365d"`

**GPU out of memory?**
→ Reduce batch size or embedding dimensions

## Next Steps

After comparison:
1. Document results in `docs/experiment_journal.md`
2. Choose direction (Baseline vs Hybrid)
3. Follow Experiment 13A or 13B plan

---

For detailed instructions, see: `docs/experiment_13_setup_guide.md`

