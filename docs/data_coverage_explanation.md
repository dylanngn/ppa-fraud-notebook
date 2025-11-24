# Data Coverage and Training Window Explanation

## Issue 1: Data Range

### Current Status ✅
Your data **DOES** include the full Jan 2023 - Nov 2025 range:
- **Actual Range**: Jan 1, 2023 - Nov 1, 2025 (34 months)
- **Total Listings**: 237,695
- **Used in Models**: 237,695 (all data is being used)

### Why the Report Said "Nov 2023 - Nov 2025"?
The density check script had a hardcoded filter:
```python
start_date = datetime(2023, 11, 1)  # Was filtering out Jan-Oct 2023!
```

**Action Required**: ✅ The models ARE using all data from Jan 2023. The density check output was misleading but the actual training uses the full dataset.

---

## Issue 2: 90-Day Training Window - Are We Wasting Data?

### Short Answer: **No, we're using ALL the data efficiently!**

### How Sliding Window Works:

```
Time →
├─────────────────────────────────────────────────────────────────┤
Jan 2023                                                    Nov 2025

Training Windows (90 days each):
┌────────────┐← Train ┬─Test→┐               Window 1
  ┌────────────┐← Train ┬─Test→┐             Window 2
    ┌────────────┐← Train ┬─Test→┐           Window 3
      ┌────────────┐← Train ┬─Test→┐         Window 4
                     ... (continues)
                            ┌────────────┐← Train ┬─Test→┐  Final Window
```

### Example:
- **Window 1**: Train on Jan 1 - Mar 31 (90d) → Test on Apr 1 - Apr 14 (14d)
- **Window 2**: Train on Jan 15 - Apr 14 (90d) → Test on Apr 15 - Apr 28 (14d)
- **Window 3**: Train on Jan 29 - Apr 28 (90d) → Test on Apr 29 - May 12 (14d)
- ... continues sliding until Nov 2025

### Data Utilization:
- **Not Wasted**: Every listing from Jan 2023 - Nov 2025 is used
- **Multiple Times**: Each listing appears in multiple training windows (up to 6-7 times)
- **Proper Temporal Ordering**: We never train on future data to predict the past

### Why 90 Days?
1. **Concept Drift**: Fraud patterns change. Too much old data = learning outdated patterns
2. **Recency Bias**: Recent fraud is more similar to current fraud
3. **Computational**: 90-day windows train in ~10-15 min. 365-day windows would be slower
4. **Industry Standard**: Most fraud detection systems retrain weekly/monthly on recent data

### Could We Try Larger Windows?
**Yes!** We can experiment:
```bash
# Try 180-day training windows
python src/cli.py train-baseline --window-days 180 --step-days 14
```

**Trade-offs**:
- ✅ More historical context
- ✅ More training data per window
- ❌ Risk of concept drift (old fraud patterns may not apply)
- ❌ Longer training time

---

## Issue 3: Graph Sparsity Hypothesis

### Analysis Notebook Created ✅
**Path**: `notebooks/04_graph_sparsity_analysis.ipynb`

This notebook will answer:
1. **What percentage of listings are isolated** (no connections)?
2. **Are fraud listings more/less connected** than legitimate ones?
3. **Do fraud rings exist** (groups of connected fraud listings)?
4. **Which edge types are most common** (IP, email, phone)?

### How to Run:
```bash
jupyter notebook notebooks/04_graph_sparsity_analysis.ipynb
```

### Expected Findings:
If we find:
- **>50% isolated listings** → Graph is sparse, explains GNN underperformance
- **Fraud = Similar connectivity to legit** → No relational signal
- **Few/no fraud rings** → Fraud is individual, not network-based

---

## Recommendations

### Immediate:
1. ✅ **Run the sparsity notebook** to confirm/reject the sparse graph hypothesis
2. 🔄 **Update density check script** to show full Jan 2023 - Nov 2025 range (optional, cosmetic)

### Optional Experiments:
1. **Larger Training Windows**:
   ```bash
   python src/cli.py train-baseline --window-days 180
   python src/cli.py train-hybrid --model hgt --window-days 180
   ```

2. **Feature Importance**:
   ```python
   # After training baseline
   import xgboost as xgb
   model = xgb.Booster()
   model.load_model("artifacts/model_baseline.json")
   importance = model.get_score(importance_type='weight')
   # Check if account_age_days dominates
   ```

3. **Richer Graph** (if sparse):
   - Add device fingerprints
   - Add browser/OS data
   - Add session IDs

---

## Summary

| Question | Answer |
|----------|--------|
| **Is data missing?** | ❌ No, full Jan 2023 - Nov 2025 is used (237K listings) |
| **Are we wasting data?** | ❌ No, sliding window uses all data efficiently |
| **90 days enough?** | ✅ Yes, balances recency vs volume. Can experiment with 180d |
| **Graph too sparse?** | ❓ Run notebook to confirm |

**Next Step**: Run `notebooks/04_graph_sparsity_analysis.ipynb` to diagnose the root cause of GNN underperformance.
