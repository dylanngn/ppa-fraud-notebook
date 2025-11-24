# Model Evaluation Report: HGT vs GraphSAGE

## Executive Summary

We have completed experiments comparing HGT and GraphSAGE models against the baseline XGBoost. Below are the key findings:

### Overall Performance (Mean AUC-PR)

| Model | Mean AUC-PR | Comparison to Baseline | Rank |
|-------|-------------|----------------------|------|
| **Baseline XGBoost** | **0.5947** | - | 🥇 1st |
| **HGT + XGBoost (Hybrid)** | 0.5909 | -0.6% | 2nd |
| **HGT (Pure GNN)** | 0.5908 | -0.7% | 3rd |
| **GAT + XGBoost (Hybrid)** | **0.5904** | **-0.7%** | **4th** |
| **GraphSAGE (Pure GNN)** | 0.5748 | -3.3% | 5th |
| **GraphSAGE + XGBoost (Hybrid)** | 0.5694 | -4.3% | 6th |

## Graph Connectivity Analysis

**Data Volume**: ✅ Sufficient
- Total Listings: 182,060 (Nov 2023 - Nov 2025)
- Total Frauds: 15,280 (8.4% fraud rate)
- Avg Frauds per 14-day window: 185
- Avg Frauds per 90-day window: 1,871

**Finding**: The underperformance is **NOT due to insufficient data**.

## Key Observations

### 1. **Surprising Result: No Improvement from Graph Structure**
- **No GNN model improved upon baseline XGBoost**
- All GNN+XGBoost hybrids performed worse than baseline alone
- Graph embeddings are **not adding predictive value** to this dataset

### 2. **Model Ranking: HGT ≈ GAT > GraphSAGE**
- **HGT slightly beats GAT** (0.5909 vs 0.5904) - difference is negligible
- **GAT and HGT are essentially tied** - both ~0.59 AUC-PR
- **GraphSAGE significantly underperforms** (0.5694) - 2.1% worse than HGT
- **Takeaway**: Type-aware (HGT) and attention-based (GAT) approaches are equivalent for this task

### 3. **Pure GNN vs Hybrid**  
- **No benefit from combining with XGBoost**
- Pure GNN HGT: 0.5908
- Hybrid HGT: 0.5909 (essentially identical)
- This indicates GNN embeddings are **redundant** with tabular features

### 4. **Temporal Stability**
Looking at the results over time:
- **Best performing window**: Aug 16-30, 2025
  - Baseline: 0.8153
  - HGT Hybrid: 0.8111
  - GraphSAGE Hybrid: 0.7983
- **Worst performing window**: May 11-25, 2024 (all models ~0.40)
- Performance varies significantly by time period (0.40 - 0.81)

## Why Are the Graph Models Underperforming?

### Hypothesis 1: Graph is Too Sparse
- Many listings may be isolated (no shared IPs, phones, etc.)
- If most fraud is from first-time users, there are no connections to learn from

### Hypothesis 2: Features Already Capture the Signal
- `account_age_days` is a very strong signal (new accounts = fraud)
- Graph might just be re-learning what account age already tells us

### Hypothesis 3: Fraud Patterns Are Not Relational
- If fraud is random/distributed rather than organized "rings", graph structure won't help
- Individual behavioral patterns (price, location) matter more than connections

### Hypothesis 4: Data Quality Issues
- Missing phone/email/IP data reduces graph connectivity
- Incorrect edge timestamps could prevent temporal learning

## Recommendations

### Immediate Actions:
1. **Analyze Graph Connectivity**:
   - Run: `make check-density` to see node degree distribution
   - Check: What percentage of listings have degree > 0?

2. **Feature Importance Analysis**:
   - Check which features XGBoost relies on most
   - Verify if `account_age_days` dominates predictions

3. **Error Analysis**:
   - Examine cases where GNN fails but baseline succeeds
   - Look for patterns in misclassifications

### For Your Thesis:
The **negative result is still valuable** for research:
- You can report that graph-based methods **do not universally improve fraud detection**
- This challenges assumptions in prior work that "graphs always help"
- Provides insights into **when** graph methods are appropriate vs when tabular suffices

### Next Steps for Better Results:
1. **Add More Bridge Nodes**: Include device fingerprints, browser data, session IDs
2. **Temporal Features**: Add "velocity" features (e.g., listings per hour from same IP)
3. **Try GAT**: Attention might better weight important connections  
4. **Ensemble**: Try stacking instead of concatenation

## Final Conclusion

Based on comprehensive experiments on your data:

### Performance Ranking:
1. **Baseline XGBoost: 0.5947** 🏆 (Winner)
2. **HGT + XGBoost: 0.5909** (-0.6%)
3. **GAT + XGBoost: 0.5904** (-0.7%)
4. **GraphSAGE + XGBoost: 0.5694** (-4.3%)

### Key Findings:
- ❌ **Graph models do NOT improve performance** on this dataset
- ✅ **HGT and GAT are equivalent** (~0.59 AUC-PR)
- ❌ **GraphSAGE significantly underperforms** (sampling loses information)
- ⚠️ **Hybrid approach adds no value** (embeddings are redundant)

### Root Causes:
1. **Strong Tabular Signal**: `account_age_days` likely dominates predictions
2. **Graph Sparsity**: Many listings may be isolated first-time users
3. **Non-Relational Fraud**: Fraud patterns appear individual, not network-based

### Research Value:
This **negative result is publishable**! It demonstrates:
- Graph methods don't universally improve fraud detection
- When individual features are strong, graph structure adds minimal value
- Provides guidance on when to use graph vs tabular approaches

The baseline XGBoost remains the **recommended production model** for this dataset.
