# Seon Evaluation Guide

## Overview

**Seon** is the current production fraud detection system that our research aims to beat. This guide explains how to evaluate Seon's performance and compare it with our research models (XGBoost baseline and Hybrid GNN+XGBoost).

## Key Differences: Seon vs Research Models

| Aspect | Seon (Production) | Research Models |
|--------|-------------------|-----------------|
| **Decision Type** | Binary (approve/reject) | Probabilistic ranking |
| **Timing** | Pre-publication | Pre-publication |
| **Output** | Yes/No | Fraud probability 0-1 |
| **Use Case** | Automated gating | Manual review prioritization |
| **Metric Focus** | Precision, Recall, F1 | AUC-PR, P@100, Lift@100 |

## How Seon Works

1. **Before Publication**: Seon evaluates every listing submission
2. **Binary Decision**: 
   - `seonApproved=true` → Listing goes online
   - `seonApproved=false` → Listing blocked/flagged
3. **Post-Publication**: Users can report fraud, setting `fraud_flag`
4. **Ground Truth**: `fraud_flag` is our label (set weeks/months later)

## Evaluation Logic

### Primary Logic (when `seonApproved` available):

```
seonApproved=true  + fraud_flag=null  → TN (correct approval)
seonApproved=true  + fraud_flag!=null → FN (missed fraud)
seonApproved=false + fraud_flag!=null → TP (caught fraud)
seonApproved=false + fraud_flag=null  → FP (false alarm)
```

### Fallback Logic (when `seonApproved` missing):

```
first_published_date < fraud_flag     → FN (published then flagged)
fraud_flag!=null + published=null     → TP (blocked, was fraud)
fraud_flag=null + published!=null     → TN (published, no fraud)

Note: Cannot detect FP in fallback mode
```

## Quick Start

### 1. Evaluate Seon Performance

```bash
python -m src.cli evaluate-seon --window-days 90 --step-days 7
```

**Output:**
- `artifacts/results/seon_baseline_results.csv`
- Performance metrics per window
- Coverage statistics (% with Seon data)

### 2. Compare All Models

```bash
# First, train research models
python -m src.cli train-baseline
python -m src.cli train-hybrid --model hgt

# Then evaluate Seon
python -m src.cli evaluate-seon

# Finally, compare all
python -m src.cli compare-all-models
```

**Output:**
- `artifacts/results/comparison/all_models_comparison.csv`
- `artifacts/results/comparison/all_models_performance.png`
- `artifacts/results/comparison/all_models_trends.png`

## CLI Commands

### `evaluate-seon`

Evaluate Seon's performance using sliding window methodology (matching research models).

```bash
python -m src.cli evaluate-seon \
    --window-days 90 \
    --step-days 7 \
    --include-fallback true
```

**Options:**
- `--window-days`: Window size (default: 90)
- `--step-days`: Step size (default: 7)
- `--include-fallback`: Use fallback logic when Seon data missing (default: true)

**Metrics Calculated:**
- Precision, Recall, F1 Score
- True Positives, False Positives
- True Negatives, False Negatives
- Catch Rate (same as recall)
- False Alarm Rate
- Coverage (% with Seon data)

### `compare-all-models`

Generate comprehensive comparison across all models.

```bash
python -m src.cli compare-all-models \
    --window-days 90 \
    --step-days 7 \
    --output-dir artifacts/results/comparison
```

**Compares:**
1. **Seon (Production)** - Binary decisions, F1-focused
2. **XGBoost Baseline** - Ranked predictions, AUC-PR/P@100
3. **Hybrid Model** - GNN+XGBoost, AUC-PR/P@100

## Python API

### Evaluate Seon

```python
from src.models.evaluate_seon import run_seon_evaluation

results = run_seon_evaluation(
    window_days=90,
    step_days=14,
    include_fallback=True
)
```

### Parse Seon Approval

```python
from src.models.evaluate_seon import parse_seon_approval, calculate_seon_labels
import polars as pl

# Load data
df = pl.read_parquet("artifacts/nodes_listing.parquet")

# Extract Seon approval
df = df.with_columns([
    parse_seon_approval(pl.col("auto_approval_criteria")).alias("seon_approved")
])

# Calculate predictions
df = calculate_seon_labels(df)
```

## Understanding Seon Data Coverage

### Expected Coverage

- **High Coverage (>80%)**: Recent data, good Seon integration
- **Medium Coverage (50-80%)**: Some missing data, fallback used
- **Low Coverage (<50%)**: Limited Seon data, mostly fallback

### Fallback Limitations

When using `first_published_date` fallback:
- ✅ Can detect: TN, TP, FN
- ❌ Cannot detect: FP (no way to know if unpublished was legitimate)
- Result: **Fallback metrics may overestimate performance** (missing FPs)

### Recommendations

1. **Always report coverage** in comparisons
2. **Use `--include-fallback false`** for pure Seon evaluation
3. **Compare similar coverage periods** for fairness

## Interpreting Results

### Seon Performance Metrics

**Good Performance:**
- Precision > 0.80 (few false alarms)
- Recall > 0.70 (catches most fraud)
- F1 > 0.75 (balanced)

**Typical Trade-offs:**
- High precision → Low false alarms, might miss fraud
- High recall → Catches more fraud, more false alarms

### Comparison with Research Models

#### Metric Alignment

| Seon Metric | Research Equivalent | Interpretation |
|-------------|---------------------|----------------|
| Precision | P@K (e.g., P@100) | Accuracy in top predictions |
| Recall | Recall@K | % of fraud caught |
| F1 Score | - | Balanced performance |
| - | AUC-PR | Overall ranking quality |

#### When to Use Each Model

**Use Seon-style (binary) when:**
- Automated decisions required
- Clear approve/reject needed
- Explainability important
- Fast decisions required

**Use Research Models (ranking) when:**
- Manual review workflow
- Prioritizing limited resources
- Top-K matters more than all cases
- Continuous improvement focus

## Example Workflow

### Scenario: Evaluate if Research Beats Seon

```bash
# Step 1: Evaluate Seon (production baseline)
python -m src.cli evaluate-seon --window-days 90

# Step 2: Train baseline model
python -m src.cli train-baseline --window-days 90

# Step 3: Train hybrid model
python -m src.cli train-hybrid --model hgt --window-days 90

# Step 4: Compare all
python -m src.cli compare-all-models
```

**Review Output:**
1. Check `artifacts/results/comparison/all_models_comparison.csv`
2. Look at performance plots
3. Compare:
   - Seon F1 vs Baseline P@100
   - Coverage statistics
   - Temporal stability

### Expected Outcomes

**If Research Wins:**
- P@100 > Seon Precision
- Fewer false alarms at top of ranking
- Better prioritization for manual review
- **Decision**: Deploy research model for review queue

**If Seon Wins:**
- Seon F1 > Research metrics
- Better balance of precision/recall
- More suitable for automated decisions
- **Decision**: Keep Seon, use research as supplement

**If Hybrid Wins:**
- Hybrid > Baseline > Seon
- GNN embeddings add value
- Graph structure captures fraud patterns
- **Decision**: Deploy hybrid model

## Troubleshooting

### Low Coverage (<50%)

**Problem**: Most predictions use fallback logic

**Solutions:**
1. Check if `auto_approval_criteria` is populated in database
2. Verify date range (Seon might not have been deployed early on)
3. Use `--include-fallback false` to see pure Seon performance
4. Consider comparing only recent periods

### Seon Performance Seems Too Good

**Problem**: Suspiciously high metrics (>95% precision AND recall)

**Check:**
1. Coverage percentage (might be using lots of fallback)
2. Fallback doesn't count FPs → inflated metrics
3. Time range (might only have easy cases)
4. Test set size (might be too small)

### Cannot Compare Seon with Research Models

**Problem**: Different metrics make comparison difficult

**Solutions:**
1. Convert research models to binary decisions (threshold at 0.5)
2. Calculate Precision@K for Seon (if ranking available)
3. Use both for different use cases:
   - Seon: Automated blocking
   - Research: Manual review prioritization

## Advanced Topics

### Temporal Alignment

Seon operates in real-time, but `fraud_flag` is set later:
- Seon decision: Time T
- Fraud flag: Time T + days/weeks/months

**Implications:**
- Seon can only use info available at T
- Research models can use same info at T
- Fair comparison requires temporal awareness

### Threshold Calibration

To compare Seon (binary) with Research (probabilistic):

```python
# Convert research model to binary decisions
threshold = 0.5  # Adjust for precision/recall tradeoff
df = df.with_columns([
    (pl.col("fraud_probability") >= threshold).cast(pl.Int8).alias("binary_prediction")
])

# Now can calculate F1, same as Seon
```

### Cost-Sensitive Evaluation

Different errors have different costs:
- **False Positive**: Manual review time, user friction
- **False Negative**: Fraud loss, reputation damage

**Adjust thresholds accordingly:**
- High FP cost → Increase threshold (fewer FPs, more FNs)
- High FN cost → Decrease threshold (catch more fraud, more FPs)

## Files and Outputs

### Seon Evaluation Outputs

```
artifacts/results/
  seon_baseline_results.csv     # Per-window metrics
```

**Columns:**
- `window_idx`: Window number
- `window_start`, `window_end`: Time range
- `total_test`: Total test instances
- `evaluated`: Instances with predictions
- `seon_approved_available`: Count with Seon data
- `fallback_used`: Count using fallback logic
- `tp`, `tn`, `fp`, `fn`: Confusion matrix
- `precision`, `recall`, `f1_score`: Performance metrics
- `catch_rate`, `false_alarm_rate`: Fraud-specific metrics

### Comparison Outputs

```
artifacts/results/comparison/
  all_models_comparison.csv           # Summary table
  all_models_performance.png          # Bar charts
  all_models_trends.png               # Performance over time
```

## Key Takeaways

1. **Seon is the TRUE baseline** - it's what we need to beat
2. **Different use cases** - binary vs ranking
3. **Coverage matters** - check how much Seon data is available
4. **Fallback has limitations** - can't detect all FPs
5. **Temporal awareness** - ensure fair comparison
6. **Both have value** - automated (Seon) + prioritization (Research)

## Next Steps

After evaluation:

1. **If Research Wins**: Consider deployment for review prioritization
2. **If Seon Wins**: Analyze why (SHAP), improve features
3. **If Hybrid Wins**: Deploy hybrid approach
4. **If Close**: Use both (Seon for automation, Research for review)

For deep analysis, use SHAP to understand feature importance differences between Seon and research models.

