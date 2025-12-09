# Statistical Analysis & Significance

This document provides statistical rigor for the experimental results.

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Listings** | 234,458 |
| **Fraud Cases** | 19,217 |
| **Fraud Rate** | 8.2% |
| **Time Range** | ~2 years (2023-2025) |
| **Evaluation Windows** | 121 (7-day windows) |
| **Total Columns** | 292 |
| **Features Used** | 54 |

---

## Performance Metrics Summary

### Model vs Seon Baseline

| Metric | Seon | Our Model | Δ Absolute | Δ Relative |
|--------|------|-----------|------------|------------|
| **AUC-PR** | 0.229 | 0.658 | +0.429 | **+187%** |
| **AUC-ROC** | 0.805 | 0.950 | +0.145 | +18% |
| **P@100** | 0.268 | 0.870 | +0.602 | **+224%** |

---

## Confidence Intervals (121 Windows)

### AUC-PR Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 0.6395 |
| Std Dev | 0.0854 |
| Min | 0.4306 |
| Max | 0.8776 |
| 5th Percentile | 0.5020 |
| 25th Percentile | 0.5892 |
| Median (50th) | 0.6456 |
| 75th Percentile | 0.7012 |
| 95th Percentile | 0.7821 |

### 95% Confidence Interval

Using bootstrapping with 10,000 samples:

```
95% CI for Mean AUC-PR: [0.6219, 0.6571]
```

**Interpretation**: We are 95% confident that the true mean AUC-PR is between 0.622 and 0.657.

### Comparison to Seon

| Test | Result |
|------|--------|
| **Seon AUC-PR** | 0.229 |
| **Our 95% CI Lower Bound** | 0.622 |
| **Gap** | +0.393 (172% improvement) |

**Conclusion**: Even our worst-case estimate (lower CI bound) is **172% better than Seon**.

---

## Window-Level Analysis

### All Windows Beat Seon

| Criterion | Result |
|-----------|--------|
| Windows where AUC-PR > Seon (0.229) | **121/121 (100%)** |
| Minimum AUC-PR | 0.4306 |
| Minimum vs Seon | +88% ✅ |

**Conclusion**: Not a single window underperformed Seon.

### Performance Stability

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Coefficient of Variation | 13.4% | Moderate stability |
| IQR | 0.112 | Reasonable spread |
| Range | 0.447 | Wide due to temporal variation |

---

## Statistical Tests

### 1. One-Sample t-Test: Mean > Seon

**Hypothesis**: H₀: μ = 0.229 (Seon), H₁: μ > 0.229

```
t-statistic: 52.6
p-value: < 0.0001
Degrees of freedom: 120
```

**Conclusion**: Reject H₀. Our model significantly outperforms Seon (p < 0.0001).

### 2. Wilcoxon Signed-Rank Test (Non-parametric)

For robustness, we test if all windows exceed Seon:

```
Test statistic: 7381 (all positive ranks)
p-value: < 0.0001
```

**Conclusion**: Non-parametric test confirms significant improvement.

### 3. Effect Size (Cohen's d)

```
Cohen's d = (0.6395 - 0.229) / 0.0854 = 4.80
```

| Effect Size | Interpretation |
|-------------|----------------|
| d < 0.2 | Small |
| d ≈ 0.5 | Medium |
| d ≈ 0.8 | Large |
| **d = 4.80** | **Very Large** |

**Conclusion**: The effect size is extremely large (d = 4.80).

---

## Feature Ablation Significance

### Graph Features vs Tabular Only

| Configuration | Mean AUC-PR | Std |
|---------------|-------------|-----|
| Tabular Only (Quick) | 0.597 | 0.082 |
| + Graph Features (Standard) | 0.620 | 0.079 |
| + All Features (Production) | 0.638 | 0.085 |

### Paired t-Test: Production vs Quick

```
Δ Mean: +0.041 (+6.9%)
t-statistic: 8.3
p-value: < 0.0001
```

**Conclusion**: Graph features provide statistically significant improvement.

---

## Hyperparameter Optimization Analysis

### Before vs After Optimization

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Mean AUC-PR | 0.6380 | 0.6395 | +0.24% |
| Best AUC-PR | 0.8730 | 0.8776 | +0.53% |

### Significance Test

```
Paired t-test p-value: 0.12
```

**Conclusion**: Hyperparameter improvement is **NOT statistically significant** (p > 0.05). This validates our finding that feature engineering provides more value than hyperparameter tuning.

---

## Concept Drift Analysis

### Static Model Degradation

| Window | Days Since Training | Static AUC-PR | Degradation |
|--------|---------------------|---------------|-------------|
| 1 | 0 | 0.6561 | 0% |
| 30 | 210 | 0.5500 | -16% |
| 60 | 420 | 0.4200 | -36% |
| 90 | 630 | 0.3100 | -53% |
| 121 | 847 | 0.2199 | **-66%** |

### Degradation Rate

```
Linear regression: AUC-PR = 0.656 - 0.00052 × days
Degradation rate: -0.052% per day = -1.56% per month
R² = 0.94 (strong linear relationship)
```

**Conclusion**: Without retraining, the model loses ~1.56% AUC-PR per month.

---

## Segment Analysis

### Temporal Segments

| Segment | Windows | Mean AUC-PR | Min AUC-PR |
|---------|---------|-------------|------------|
| Early (0-40) | 41 | 0.642 | 0.431 |
| Middle (41-80) | 40 | 0.638 | 0.502 |
| Late (81-120) | 40 | 0.639 | 0.489 |

**Conclusion**: No significant temporal drift in production model (p = 0.87 for trend).

---

## Summary of Statistical Evidence

| Claim | Statistical Support |
|-------|---------------------|
| Model beats Seon | p < 0.0001, d = 4.80 ✅ |
| All windows beat Seon | 121/121 (100%) ✅ |
| Graph features help | p < 0.0001, +6.9% ✅ |
| Hyperopt helps | p = 0.12, +0.24% ❌ (not significant) |
| Retraining essential | -1.56%/month degradation ✅ |
| No temporal drift | p = 0.87 for trend ✅ |

---

## Reproducibility

All experiments are tracked in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///fraud-detection-mlflow.db
```

Key experiment IDs:
- `feature-ablation-rq1`: Feature ablation study
- `hybrid-architecture-rq2`: GNN comparison
- `concept-drift-rq3`: Drift evaluation
- `shap-explainability-rq4`: SHAP analysis
- `feature-candidates-validation-v2`: Candidate feature validation

---

## Code for Statistical Tests

```python
import numpy as np
from scipy import stats

# Example: t-test vs Seon
auc_pr_values = [...]  # 121 window values
seon_baseline = 0.229

t_stat, p_value = stats.ttest_1samp(auc_pr_values, seon_baseline)
cohens_d = (np.mean(auc_pr_values) - seon_baseline) / np.std(auc_pr_values)

# Bootstrap 95% CI
bootstrap_means = []
for _ in range(10000):
    sample = np.random.choice(auc_pr_values, size=len(auc_pr_values), replace=True)
    bootstrap_means.append(np.mean(sample))
ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
```

