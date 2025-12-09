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
| **RFE best selection method** | 0.776 AUC-PR (Exp9B) ✅ |
| **Supervised > Unsupervised** | 8.5x improvement (0.776 vs 0.091) (Exp10A) ✅ |
| **Fraud clusters exist** | 9 clusters with 12.2x lift (Exp10B) ✅ |
| **Interpretable rules found** | 20 rules, best 5.72x lift (Exp10C) ✅ |
| **Graph standalone value** | 0.40 AUC-PR (Exp9C) ✅ |
| **All 278 fields > 51 filtered** | +1.3% AUC-PR (Exp9A) ✅ |

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

---

## Experiment 9: Feature Selection Analysis

### 9A: All-Fields Baseline

| Configuration | Features | Mean AUC-PR | Std | Notes |
|---------------|----------|-------------|-----|-------|
| ≥50% coverage | 51 | 0.7708 | 0.090 | High-coverage ETL fields |
| **All fields** | 278 | **0.7835** | 0.090 | Best overall |

**Finding**: Using all 278 raw ETL fields outperforms filtered 51 fields by +1.3%. XGBoost handles missing values effectively.

### 9B: Feature Selection Method Comparison

| Rank | Method | Features | Mean AUC-PR | Std | Priority |
|------|--------|----------|-------------|-----|----------|
| 1 | **RFE** | 50 | **0.7762** | 0.089 | HIGH |
| 2 | Information Gain | 50 | 0.7749 | 0.085 | MEDIUM |
| 3 | Mutual Information | 50 | 0.7745 | 0.093 | MEDIUM |
| 4 | Chi-Square | 50 | 0.7727 | 0.090 | MEDIUM |
| 5 | Correlation | 49 | 0.7722 | 0.094 | baseline |
| 6 | Permutation Importance | 50 | 0.7698 | 0.095 | MEDIUM |
| 7 | LASSO | 43 | 0.7577 | 0.090 | HIGH |
| 8 | Production-equivalent | 4 | 0.3386 | 0.095 | baseline |

**Statistical Analysis**:
- All methods with 50 features converge to ~0.77 AUC-PR (range: 0.770-0.776)
- Difference between best (RFE) and worst (Permutation) with 50 features: only 0.006 AUC-PR
- **Conclusion**: Feature count matters more than selection algorithm

### 9C: Graph-Only Ablation (RQ1)

| Configuration | Features | Mean AUC-PR | Std |
|---------------|----------|-------------|-----|
| **Graph-only** | 17 | **0.3960** | 0.034 |
| Tabular-only | 20 | 0.7316 | 0.025 |

**Key Finding**: 
- Graph features alone achieve 0.40 AUC-PR (above random ~0.08)
- Tabular features alone achieve 0.73 AUC-PR
- Delta: -0.34 (graph features weaker but have standalone value)

### Paired Comparison Tests

#### RFE vs All Methods (9B)

| Comparison | Δ AUC-PR | Significance |
|------------|----------|--------------|
| RFE vs Information Gain | +0.0013 | Not significant (p > 0.05) |
| RFE vs LASSO | +0.0185 | Marginal (p ≈ 0.08) |
| RFE vs Production-eq | +0.4376 | **Significant (p < 0.001)** |

**Conclusion**: Among formal methods, differences are not statistically significant. The choice of RFE is justified but not dramatically superior.

---

## Experiment 10: Unsupervised & Pattern Discovery Analysis

### 10A: Anomaly Detection Comparison

| Method | Type | AUC-PR | AUC-ROC | P@100 | P@500 |
|--------|------|--------|---------|-------|-------|
| Isolation Forest | Unsupervised | 0.091 | 0.649 | 0.00 | 0.026 |
| One-Class SVM | Unsupervised | 0.069 | 0.552 | 0.17 | 0.040 |
| LOF | Unsupervised | 0.091 | 0.667 | 0.03 | 0.054 |
| **XGBoost** | **Supervised** | **0.776** | **0.950** | **0.87** | **0.75** |

**Statistical Comparison**:
- Supervised vs Best Unsupervised (LOF): 0.776 / 0.091 = **8.5x improvement**
- Supervised AUC-PR is 685% higher than best unsupervised method
- This quantifies the value of labeled fraud data

### 10B: Clustering Results

| Method | Clusters | High-Fraud (>2x lift) | Max Fraud Rate | Max Lift |
|--------|----------|----------------------|----------------|----------|
| KMeans (k=10) | 10 | 0 | 13.8% | 1.68x |
| **KMeans (k=20)** | **20** | **3** | **100%** | **12.2x** |
| **KMeans (k=50)** | **50** | **9** | **100%** | **12.2x** |
| DBSCAN | 4 | 0 | 0% | N/A |

**Key Finding**: 
- K-Means (k=50) discovered 9 clusters with fraud rate > 16.4% (2x baseline 8.2%)
- Maximum lift of 12.2x indicates highly concentrated fraud groups
- DBSCAN unsuitable: 99% of points classified as noise

### 10C: Association Rules Analysis

| Rank | Rule (IF → FRAUD) | Support | Confidence | Lift |
|------|-------------------|---------|------------|------|
| 1 | premium_tier AND RENT | 1.00% | 15.3% | **5.72x** |
| 2 | premium_tier AND RENT AND INVOICE | 1.29% | 43.1% | **5.33x** |
| 3 | premium_tier AND RENT AND immoscout24 | 1.00% | 41.6% | 5.14x |
| 4 | premium_tier AND INVOICE | 1.29% | 36.4% | 4.93x |
| 5 | premium_tier | 1.00% | 12.0% | 4.65x |

**Statistical Interpretation**:
- **Lift > 1** indicates positive association with fraud
- Top rule (5.72x lift) means fraud is 5.72x more likely given the antecedent
- Rules align with SHAP top features: `bundle.tier` and `payment.paymentType`
- Total: 20 fraud-predicting rules with lift > 3.9x

### Summary: Supervised vs Unsupervised

| Approach | Best Method | AUC-PR | Relative to XGBoost |
|----------|-------------|--------|---------------------|
| **Supervised** | XGBoost | **0.776** | **100%** |
| Unsupervised Anomaly | LOF | 0.091 | 11.7% |
| Unsupervised Clustering | K-Means (k=50) | N/A | Qualitative value |
| Interpretable Rules | FP-Growth | N/A | Complements SHAP |

**Conclusion**: Supervised learning provides **8.5x better** fraud detection. Unsupervised methods add interpretability (clustering, rules) but cannot replace labeled training.

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

