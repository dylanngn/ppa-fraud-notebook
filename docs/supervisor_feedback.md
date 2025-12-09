# Supervisor Feedback: Research Review

**Date**: 2025-12-09  
**Reviewer**: Research Supervisor  
**Project**: Fraud Detection in Online Real Estate Marketplaces Using a Hybrid Graph and Gradient Boosting Model

---

## Executive Summary

| Criterion | Grade | Notes |
|-----------|-------|-------|
| **Methodology** | B+ | Strong temporal evaluation, needs feature selection formalization |
| **Data Mining Alignment** | A- | Good core techniques, missing unsupervised methods |
| **RQ Coverage** | A- | All RQs addressed, minor gaps |
| **Reproducibility** | A | MLflow tracking, Hydra config excellent |
| **Documentation** | A | Experiment journal, architecture docs thorough |
| **Statistical Rigor** | A- | Good confidence intervals, missing feature selection stats |

**Verdict**: Solid applied data mining research. Address HIGH priority items before submission.

---

## 1. Alignment with Data Mining Topic

### Assessment: ✅ WELL ALIGNED

Your research is fundamentally a data mining project, specifically focused on **fraud detection** which is a classic application of data mining.

### Data Mining Elements Present

| Data Mining Component | Your Implementation | Status |
|----------------------|---------------------|--------|
| **Classification** | XGBoost binary classification (fraud/non-fraud) | ✅ Core |
| **Feature Engineering** | Handcrafted graph features, text features, time-weighted features | ✅ Strong |
| **Graph Mining** | PageRank, component analysis, shared contact patterns | ✅ Novel contribution |
| **Temporal Analysis** | Accumulating window training, concept drift evaluation | ✅ Strong |
| **Explainability** | SHAP analysis for feature importance | ✅ Strong |

### KDD Process Alignment

Your research follows the Knowledge Discovery in Databases (KDD) process:

```
Data Preparation → Feature Engineering → Modeling → Evaluation → Deployment Considerations
     ✅                   ✅                ✅           ✅              ✅
```

**No action required** - This aspect is well-covered.

---

## 2. Missing Data Mining Methods/Techniques

### Assessment: ⚠️ SEVERAL GAPS TO ADDRESS

### 2.1 Critical Missing: Unsupervised Anomaly Detection

| Priority | Technique | Why It's Important |
|----------|-----------|-------------------|
| 🔴 **HIGH** | Isolation Forest | Doesn't require labels; can detect novel fraud patterns not in training data |
| 🔴 **HIGH** | One-Class SVM | Models "normal" behavior; flags deviations as potential fraud |
| 🟡 **MEDIUM** | Local Outlier Factor (LOF) | Density-based; good for fraud in feature space |

**Action Required**:
```python
# Add to experiments/ as exp9_anomaly_detection.py
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def compare_anomaly_methods():
    """Compare unsupervised anomaly detection with supervised XGBoost."""
    methods = {
        'isolation_forest': IsolationForest(contamination=0.082),
        'one_class_svm': OneClassSVM(nu=0.082),
    }
    # Train on non-fraud only, evaluate on full test set
    # Compare AUC-PR, P@100 with your XGBoost model
```

**Rationale for Thesis**: You acknowledge "novel fraud patterns" as a limitation. Demonstrating that you considered unsupervised methods shows methodological completeness.

---

### 2.2 Missing: Clustering Analysis

| Priority | Technique | Application |
|----------|-----------|-------------|
| 🟡 **MEDIUM** | K-Means on embeddings | Discover fraud "rings" or behaviorally similar groups |
| 🟡 **MEDIUM** | DBSCAN | Density-based clustering for outlier detection |
| 🟢 **LOW** | Hierarchical clustering | Visualize fraud pattern hierarchy |

**Action Required**:
```python
# Add clustering analysis to strengthen RQ1 (relational indicators)
from sklearn.cluster import KMeans, DBSCAN

def cluster_fraud_patterns():
    """Cluster listings based on graph embeddings to discover fraud rings."""
    # Use SAGE embeddings (64-dim)
    embeddings = load_embeddings('artifacts/embeddings_sage.pt')
    
    # Cluster
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Analyze fraud rate per cluster
    # Report: "Cluster X has 45% fraud rate vs 8% baseline"
```

**Rationale**: This directly supports RQ1 by discovering **structural fraud patterns** in the graph.

---

### 2.3 Missing: Association Rule Mining

| Priority | Technique | Application |
|----------|-----------|-------------|
| 🟡 **MEDIUM** | Apriori / FP-Growth | Discover frequent fraud patterns |

**Action Required**:
```python
# Add association rule mining for interpretable patterns
from mlxtend.frequent_patterns import apriori, association_rules

def mine_fraud_rules():
    """Discover association rules like: 
    {invoice_payment, new_account, premium_tier} → fraud (confidence: 0.85)
    """
    # Binarize features
    fraud_df = df[df['is_fraud'] == 1]
    
    # Mine frequent itemsets
    frequent = apriori(fraud_df[binary_features], min_support=0.1)
    rules = association_rules(frequent, metric='confidence', min_threshold=0.7)
    
    # Top 10 rules by lift
    return rules.sort_values('lift', ascending=False).head(10)
```

**Rationale**: This provides **interpretable rules** for fraud analysts (strengthens RQ4) and complements SHAP analysis.

---

### 2.4 Missing: Formal Feature Selection

| Priority | Technique | Current Gap |
|----------|-----------|-------------|
| 🔴 **HIGH** | Recursive Feature Elimination (RFE) | Not applied |
| 🔴 **HIGH** | LASSO regularization | Not applied |
| 🟡 **MEDIUM** | Information Gain / Chi-Square | Not applied |
| 🟡 **MEDIUM** | Permutation Importance | Only SHAP used |

**This is addressed in detail in Section 4.**

---

### 2.5 Optional: Ensemble Method Comparison

| Priority | Technique | Notes |
|----------|-----------|-------|
| 🟢 **LOW** | LightGBM | Faster than XGBoost, similar performance |
| 🟢 **LOW** | CatBoost | Better categorical handling |
| 🟢 **LOW** | Random Forest | Classic baseline |

**Action**: Optional, but would strengthen the "why XGBoost" justification.

---

## 3. Research Question Alignment

### Assessment: ✅ MOSTLY ALIGNED, with gaps

### 3.1 RQ1: Novel Relational Indicators

> "What novel relational indicators of fraud can be identified in a real-estate marketplace dataset using graph-based analysis?"

| Aspect | Status | Evidence |
|--------|--------|----------|
| Graph features identified | ✅ | PageRank, component size, shared contacts |
| SHAP importance validated | ✅ | `listing_pagerank` is #2 feature |
| Comparison with non-graph | ⚠️ | Need isolated ablation |

**Gap**: Your Experiment 1 shows graph features add +3.9% AUC-PR, but you compare "base + graph" vs "base only". 

**Action Required**: Add explicit comparison:
```
| Configuration        | AUC-PR | Delta |
|---------------------|--------|-------|
| Tabular only        | 0.597  | -     |
| Graph only          | ???    | ???   |  ← MISSING
| Tabular + Graph     | 0.620  | +3.9% |
```

---

### 3.2 RQ2: Hybrid GNN + XGBoost Design

> "How can a hybrid GNN + XGBoost architecture be designed to effectively model these indicators for automated fraud detection?"

| Aspect | Status | Evidence |
|--------|--------|----------|
| SAGE vs HGT comparison | ✅ | SAGE +1.67%, HGT -0.24% |
| Handcrafted vs GNN | ✅ | Production (+2.90%) > SAGE (+1.67%) |
| Architecture explained | ✅ | Clear in architecture.md |

**Issue: Title Inconsistency**

Your topic says "Hybrid Graph and Gradient Boosting Model" but:
- SAGE Hybrid: +1.67% improvement
- HGT Hybrid: -0.24% (worse!)
- **Handcrafted graph features outperform GNN embeddings**

This means your best model is NOT a GNN hybrid, it's **XGBoost + handcrafted graph statistics**.

**Action Required** (choose one):

| Option | Action |
|--------|--------|
| **A. Reframe title** | Change to "Graph-Enhanced Gradient Boosting for Fraud Detection" |
| **B. Investigate GNN underperformance** | Add analysis of WHY GNN underperforms and report as finding |
| **C. Improve GNN** | Tune GNN hyperparameters, try different architectures (GAT, GIN) |

**Recommended**: Option B - Report this as an interesting finding. GNN underperformance in fraud detection (heterophily problem) is a valid research contribution.

---

### 3.3 RQ3: Periodic Retraining vs Concept Drift

> "To what extent does periodic retraining maintain the model's predictive performance against concept drift over sequential time windows?"

| Aspect | Status | Evidence |
|--------|--------|----------|
| Static vs accumulating | ✅ | -1.56%/month degradation without retraining |
| Quantified degradation | ✅ | 66.5% performance loss over 2 years |
| Retraining frequency | ✅ | Weekly recommended |

**No action required** - This is excellently addressed.

---

### 3.4 RQ4: XAI for Actionable Insights

> "How can Explainable AI (SHAP) translate the model's predictions into actionable insights for fraud analysts?"

| Aspect | Status | Evidence |
|--------|--------|----------|
| Global importance | ✅ | Top 20 features ranked |
| Local explanations | ✅ | Waterfall plots for cases |
| Actionable rules | ⚠️ | Could be more formalized |

**Action Required**: Formalize SHAP insights into decision rules:

```markdown
## Analyst Decision Support Rules (add to experiment_journal.md)

| Priority | Condition | Action | Confidence |
|----------|-----------|--------|------------|
| 🔴 HIGH | account_age_days < 7 AND payment_type = INVOICE | Manual review | 85% fraud |
| 🔴 HIGH | listing_pagerank = 0 AND user_listing_count = 1 | Check for isolation | 72% fraud |
| 🟡 MEDIUM | shared_contact_email_count > 5 | Investigate ring | 65% fraud |
```

This makes RQ4 more actionable.

---

## 4. Feature Selection: All Fields vs. Handpicking

### Assessment: ⚠️ INSUFFICIENT JUSTIFICATION

This is the **most significant methodological weakness** in your research.

### 4.1 Current State

| Metric | Value |
|--------|-------|
| Raw fields available | 292 |
| Fields with ≥50% coverage | 127 |
| Features currently used | 54 |
| Selection method | Ad-hoc (coverage + domain knowledge + experimentation) |

### 4.2 What's Missing

#### Problem 1: No Formal Feature Selection Algorithm

You haven't applied any of these systematically:
- Recursive Feature Elimination (RFE)
- LASSO regularization
- Information Gain
- Chi-Square test
- Mutual Information

#### Problem 2: Arbitrary Thresholds

Your Feature Discovery Pipeline uses:
- Correlation threshold: 0.1 (high-value)
- Correlation threshold: 0.05 (medium-value)
- Coverage threshold: 50%

**These are not justified statistically.**

#### Problem 3: High-Value Fields Ignored

Experiment 8C found unused fields with **higher correlation than many current features**:

| Field | Correlation | Coverage | Status |
|-------|-------------|----------|--------|
| `rent.interval` | **0.374** | 82% | ❌ Not used |
| `platforms` | **0.284** | 100% | ❌ Not used |
| `billing.language` | **0.263** | 98% | ❌ Not used |

For comparison, your current features:
- `listing_component_size`: 0.262 (used)
- `shared_contact_email_count`: 0.150 (used)

**Why is `rent.interval` (0.374) not used but `shared_contact_email_count` (0.150) is?**

#### Problem 4: "All Fields" Experiment Not Done

You have NOT tested:
1. Using all 127 fields with ≥50% coverage
2. Using all 292 fields (with proper imputation)
3. Comparing performance across selection methods

---

### 4.3 Required Action: Add Experiment 9

**Create**: `src/experiments/exp9_feature_selection.py`

```python
"""
Experiment 9: Feature Selection Methodology Comparison

Research Gap: Current feature selection is ad-hoc. This experiment
provides statistical justification for feature choices.
"""
import logging
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LassoCV
import xgboost as xgb

logger = logging.getLogger(__name__)

def exp9_feature_selection():
    """Compare feature selection methods."""
    
    # Load all 127 fields with ≥50% coverage
    all_features = get_all_available_features()  # 127 features
    
    methods = {
        # Method 1: Use ALL available fields
        "all_127_fields": {
            "features": all_features,
            "description": "All fields with ≥50% coverage"
        },
        
        # Method 2: Current production (manual selection)
        "production_manual_54": {
            "features": PRODUCTION_PROFILE_FEATURES,  # 54 features
            "description": "Current handpicked features"
        },
        
        # Method 3: RFE with XGBoost
        "rfe_top_50": {
            "features": rfe_select(all_features, n=50),
            "description": "Recursive Feature Elimination"
        },
        
        # Method 4: LASSO regularization
        "lasso_selected": {
            "features": lasso_select(all_features),
            "description": "LASSO non-zero coefficients"
        },
        
        # Method 5: Correlation-based (formalized)
        "correlation_top_50": {
            "features": correlation_select(all_features, n=50),
            "description": "Top 50 by fraud correlation"
        },
        
        # Method 6: Mutual Information
        "mutual_info_top_50": {
            "features": mutual_info_select(all_features, n=50),
            "description": "Top 50 by mutual information"
        },
    }
    
    results = []
    for name, config in methods.items():
        logger.info(f"Training with {name}: {len(config['features'])} features")
        
        # Train accumulating window
        metrics = train_and_evaluate(config['features'])
        
        results.append({
            'method': name,
            'n_features': len(config['features']),
            'mean_auc_pr': metrics['mean_auc_pr'],
            'best_auc_pr': metrics['best_auc_pr'],
            'p_at_100': metrics['mean_p_at_100'],
            'description': config['description'],
        })
    
    # Generate comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_auc_pr', ascending=False)
    
    return results_df
```

---

### 4.4 Expected Output

Add this table to `experiment_journal.md`:

```markdown
## Experiment 9: Feature Selection Comparison

| Method | Features | Mean AUC-PR | P@100 | Notes |
|--------|----------|-------------|-------|-------|
| All 127 fields | 127 | ? | ? | May overfit |
| Production (manual) | 54 | 0.658 | 0.870 | Current baseline |
| RFE top-50 | 50 | ? | ? | Algorithmic selection |
| LASSO selected | ? | ? | ? | Regularized selection |
| Correlation top-50 | 50 | ? | ? | Simple but interpretable |
| Mutual Info top-50 | 50 | ? | ? | Non-linear relationships |

**Conclusion**: [Your analysis of which method is best and why]
```

---

### 4.5 Why This Matters for Your Thesis

| Without Experiment 9 | With Experiment 9 |
|---------------------|-------------------|
| "We selected features based on coverage and domain knowledge" | "We compared 6 feature selection methods; RFE achieved highest AUC-PR of X, validating our choice of Y features" |
| Reviewer: "Why these 54 features?" | Reviewer: "Rigorous comparison, justified selection" |
| Methodological weakness | Methodological strength |

---

## 5. Priority Action Items

### 🔴 HIGH Priority (Required before submission)

| # | Action | File to Create/Modify | Estimated Effort |
|---|--------|----------------------|------------------|
| 1 | **Add Feature Selection Experiment** | `src/experiments/exp9_feature_selection.py` | 1-2 days |
| 2 | **Add Unsupervised Anomaly Detection** | `src/experiments/exp10_anomaly_detection.py` | 1 day |
| 3 | **Clarify "Hybrid" terminology** | `docs/experiment_journal.md` | 2 hours |

### 🟡 MEDIUM Priority (Recommended)

| # | Action | File to Create/Modify | Estimated Effort |
|---|--------|----------------------|------------------|
| 4 | Add Association Rule Mining | `src/experiments/exp11_association_rules.py` | 1 day |
| 5 | Add Clustering Analysis | `notebooks/experiment9_clustering.ipynb` | 0.5 days |
| 6 | Formalize SHAP into Decision Rules | `docs/experiment_journal.md` | 2 hours |
| 7 | Add "Graph Only" ablation for RQ1 | `notebooks/experiment1_extended.ipynb` | 0.5 days |

### 🟢 LOW Priority (Optional)

| # | Action | File to Create/Modify | Estimated Effort |
|---|--------|----------------------|------------------|
| 8 | Compare with LightGBM/CatBoost | `src/experiments/exp12_model_comparison.py` | 1 day |
| 9 | Improve GNN architecture | `src/models/gnn/` | 2-3 days |

---

## 6. Suggested Timeline

```
Week 1:
├── Day 1-2: Experiment 9 (Feature Selection) ← CRITICAL
├── Day 3: Experiment 10 (Anomaly Detection)
└── Day 4-5: Update documentation, clarify hybrid terminology

Week 2:
├── Day 1: Association Rule Mining
├── Day 2: Clustering Analysis
├── Day 3: Formalize SHAP rules + Graph-only ablation
└── Day 4-5: Final review, thesis writing
```

---

## 7. Checklist Before Submission

### Methodology Completeness

- [ ] Feature selection formally compared (Experiment 9)
- [ ] At least one unsupervised method tested (Experiment 10)
- [ ] "Hybrid" terminology clarified in documentation
- [ ] All 4 RQs have explicit experiment mapping

### Documentation

- [ ] `experiment_journal.md` updated with new experiments
- [ ] `statistical_analysis.md` includes feature selection stats
- [ ] `limitations.md` acknowledges why GNN underperforms

### Reproducibility

- [ ] All experiments tracked in MLflow
- [ ] Random seeds set for all experiments
- [ ] Requirements.txt up to date

---

## Summary

Your research is fundamentally sound with strong temporal evaluation, excellent reproducibility, and good RQ coverage. The main weaknesses are:

1. **Feature selection lacks formal justification** (highest priority)
2. **Missing unsupervised methods** (data mining completeness)
3. **Title-methodology alignment** regarding "hybrid" architecture

Address the HIGH priority items, and you'll have a well-rounded thesis ready for submission.

---

*Feedback prepared by Research Supervisor, 2025-12-09*

