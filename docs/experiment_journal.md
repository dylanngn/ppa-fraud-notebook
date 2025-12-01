# Experiment Journal

## Research Overview

**Topic**: Fraud Detection in Online Real Estate Marketplaces Using a Hybrid Graph and Gradient Boosting Model

**Research Questions** (aligned with thesis proposal):

1. **RQ1**: What novel relational indicators of fraud can be identified in a real-estate marketplace dataset using graph-based analysis?
2. **RQ2**: How can a hybrid GNN + XGBoost architecture be designed to effectively model these indicators for automated fraud detection?
3. **RQ3**: To what extent does periodic retraining maintain the model's predictive performance against concept drift over sequential time windows?
4. **RQ4**: How can Explainable AI (SHAP) translate the model's predictions into actionable, human-understandable insights for fraud analysts?

**Practical Consideration** (complements RQ2):
- What is the optimal balance between model complexity, training time, and predictive performance for production deployment?

**Core Framework**: Hybrid GNN-XGBoost with Accumulating Window Training
- XGBoost serves as the final decision-maker
- Features can come from: (a) tabular data, (b) handcrafted graph statistics, or (c) GNN embeddings
- Accumulating window training serves as passive concept drift adaptation
- SHAP provides both global feature importance and local prediction explanations

---

## Project Setup

### Configuration Framework (Hydra + MLflow)

```yaml
# conf/config.yaml - Main configuration
defaults:
  - data: default
  - features: baseline      # Options: quick, standard, production, ablation
  - model: xgboost

experiment_name: "ppa-fraud-detection"
seed: 42
```

### Execution Commands

All commands use Hydra for configuration. Run from project root.

```bash
# Data Pipeline (run once)
python -m src.data.pipeline                    # Extract, transform, load data
python -m src.data.graph.create_artifacts      # Create graph artifacts (nodes/edges parquet)
python -m src.utils.evaluate_seon              # Generate static Seon baseline (one-time)

# Model Training (all use accumulating windows + MLflow tracking)
python -m src.models.train features=production # XGBoost with production features
python -m src.models.train features=quick      # XGBoost with core features (fast iteration)
python -m src.models.gnn.sage                  # SAGE GNN hybrid (builds graph.pt if needed)
python -m src.models.gnn.hgt                   # HGT GNN hybrid (builds graph.pt if needed)

# Hyperparameter Optimization (Optuna-based)
python -m src.models.hyperopt.xgboost                    # Optimize XGBoost params
python -m src.models.hyperopt.pytorch gnn.model_type=sage # Optimize SAGE GNN params
python -m src.models.hyperopt.pytorch gnn.model_type=hgt  # Optimize HGT GNN params

# Analysis
python -m src.utils.data_quality_report        # Generate coverage analysis
```

### MLflow Tracking

All experiments are automatically tracked in MLflow:
- Parent run: Overall experiment
- Nested runs: Per-window evaluations
- Artifacts: Models, embeddings, feature importance

```bash
# Compare candidate model vs production
python -m src.utils.mlflow_model_comparison compare \
  --model-name fraud-detection-xgboost \
  --candidate-run-id <RUN_ID>

# Get deployment recommendation
python -m src.utils.mlflow_model_comparison recommend \
  --model-name fraud-detection-xgboost \
  --candidate-run-id <RUN_ID>

# Analyze model drift
python -m src.utils.mlflow_model_comparison drift \
  --model-name fraud-detection-xgboost
```

---

## Experiment 0: Feature & Graph Structure Validation

**Date**: 2025-11-30  
**Objective**: Validate feature cleanup and graph simplification changes  
**Status**: ✅ IMPLEMENTED (Ready for Validation)

### Background

Based on data quality analysis (`data_quality_reports/`), we implemented:

1. **Graph Simplification**: Merged phone edges from 2 → 1 (unified billing+lister phone)
2. **Feature Groups**: Organized features into explicit groups for ablation (see `constants.py`)
3. **Boolean Semantics**: Discovered NULL = FALSE semantics for boolean fields (100% semantic coverage)

### Changes Implemented

| Component | Change | Rationale |
|-----------|--------|-----------|
| `FeatureProcessor` | Added `from_config()` + `include_groups` support | Explicit feature selection (no silent failures) |
| `constants.py` | Added `FEATURE_GROUPS` dict | Define testable feature groups |
| `create_artifacts.py` | Unified phone edge (coalesce billing→lister) | Reduce edge types from 9 → 8, maximize coverage |
| `graph_builder.py` | Updated to use single `has_phone` edge | Simplified graph structure |
| `conf/features/*.yaml` | Created quick/standard/production/ablation profiles | Enable systematic experiments |

### Validation Steps

```bash
# Step 1: Rebuild graph artifacts (parquet files only)
python -m src.data.graph.create_artifacts

# Step 2: Verify parquet artifacts exist
ls -la artifacts/nodes_*.parquet artifacts/edges_*.parquet

# Step 3: Build PyG graph and verify structure
python -c "
from src.data.graph.graph_builder import build_graph
data = build_graph()  # Creates artifacts/graph.pt
print('Edge types:', data.edge_types)
print('Phone edges:', data['listing', 'has_phone', 'phone'].edge_index.shape)
"

# Step 4: Train with different feature profiles
python -m src.models.train features=quick      # core_numerical + boolean_high
python -m src.models.train features=production # All feature groups
```

### Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Edge types | 9 | 8 | -1 (phone merge) |
| Phone edge coverage | 70-98% | ~99% | +29% (coalesce) |
| Boolean features | 4 | 12 | +8 |
| Feature groups | N/A | 12 | Explicit group definitions |
| AUC-PR | ~0.70 | ~0.70 | No regression expected |

### Decision Branching

Based on validation results:

- **If AUC-PR drops > 2%**: Revert phone edge merge, investigate
- **If AUC-PR stable**: Proceed with simplified structure for all experiments
- **If AUC-PR improves**: Document as finding, update baseline

---

## Experiment 1: Feature Ablation Study (RQ1)

**Date**: TBD  
**Objective**: Identify which feature groups contribute to fraud detection performance  
**Status**: 🔄 PLANNED  
**Answers**: RQ1 (novel relational indicators)

### Hypothesis

Graph-derived features provide significant improvement over tabular-only baselines, demonstrating the value of relational indicators for fraud detection.

### Feature Tiers

| Tier | Features | Coverage | Description |
|------|----------|----------|-------------|
| **Tier 1a** | `account_age_days`, `payment_type`, `bundle_tier`, `log_price`, `latitude/longitude`, `offer_type`, `living_space`, `rooms` | >80% | Core numerical features |
| **Tier 1b** | `has_balcony`, `has_parking`, `has_elevator`, etc. | 100% (semantic) | Boolean indicators (NULL = FALSE) |
| **Tier 2** | `shared_contact_email_count`, `listing_component_size`, `listing_pagerank`, etc. | Computed | **Handcrafted graph statistics** |
| **Tier 3** | `email_time_spread`, `email_recency_weighted`, text features | Computed | Time-weighted & text features |

### Methodology

```bash
# Tier 1 only (no graph features) - Baseline
python -m src.models.train features=quick

# Tier 1 + Tier 2 (add graph features)
python -m src.models.train features=standard

# Full production (all tiers)
python -m src.models.train features=production

# Boolean ablation (optional sub-experiment)
python -m src.models.train features=ablation +features.experiment=no_booleans
python -m src.models.train features=ablation +features.experiment=tier1_with_boolean_all
```

### Expected Results

| Configuration | Expected AUC-PR | Delta | Key Insight |
|---------------|-----------------|-------|-------------|
| Tier 1 only (tabular) | ~0.59 | - | Baseline without graph |
| + Boolean features | ~0.61 | +2% | Boolean indicators add moderate value |
| + Graph features (Tier 2) | ~0.67 | +6% | **Graph features are critical** |
| + Time/Text (Tier 3) | ~0.70 | +3% | Enhanced features provide lift |

### Analysis for RQ1

Document which **graph-derived features** have highest SHAP importance:
- `shared_contact_email_count` → Fraud ring detection
- `listing_component_size` → Network connectivity
- `user_listing_count` → User behavior patterns

### Results
*To be filled after experiment*

### Decision Branching

- **If Tier 2 adds < 5% AUC-PR**: Investigate graph feature computation
- **If Tier 2 adds > 8% AUC-PR**: Graph features confirmed as primary contribution
- **If Tier 3 adds < 1% AUC-PR**: Consider removing for simplicity

---

## Experiment 2: Hybrid Architecture Comparison (RQ2)

**Date**: TBD  
**Objective**: Compare hybrid GNN-XGBoost architecture against alternatives  
**Status**: 🔄 PLANNED  
**Answers**: RQ2 (hybrid architecture design)

### Research Question

How does a hybrid GNN + XGBoost architecture compare to:
- XGBoost with handcrafted graph features (no GNN)
- Pure GNN approaches
- Production baseline (Seon)

### Models to Compare

| Model | Architecture | Features | Category |
|-------|--------------|----------|----------|
| **Seon** | Rule-based | Heuristics | Production Baseline |
| **XGBoost (tabular)** | XGBoost | Tabular only | Ablation Baseline |
| **XGBoost (+ graph)** | XGBoost | Tabular + Handcrafted Graph | Main Model |
| **SAGE Hybrid** | SAGE → XGBoost | Tabular + 64-dim embeddings | GNN Hybrid |
| **HGT Hybrid** | HGT → XGBoost | Tabular + 64-dim embeddings | GNN Hybrid |

### Methodology

```bash
# Baseline: XGBoost with tabular only
python -m src.models.train features=quick

# Main model: XGBoost with handcrafted graph features
python -m src.models.train features=production

# GNN Hybrid: SAGE embeddings + XGBoost
python -m src.models.gnn.sage

# GNN Hybrid: HGT embeddings + XGBoost
python -m src.models.gnn.hgt
```

### Evaluation Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| **Performance** | AUC-PR, P@100, F1 | Higher is better |
| **Training Time** | Wall-clock minutes | Lower is better |
| **Complexity** | GPU required, dependencies | Lower is better |
| **Interpretability** | SHAP compatibility | Full is better |

### Expected Results

| Model | AUC-PR | P@100 | Training Time | GPU |
|-------|--------|-------|---------------|-----|
| Seon (Prod) | ~0.50 | ~0.60 | N/A | No |
| XGBoost (tabular) | ~0.59 | ~0.65 | ~10 min | No |
| XGBoost (+ graph) | ~0.70 | ~0.77 | ~15 min | No |
| SAGE Hybrid | ~0.68 | ~0.75 | ~2-3 hrs | Yes |
| HGT Hybrid | ~0.68 | ~0.75 | ~2-3 hrs | Yes |

### Analysis for RQ2

1. **Performance Gap**: If handcrafted ≈ GNN embeddings, handcrafted wins on simplicity
2. **Efficiency**: Training time × GPU cost vs performance improvement
3. **Interpretability**: Handcrafted features are directly interpretable via SHAP

### Results
*To be filled after experiment*

### Decision Branching

- **If GNN outperforms by > 3%**: Consider GNN for production
- **If GNN matches or underperforms**: Use handcrafted features, document GNN limitations
- **If training time > 4 hrs**: GNN not viable for frequent retraining

---

## Experiment 3: Hyperparameter Optimization

**Date**: TBD  
**Objective**: Optimize XGBoost hyperparameters for fraud detection  
**Status**: 🔄 PLANNED

### Methodology

```bash
# Run hyperparameter optimization (Optuna-based)
python -m src.models.hyperopt.xgboost

# Optional: Configure optimization parameters
python -m src.models.hyperopt.xgboost hyperopt.n_trials=50 hyperopt.n_windows=5
```

### Search Space

```python
search_space = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_child_weight': [1, 5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0.0, 0.1, 0.4],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [1.0, 3.0, 5.0, 10.0],
}
```

### Optimization Objective

```python
# Weighted objective balancing ranking quality and top-k precision
objective = 0.7 * auc_pr + 0.3 * precision_at_100
```

### Success Criteria
- AUC-PR improvement > 1% over default parameters
- Target: Push to **0.72+ AUC-PR**

### Results
*To be filled after experiment*

---

## Experiment 4: Concept Drift & Retraining Evaluation (RQ3)

**Date**: TBD  
**Objective**: Validate that periodic retraining mitigates concept drift  
**Status**: 🔄 PLANNED  
**Answers**: RQ3 (periodic retraining effectiveness)

### Research Question

To what extent does accumulating window (periodic retraining) maintain model performance compared to a static model that degrades over time?

### Methodology

**Part A: Accumulating Window (WITH Retraining)**
```
Window 1: Train [Month 1-3] → Test [Month 4] → AUC-PR₁
Window 2: Train [Month 1-4] → Test [Month 5] → AUC-PR₂  (model updated)
Window 3: Train [Month 1-5] → Test [Month 6] → AUC-PR₃  (model updated)
...
```

**Part B: Static Model (WITHOUT Retraining)**
```
Static: Train [Month 1-3] → Test [Month 4] → AUC-PR₁
Static: Use Month 3 model → Test [Month 5] → AUC-PR₂  (same model, no update)
Static: Use Month 3 model → Test [Month 6] → AUC-PR₃  (same model, no update)
...
```

```bash
# Part A: Normal accumulating window training
python -m src.models.train features=production

# Part B: Train once, evaluate on all future windows (manual analysis)
# Use MLflow to compare window-by-window performance
```

### Expected Results

| Window | Test Period | WITH Retraining | WITHOUT Retraining | Degradation |
|--------|-------------|-----------------|-------------------|-------------|
| 1 | Month 4 | 0.70 | 0.70 | 0% |
| 2 | Month 5 | 0.69 | 0.67 | -3% |
| 3 | Month 6 | 0.69 | 0.64 | -6% |
| 4 | Month 7 | 0.68 | 0.60 | -10% |
| 5 | Month 8 | 0.68 | 0.55 | -15% |

### Analysis for RQ3

1. **Degradation Rate**: Calculate % AUC-PR drop per month without retraining
2. **Stability**: Show that periodic retraining maintains performance within ±2%
3. **Recommendation**: Suggest optimal retraining frequency (weekly/monthly)

### Metrics to Track

| Metric | Description |
|--------|-------------|
| `auc_pr_with_retrain` | Performance per window with accumulating updates |
| `auc_pr_static` | Performance per window using initial model |
| `degradation_rate` | % drop per month without retraining |
| `stability_variance` | Variance of AUC-PR across windows with retraining |

### Results
*To be filled after experiment*

---

## Experiment 5: Explainability Analysis (RQ4)

**Date**: TBD  
**Objective**: Generate actionable insights using SHAP  
**Status**: 🔄 PLANNED  
**Answers**: RQ4 (XAI for actionable insights)

### Methodology

1. Train production model with full features
2. Generate SHAP explanations (global + local)
3. Create case studies for analyst review
4. Document actionable insights

### Part A: Global Feature Importance

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Part B: Local Case Studies (3-5 examples)

For each flagged fraud case, generate:
- Top 5 contributing features
- Direction of contribution (increases/decreases fraud probability)
- Analyst-friendly explanation

| Case | Top Features | Explanation |
|------|--------------|-------------|
| Case 1 | `payment_type=INVOICE`, `account_age_days=2` | New account using invoice payment (high risk pattern) |
| Case 2 | `shared_email_count=15`, `component_size=23` | Part of large fraud ring sharing contact info |
| Case 3 | `listing_pagerank=0.001`, `user_listing_count=1` | Isolated listing with no network connections |

### Expected Top Features (Hypothesis)

| Rank | Feature | Hypothesis | Category |
|------|---------|------------|----------|
| 1 | `payment_type` | Fraudsters avoid direct payment | Tabular |
| 2 | `account_age_days` | New accounts are higher risk | Tabular |
| 3 | `shared_contact_email_count` | Email reuse indicates fraud rings | **Graph** |
| 4 | `listing_component_size` | Connected fraud networks | **Graph** |
| 5 | `email_time_spread` | Temporal patterns in email reuse | Time |

### Analysis for RQ4

1. **Feature Ranking**: Which features are most predictive?
2. **Graph Feature Value**: Are graph features in top 10?
3. **Actionable Insights**: What patterns should analysts look for?

### Deliverables

- [ ] SHAP summary plot (global importance)
- [ ] SHAP dependence plots (top 5 features)
- [ ] 3-5 local case study explanations
- [ ] Written insights for analyst consumption

### Results
*To be filled after experiment*

---

## Experiment 6: Production Readiness Validation

**Date**: TBD  
**Objective**: Validate model meets production requirements  
**Status**: 🔄 PLANNED

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-PR | >0.70 | Better than initial baseline |
| P@100 | >0.75 | 3 out of 4 flagged listings are fraud |
| Inference latency | <100ms | Real-time scoring capability |
| vs Seon | >2x precision | Significant improvement over production |

### Methodology

```bash
# Compare with Seon baseline (pre-computed)
python -c "
from src.utils.evaluate_seon import get_seon_metrics_for_comparison
seon_metrics = get_seon_metrics_for_comparison()
print(seon_metrics)
"

# Get deployment recommendation
python -m src.utils.mlflow_model_comparison recommend \
  --model-name fraud-detection-xgboost \
  --candidate-run-id <RUN_ID>
```

### Latency Measurement

Latency is automatically tracked per-window in MLflow:
- `latency_mean_ms`: Average inference time
- `latency_p95_ms`: 95th percentile latency
- `latency_per_sample_ms`: Per-prediction latency

### Results
*To be filled after experiment*

---

## Key Learnings (Updated During Experiments)

### Data Quality Insights

| Finding | Impact | Action |
|---------|--------|--------|
| **Boolean fields use NULL = FALSE semantics** | 50 fields have 100% semantic coverage | Include in model |
| `has_elevator` was incorrectly excluded | Lost valid discriminative feature | Re-included (40.9% TRUE rate) |
| Phone field coverage varies | Noisy phone edges | Unified phone edge with coalesce |
| Fraudsters avoid direct payment | High-value signal | Prioritize `payment_type` feature |

### Graph Structure Insights

| Finding | Impact | Action |
|---------|--------|--------|
| Fraud is isolated (lower connectivity) | GNNs struggle with heterophily | Use isolation as feature |
| Multiple phone edges redundant | Complexity without benefit | Merged into single edge |
| Node separation breaks transitivity | Performance drops | Use unified graph |

### Model Architecture Insights

| Finding | Impact | Action |
|---------|--------|--------|
| XGBoost learns interactions naturally | Explicit interaction features don't help | Skip interaction engineering |
| GNNs need complete graph history | Sliding windows hurt GNN performance | Use accumulating windows |
| Handcrafted features are explicit | Better interpretability | Prefer for production |

---

## Roadmap

### Phase 1: Feature Engineering & Baselines (Week 1-2)
- [x] Experiment 0: Feature & graph structure validation
- [ ] Experiment 1: Feature ablation study (RQ1)
- [ ] Experiment 2: Hybrid architecture comparison (RQ2)

### Phase 2: Optimization & Validation (Week 3-4)
- [ ] Experiment 3: Hyperparameter optimization
- [ ] Experiment 4: Concept drift evaluation (RQ3)

### Phase 3: Explainability & Deployment (Week 5-6)
- [ ] Experiment 5: SHAP analysis (RQ4)
- [ ] Experiment 6: Production readiness validation

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| AUC-PR | 0.72+ | TBD |
| P@100 | 0.80+ | TBD |
| Training time | <20 min | TBD |
| Inference latency | <100ms | TBD |

---

## Research Contributions

1. **Novel Relational Indicators (RQ1)**: Identification and validation of graph-derived features for real estate fraud detection
2. **Hybrid Architecture (RQ2)**: Comparison of handcrafted graph features vs GNN embeddings for XGBoost-based fraud detection
3. **Concept Drift Mitigation (RQ3)**: Validation of accumulating window training as passive concept drift adaptation
4. **Explainable Fraud Detection (RQ4)**: SHAP-based insights for operational fraud analysts

---

## References

- **Data & Features**: `docs/knowledge_base.md` (source of truth for data quality, feature definitions)
- **Architecture**: `docs/architecture.md` (project structure, technical design)
- **Feature Groups**: `src/models/config/constants.py` (FEATURE_GROUPS dict)
- **Ablation Config**: `conf/features/ablation.yaml` (experiment definitions)
- **Research Proposal**: `docs/NguyenHoangMinh_ResearchProposal.md`
