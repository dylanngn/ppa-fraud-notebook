# Experiment Journal

## Research Overview

**Topic**: Graph-Based Feature Engineering for Real Estate Fraud Detection: A Data Mining Approach

**Research Questions**:
1. How can graph-derived features improve fraud detection in sparse, heterophilic networks?
2. Is it more efficient to use handcrafted graph features or GNN-produced embeddings alongside tabular features for XGBoost-based fraud detection?
3. What is the optimal balance between model complexity, training time, and predictive performance?

**Core Framework**: Continuous Fraud Detection with XGBoost
- XGBoost serves as the final decision-maker
- Features can come from: (a) tabular data, (b) handcrafted graph statistics, or (c) GNN embeddings
- All experiments use accumulating (expanding) window training for production realism

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

```bash
# Data Pipeline (run once)
make etl                    # Extract, transform, load data
make build-graph            # Build PyTorch Geometric graph
make seon-baseline          # Generate static Seon baseline (one-time)

# Model Training (all use accumulating windows + MLflow tracking)
make train                  # XGBoost with production features (primary model)
make train-quick            # XGBoost with core features only (fast iteration)
make train-sage             # SAGE GNN hybrid (alternative)
make train-hgt              # HGT GNN hybrid (alternative)

# Hyperparameter Optimization
make optimize-xgboost       # Optimize XGBoost params
make optimize-pytorch MODEL_TYPE=sage  # Optimize GNN params

# Analysis
make data-quality-report    # Generate coverage analysis
```

### MLflow Tracking

All experiments are automatically tracked in MLflow:
- Parent run: Overall experiment
- Nested runs: Per-window evaluations
- Artifacts: Models, embeddings, feature importance

```bash
# Compare models (XGBoost registered as fraud-detection-xgboost)
make mlflow-compare-models MODEL_NAME=fraud-detection-xgboost CANDIDATE_RUN_ID=xxx

# Get deployment recommendation
make mlflow-deployment-recommendation MODEL_NAME=fraud-detection-xgboost CANDIDATE_RUN_ID=xxx
```

---

## Experiment 0: Feature & Graph Structure Validation

**Date**: 2025-11-30  
**Objective**: Validate feature cleanup and graph simplification changes  
**Status**: ✅ IMPLEMENTED (Ready for Validation)

### Background

Based on data quality analysis (`data_quality_reports/`), we implemented:

1. **Feature Processor Enhancement**: Added `exclude` list support in Hydra config
2. **Graph Simplification**: Merged phone edges from 2 → 1 (unified billing+lister phone)
3. **Feature Tiers**: Organized features into CORE, GRAPH, ENHANCED, DEPRECATED

### Changes Implemented

| Component | Change | Rationale |
|-----------|--------|-----------|
| `FeatureProcessor` | Added `from_config()` + `exclude` support | Enable config-driven feature exclusion |
| `create_artifacts.py` | Unified phone edge (coalesce billing→lister) | Reduce edge types from 9 → 8, maximize coverage |
| `graph_builder.py` | Updated to use single `has_phone` edge | Simplified graph structure |
| `conf/features/*.yaml` | Created quick/standard/production profiles | Enable ablation studies |

### Validation Steps

```bash
# Step 1: Rebuild graph with simplified structure
make build-graph

# Step 2: Verify graph statistics
python -c "
import torch
data = torch.load('artifacts/graph.pt')
print('Edge types:', data.edge_types)
print('Phone edges:', data['listing', 'has_phone', 'phone'].edge_index.shape)
"

# Step 3: Train with different feature profiles
python -m src.models.train features=quick      # Should exclude deprecated features
python -m src.models.train features=production # Full features minus deprecated
```

### Expected Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Edge types | 9 | 8 | -1 (phone merge) |
| Phone edge coverage | 70-98% | ~99% | +29% (coalesce) |
| Feature count (production) | ~50 | ~44 | -6 (deprecated removed) |
| AUC-PR | ~0.70 | ~0.70 | No regression |

### Decision Branching

Based on validation results:

- **If AUC-PR drops > 2%**: Revert phone edge merge, investigate
- **If AUC-PR stable**: Proceed with simplified structure for all experiments
- **If AUC-PR improves**: Document as finding, update baseline

---

## Experiment 1: Baseline Feature Analysis

**Date**: TBD  
**Objective**: Establish baseline performance with tabular-only features  
**Status**: 🔄 PLANNED

### Hypothesis
Base tabular features alone can achieve reasonable fraud detection, but graph-derived features will provide significant improvement.

### Methodology

```bash
# Train with different feature profiles
python -m src.models.train features=quick      # Tier 1 only
python -m src.models.train features=standard   # Tier 1 + Tier 2
python -m src.models.train features=production # All tiers
```

### Feature Tiers (from data quality analysis)

| Tier | Features | Coverage | Description |
|------|----------|----------|-------------|
| **Tier 1** | `account_age_days`, `payment_type`, `bundle_tier`, `log_price`, `latitude/longitude`, `offer_type`, `living_space`, `rooms` | >80% | Core tabular features |
| **Tier 2** | `shared_contact_email_count`, `listing_component_size`, `listing_pagerank`, etc. | Computed | Handcrafted graph statistics |
| **Tier 3** | `email_time_spread`, `email_recency_weighted`, text features | Computed | Time-weighted & text features |

### Expected Metrics

| Configuration | Expected AUC-PR | Training Time |
|---------------|-----------------|---------------|
| Tier 1 only | ~0.59 | ~10 min |
| Tier 1 + Tier 2 | ~0.67 | ~15 min |
| Full (all tiers) | ~0.70 | ~20 min |

### Results
*To be filled after experiment*

### Decision Branching

Based on Experiment 1 results:

- **If Tier 2 adds < 5% AUC-PR**: Investigate graph feature computation, may need tuning
- **If Tier 2 adds > 10% AUC-PR**: Graph features confirmed valuable, prioritize graph quality
- **If Tier 3 adds < 1% AUC-PR**: Consider removing time-weighted features for simplicity

---

## Experiment 2: Model Comparison - Handcrafted vs GNN Features

**Date**: TBD  
**Objective**: Compare efficiency of handcrafted graph features vs GNN embeddings  
**Status**: 🔄 PLANNED

### Research Question
Is it more efficient (in terms of time, complexity, performance, scalability) to:
- **Option A**: Compute handcrafted graph statistics (degree, PageRank, component size) and feed to XGBoost
- **Option B**: Train GNN to produce embeddings, then feed embeddings + tabular features to XGBoost

### Models to Compare

| Model | Description | Features | Baseline |
|-------|-------------|----------|----------|
| **Seon** | Production system (binary classifier) | Rule-based | ✅ Production Baseline |
| **XGBoost** | Our primary model with handcrafted features | Tabular + Graph Statistics | Candidate |
| **SAGE Hybrid** | GraphSAGE embeddings + XGBoost | Tabular + 64-dim embeddings | Candidate |
| **HGT Hybrid** | Heterogeneous Graph Transformer + XGBoost | Tabular + 64-dim embeddings | Candidate |

### Methodology

```bash
# Step 1: Train XGBoost with handcrafted features
make train  # or: python -m src.models.train features=production

# Step 2: Train SAGE hybrid
make train-sage

# Step 3: Train HGT hybrid  
make train-hgt
```

### Evaluation Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| **Performance** | AUC-PR, P@100 | Higher is better |
| **Training Time** | Wall-clock minutes | Lower is better |
| **Complexity** | Lines of code, dependencies | Lower is better |
| **Scalability** | Memory usage, batch support | Lower memory, batch support |
| **Interpretability** | SHAP compatibility | Full is better |

### Expected Results

| Model | AUC-PR | P@100 | Training Time | GPU Required |
|-------|--------|-------|---------------|--------------|
| Seon (Prod Baseline) | ~0.50 | ~0.60 | N/A | No |
| XGBoost | ~0.70 | ~0.77 | ~15 min | No |
| SAGE Hybrid | ~0.64 | ~0.75 | ~2-3 hrs | Yes |
| HGT Hybrid | ~0.64 | ~0.75 | ~2-3 hrs | Yes |

### Analysis Framework

1. **Performance Gap**: If |AUC-PR_xgboost - AUC-PR_gnn| < 0.02, XGBoost wins on simplicity
2. **Cost-Benefit**: Training time × (cloud GPU cost) vs performance improvement
3. **Production Viability**: Batch inference support, model serving complexity

### Results
*To be filled after experiment*

### Decision Branching

Based on Experiment 2 results:

- **If GNN outperforms by > 5%**: Invest in GNN optimization, consider hybrid production
- **If GNN underperforms or matches**: Use handcrafted features, document GNN limitations
- **If training time > 4 hrs**: Consider GNN architecture simplification or sampling

---

## Experiment 3: Hyperparameter Optimization

**Date**: TBD  
**Objective**: Optimize XGBoost hyperparameters for fraud detection  
**Status**: 🔄 PLANNED

### Methodology

```bash
# Run hyperparameter optimization (Optuna-based)
make optimize-xgboost
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

## Experiment 4: Temporal Window Analysis

**Date**: TBD  
**Objective**: Find optimal training window strategy  
**Status**: 🔄 PLANNED

### Research Question
What is the optimal training window configuration for continuous fraud detection?

### Window Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Accumulating** | Train on [start, t], test on [t, t+14] | Production deployment |
| **Sliding** | Train on [t-90, t], test on [t, t+14] | Memory-constrained |
| **Fixed** | Train on [t-365, t], test on [t, t+14] | Long-term patterns |

### Methodology

```yaml
# conf/model/xgboost.yaml - Modify training settings
training:
  initial_window_days: 180    # Try: 90, 180, 365
  step_days: 7                # Try: 7, 14, 28
  max_windows: null
```

### Expected Findings
- **Accumulating window** expected to perform best for fraud detection
- **Longer initial window** (180+ days) captures more fraud patterns
- **Step size** of 7-14 days balances evaluation granularity and speed

### Results
*To be filled after experiment*

---

## Experiment 5: Feature Importance & SHAP Analysis

**Date**: TBD  
**Objective**: Understand which features drive fraud predictions  
**Status**: 🔄 PLANNED

### Methodology

1. Train production model with full features
2. Extract SHAP values from MLflow artifacts
3. Analyze feature importance distribution
4. Identify potential new features from error cases

### Expected Top Features

Based on data quality analysis and fraud patterns:

| Rank | Feature | Hypothesis |
|------|---------|------------|
| 1 | `payment_type` | Fraudsters avoid direct payment (28% fraud rate for new+invoice) |
| 2 | `account_age_days` | New accounts are higher risk |
| 3 | `shared_contact_email_count` | Email reuse indicates fraud rings |
| 4 | `listing_component_size` | Connected fraud networks |
| 5 | `email_time_spread` | Temporal patterns in email reuse |

### Analysis Outputs
- SHAP summary plot
- Feature interaction matrix
- False negative analysis (missed fraud cases)

### Results
*To be filled after experiment*

---

## Experiment 6: Production Deployment Simulation

**Date**: TBD  
**Objective**: Validate model performance in production-like conditions  
**Status**: 🔄 PLANNED

### Methodology

1. Use most recent data window as holdout test set
2. Measure inference latency (now tracked per-window in MLflow)
3. Compare against Seon static baseline
4. Validate model serving workflow

```bash
# Compare with Seon baseline (pre-computed)
from src.utils.evaluate_seon import get_seon_metrics_for_comparison
seon_metrics = get_seon_metrics_for_comparison()

# Get deployment recommendation
make mlflow-deployment-recommendation MODEL_NAME=fraud-detection-xgboost CANDIDATE_RUN_ID=xxx
```

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-PR | >0.70 | Better than initial baseline |
| P@100 | >0.75 | 3 out of 4 flagged listings are fraud |
| Inference latency | <100ms | Real-time scoring |
| Precision vs Seon | >2x | Reduce false positives |

### Results
*To be filled after experiment*

---

## Key Learnings (Updated During Experiments)

### Data Quality Insights

| Finding | Impact | Action |
|---------|--------|--------|
| Phone field coverage varies: billing (98%) >> lister (70%) >> viewing (3%) | Noisy phone edges | Unified phone edge with coalesce |
| `is_new` is 99.99% NULL | Misleading feature | Exclude from training |
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

### Phase 1: Baseline Establishment (Week 1-2)
- [x] Experiment 0: Feature & graph structure validation (implemented)
- [ ] Experiment 1: Feature tier ablation
- [ ] Experiment 2: Model comparison (baseline vs hybrid)

### Phase 2: Optimization (Week 3-4)
- [ ] Experiment 3: Hyperparameter optimization
- [ ] Experiment 4: Window strategy optimization

### Phase 3: Analysis & Deployment (Week 5-6)
- [ ] Experiment 5: SHAP analysis
- [ ] Experiment 6: Production simulation

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| AUC-PR | 0.72+ | TBD |
| P@100 | 0.80+ | TBD |
| Training time | <20 min | TBD |
| Inference latency | <100ms | TBD |

---

## Research Contributions

1. **Framework**: Continuous fraud detection with accumulating window training
2. **Comparison**: Handcrafted graph features vs GNN embeddings (efficiency analysis)
3. **Findings**: When graph features outperform GNNs in sparse, heterophilic networks
4. **Best Practices**: Feature engineering guidelines for fraud detection

---

## References

- Project Setup: `docs/knowledge_base.md`
- Data Quality: `data_quality_reports/data_quality_report.txt`
- Feature Constants: `src/models/config/constants.py`
- Configuration: `conf/config.yaml`
