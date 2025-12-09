# Data Flow Documentation

> Last updated: 2025-12-09 (post-simplification refactoring)

## Overview

This document describes the data pipeline from raw ETL output to trained models.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  raw_insertions.parquet (ETL output)                                        │
│  • ~278 raw columns with ETL naming (e.g., listing.prices.rent.gross)       │
│  • Source of truth for all features                                         │
│  • Created by: src/data/etl/extract.py                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GRAPH ARTIFACTS LAYER                                  │
│                      create_artifacts.py                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Node Files:                                                                │
│  • nodes_listing.parquet → ALL raw cols + 3 derived:                        │
│    - insertion_id (alias of object_reference)                               │
│    - user_id (alias of owner_id)                                            │
│    - is_fraud (derived from fraud_flag)                                     │
│  • nodes_user.parquet → user_id, account_created_at                         │
│  • nodes_ip.parquet, nodes_email.parquet, nodes_phone.parquet, etc.         │
│                                                                             │
│  Edge Files:                                                                │
│  • edges_user_posts_listing.parquet                                         │
│  • edges_user_uses_ip.parquet                                               │
│  • edges_listing_contact_email.parquet                                      │
│  • edges_listing_phone.parquet                                              │
│  • etc.                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐
│    XGBoost Pipeline     │ │  GNN Pipeline   │ │   Graph Features            │
├─────────────────────────┤ ├─────────────────┤ ├─────────────────────────────┤
│                         │ │                 │ │                             │
│  loader.py              │ │ graph_builder.py│ │ graph_features.py           │
│  ↓                      │ │                 │ │ advanced_graph_features.py  │
│  FeatureProcessor       │ │ Transforms raw  │ │                             │
│  ↓                      │ │ ETL cols to     │ │ Uses:                       │
│  trainer.py             │ │ GNN features    │ │ • edges_*.parquet           │
│                         │ │ locally         │ │ • insertion_id              │
│  Uses:                  │ │                 │ │ • submission_at             │
│  • Auto tabular mode    │ │ Coalesces       │ │                             │
│  • Graph features       │ │ descriptions    │ │ Produces:                   │
│  • submission_at        │ │ for embeddings  │ │ • contact_email_count       │
│  • is_fraud (target)    │ │                 │ │ • shared_ip_user_count      │
│                         │ │                 │ │ • listing_pagerank          │
│                         │ │                 │ │ • etc. (20 graph features)  │
└─────────────────────────┘ └─────────────────┘ └─────────────────────────────┘
```

## Column Naming Strategy

### Raw ETL Names (Source of Truth)
All columns retain their original ETL names throughout the pipeline:
- `listing.prices.rent.gross`
- `listing.characteristics.hasBalcony`
- `listing.localization.primary`

### Derived Columns (Created in create_artifacts.py)
Only essential derived columns are created:

| Column | Source | Purpose |
|--------|--------|---------|
| `insertion_id` | `object_reference` | Graph node ID |
| `user_id` | `owner_id` | Graph relationship key |
| `is_fraud` | `fraud_flag IS NOT NULL` | Target label |

### Local Transformations (in graph_builder.py)
GNN-specific transformations are done locally:
```python
# Raw ETL name → Local alias for processing
pl.col("listing.prices.rent.gross").fill_null(0).alias("price_rent_gross")
```

## Feature Configuration

### Auto Mode (Default)
```yaml
# conf/features/auto.yaml
categories:
  - base   # All columns minus exclusions
  - graph  # Explicit graph-derived features
```

### Exclusions (src/models/config/constants.py)
```python
EXCLUDED_COLUMNS = {
    "fraud_flag", "is_fraud",           # Target
    "insertion_id", "object_reference", # IDs
    "submission_at",                    # Temporal
    # ...
}
```

## Key Files

| File | Purpose |
|------|---------|
| `src/data/etl/extract.py` | ETL query → raw_insertions.parquet |
| `src/data/graph/create_artifacts.py` | Raw → Graph artifacts |
| `src/data/graph/graph_builder.py` | Artifacts → PyG HeteroData |
| `src/data/loader.py` | Load nodes for training |
| `src/features/processor.py` | Feature orchestration |
| `src/features/generators/graph_features.py` | Graph feature computation |
| `src/models/xgboost/trainer.py` | XGBoost training |

## Artifact Dependencies

```
raw_insertions.parquet
    │
    └─► create_artifacts.py
            │
            ├─► nodes_listing.parquet ─┬─► loader.py ─► XGBoost
            │                          │
            │                          └─► graph_builder.py ─► GNN
            │
            ├─► nodes_user.parquet ────► loader.py (join)
            │
            └─► edges_*.parquet ───────┬─► graph_features.py ─► XGBoost
                                       │
                                       └─► graph_builder.py ─► GNN
```

## Regenerating Artifacts

After changes to create_artifacts.py:
```bash
# Regenerate graph artifacts
python -m src.data.graph.create_artifacts

# Or use make target
make artifacts
```

---

## Experiment Results Architecture

### Problem (Before)
Experiments hardcoded baseline values, leading to:
- Stale values when baselines change
- Inconsistent references across experiments
- No single source of truth for metrics

### Solution: Results Registry

```
artifacts/results/
├── xgboost_baseline.json      # Canonical XGBoost metrics
├── exp9_feature_selection.json
├── exp10a_anomaly_detection.json
├── exp10b_clustering.json
└── exp10c_association_rules.json
```

### Usage

```python
# Saving results (in experiment script)
from src.experiments.results import save_result, ExperimentID

save_result(ExperimentID.EXP10A_ANOMALY, {
    "best_method": "Isolation Forest",
    "best_auc_pr": 0.45,
    ...
})

# Loading results (in another experiment)
from src.experiments.results import get_metric, ExperimentID

baseline_auc = get_metric(ExperimentID.XGBOOST_BASELINE, "auc_pr", default=0.784)
```

### Canonical Experiment IDs

| ID | Description | Saves | Loads |
|----|-------------|-------|-------|
| `XGBOOST_BASELINE` | Best XGBoost metrics | trainer | exp10a |
| `SEON_BASELINE` | Seon fraud score baseline | evaluate_seon | exp6, exp7 |
| `EXP5_SHAP` | SHAP explainability | exp5 | - |
| `EXP6_BUSINESS_VALUE` | Business value metrics | exp6 | - |
| `EXP7_PRODUCTION` | Production readiness | exp7 | - |
| `EXP9_FEATURE_SELECTION` | Feature selection | exp9 | - |
| `EXP10A_ANOMALY` | Anomaly detection | exp10a | - |
| `EXP10B_CLUSTERING` | Clustering analysis | exp10b | - |
| `EXP10C_ASSOCIATION` | Association rules | exp10c | - |

### Benefits
- ✅ Single source of truth for metrics
- ✅ Experiments load baselines dynamically
- ✅ Results are idempotently updated on re-run
- ✅ Easy to list/query all results
