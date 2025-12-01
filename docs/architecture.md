# Architecture Documentation

## Overview

This project implements a fraud detection system for real estate listings using:
- **XGBoost** as the primary model with handcrafted features
- **GNN Hybrid** as an alternative approach (GraphSAGE/HGT + XGBoost)
- **Seon** as the production baseline to beat

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRAUD DETECTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐   │
│  │ Database │───▶│ ETL Pipeline │───▶│ Artifacts   │───▶│ Model Train  │   │
│  │ (Aurora  │    │              │    │ (Parquet)   │    │              │   │
│  │ Postgres)│    │              │    │             │    │              │   │
│  └──────────┘    └──────────────┘    └─────────────┘    └──────────────┘   │
│                                            │                    │           │
│                                            ▼                    ▼           │
│                                    ┌─────────────┐      ┌─────────────┐    │
│                                    │ Graph Build │      │ MLflow      │    │
│                                    │ (PyG)       │      │ Tracking    │    │
│                                    └─────────────┘      └─────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Architecture

### Package Structure

```
ppa-fraud-notebook/
├── conf/                           # Hydra Configuration
│   ├── config.yaml                 # Main defaults
│   ├── data/default.yaml           # Database & ETL settings
│   ├── features/                   # Feature profiles
│   │   ├── baseline.yaml           # Production features
│   │   ├── quick.yaml              # Core features only
│   │   ├── standard.yaml           # Core + Graph
│   │   └── production.yaml         # All features
│   └── model/xgboost.yaml          # Model hyperparameters
│
├── artifacts/                      # Generated Data (gitignored)
│   ├── raw_insertions.parquet      # ETL output
│   ├── nodes_*.parquet             # Graph nodes
│   ├── edges_*.parquet             # Graph edges
│   ├── graph.pt                    # PyTorch Geometric graph
│   ├── seon_baseline.json          # Static Seon metrics
│   └── embeddings_*.pt             # GNN embeddings
│
├── src/
│   ├── data/                       # Data Pipeline
│   │   ├── pipeline.py             # ETL orchestrator (Hydra entry)
│   │   ├── etl/                    # Extract-Transform-Load
│   │   │   ├── extract.py          # Database queries
│   │   │   ├── transform.py        # Data cleaning
│   │   │   └── load.py             # Parquet writing
│   │   └── graph/                  # Graph Building
│   │       ├── schema.py           # Graph schema (single source of truth)
│   │       ├── create_artifacts.py # Node/edge parquet creation
│   │       └── graph_builder.py    # PyTorch Geometric graph
│   │
│   ├── features/                   # Feature Engineering
│   │   ├── registry.py             # Feature category registry
│   │   ├── processor.py            # Feature processor (Hydra-aware)
│   │   ├── definitions/            # Feature computation
│   │   │   ├── base.py             # Tabular features
│   │   │   ├── graph.py            # Graph-derived features
│   │   │   └── text.py             # Text features
│   │   └── generators/             # Feature generators
│   │       ├── graph_features.py
│   │       ├── advanced_graph_features.py
│   │       ├── time_weighted_features.py
│   │       └── text_features.py
│   │
│   ├── models/                     # Model Training
│   │   ├── train.py                # Main entry point (Hydra)
│   │   ├── registry.py             # MLflow model registry
│   │   ├── xgboost/
│   │   │   ├── trainer.py          # XGBoost accumulating window
│   │   │   └── utils.py            # Feature validation, constraints
│   │   ├── gnn/
│   │   │   ├── sage.py             # GraphSAGE hybrid
│   │   │   └── hgt.py              # HGT hybrid
│   │   └── hyperopt/
│   │       ├── xgboost.py          # Optuna hyperopt
│   │       └── pytorch.py          # GNN hyperopt
│   │
│   └── utils/                      # Metrics, Seon baseline, MLflow CLI
│       ├── metrics.py              # Centralized metrics
│       ├── evaluate_seon.py        # Seon baseline
│       └── mlflow_model_comparison.py
│
└── docs/                           # Documentation
    ├── experiment_journal.md
    ├── knowledge_base.md
    └── architecture.md (this file)
```

---

## Model Architecture

### 1. XGBoost (Primary Model)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           XGBoost Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    ACCUMULATING WINDOW STRATEGY                       │  │
│  │                                                                       │  │
│  │  Window 1: Train [Jan-Jun] ──▶ Test [Jul 1-14]                       │  │
│  │  Window 2: Train [Jan-Jul 7] ──▶ Test [Jul 8-21]                     │  │
│  │  Window 3: Train [Jan-Jul 14] ──▶ Test [Jul 15-28]                   │  │
│  │  ...                                                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Per Window:                                                                │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐                 │
│  │ Raw Data    │───▶│ FeatureProcessor│───▶│ XGBClassifier│                │
│  │ (Polars)    │    │ (categories,    │    │              │                │
│  │             │    │  exclude)       │    │              │                │
│  └─────────────┘    └─────────────────┘    └─────────────┘                 │
│                              │                     │                        │
│                              ▼                     ▼                        │
│                     ┌─────────────────┐   ┌─────────────────┐              │
│                     │ Feature Tiers:  │   │ MLflow Logging: │              │
│                     │ - CORE (base)   │   │ - Metrics       │              │
│                     │ - GRAPH         │   │ - Model         │              │
│                     │ - ENHANCED      │   │ - Latency       │              │
│                     └─────────────────┘   └─────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. GNN Hybrid (Alternative)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GNN Hybrid Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: GNN Pre-training (Once)                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  ┌───────────┐    ┌─────────────────┐    ┌─────────────────┐           ││
│  │  │ HeteroData│───▶│ SAGE/HGT Model  │───▶│ 64-dim Embeddings│          ││
│  │  │ Graph     │    │ (2 layers,      │    │ per Listing     │           ││
│  │  │           │    │  64 hidden)     │    │                 │           ││
│  │  └───────────┘    └─────────────────┘    └─────────────────┘           ││
│  │       │                                           │                     ││
│  │       ▼                                           ▼                     ││
│  │  Node Types:                               Saved to:                    ││
│  │  - User                                    artifacts/embeddings_*.pt   ││
│  │  - Listing (target)                                                     ││
│  │  - Email, Phone, Address, IP                                            ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  Phase 2: Hybrid XGBoost (Per Window)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       ││
│  │  │ Tabular Features│ + │ GNN Embeddings  │ = │ Combined Features│      ││
│  │  │ (from processor)│   │ (64 dims)       │   │                 │       ││
│  │  └─────────────────┘   └─────────────────┘   └─────────────────┘       ││
│  │                                                       │                 ││
│  │                                                       ▼                 ││
│  │                                              ┌─────────────────┐        ││
│  │                                              │ XGBClassifier   │        ││
│  │                                              │ (Final Decision)│        ││
│  │                                              └─────────────────┘        ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Graph Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HETEROGENEOUS GRAPH (8 Edge Types)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────┐                                    │
│                              │  USER   │                                    │
│                              └────┬────┘                                    │
│                      ┌───────────┬┴┬───────────┐                           │
│                      │ posts     │  │ uses     │ has_email                  │
│                      ▼           │  ▼          ▼                            │
│               ┌──────────┐       │ ┌────┐  ┌───────┐                       │
│               │ LISTING  │       │ │ IP │  │ EMAIL │                       │
│               └────┬─────┘       │ └────┘  └───────┘                       │
│     ┌─────────────┬┴┬─────────────┴────────────┘                           │
│     │             │  │                                                      │
│     │ has_contact_│  │ has_billing_email                                   │
│     │ email       │  │                                                      │
│     │             │  │ has_phone (unified: billing + lister)               │
│     │             │  │                                                      │
│     ▼             │  ▼                                                      │
│ ┌───────┐   ┌────────┐   ┌─────────┐                                       │
│ │ EMAIL │   │ PHONE  │   │ ADDRESS │                                       │
│ └───────┘   └────────┘   └─────────┘                                       │
│                                ▲                                            │
│               located_at ─────┤                                            │
│               billing_addr ───┘                                            │
│                                                                             │
│  Coverage:                                                                  │
│  - User→Listing: 100%                                                       │
│  - Listing→Email: 98-99%                                                    │
│  - Listing→Phone: ~99% (coalesced)                                         │
│  - Listing→Address: 91-98%                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration & Tracking Framework

### Why Both Hydra AND MLflow?

These tools are **complementary**, not overlapping:

| Tool | Role | Phase | Question Answered |
|------|------|-------|-------------------|
| **Hydra** | Configuration | Before training | "What should we run?" |
| **MLflow** | Tracking/Registry | During/after training | "What happened? Which model is best?" |

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     HYDRA       │────▶│    TRAINING     │────▶│     MLFLOW      │
│  (Input Config) │     │   (Execution)   │     │ (Output Tracking)│
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ • features=quick│     │ • Load data     │     │ • Log metrics   │
│ • model.params  │     │ • Generate feats│     │ • Store model   │
│ • experiment_   │     │ • Train XGBoost │     │ • Model registry│
│   name          │     │ • Evaluate      │     │ • Compare runs  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Without Hydra**: Config scattered in code, no CLI overrides, no composable profiles  
**Without MLflow**: No metric history, no model versioning, manual comparison

Together they enable:
1. **Reproducibility**: Hydra config → exact same experiment
2. **Traceability**: MLflow links config → metrics → model
3. **Iteration**: Change config (Hydra) → compare results (MLflow)

### Logging Strategy

| Log Type | Tool | Purpose |
|----------|------|---------|
| Experiment metrics | MLflow | AUC-PR, P@100, latency → tracked & compared |
| Operational logs | Hydra | Progress, warnings, errors → console + file |
| Debug output | Hydra | `hydra.verbose=true` → DEBUG level |

Hydra logging provides:
- **Auto-configured** Python logging for `@hydra.main()` modules
- **Colored output** via [hydra-colorlog](https://hydra.cc/docs/plugins/colorlog/)
- **Log files** saved to `outputs/<date>/<time>/*.log`
- **CLI control**: `hydra.verbose=true` for debug, `hydra/job_logging=disabled` to silence

### Hydra Configuration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYDRA CONFIGURATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  conf/config.yaml (defaults)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  defaults:                                                           │   │
│  │    - data: default        ───▶ conf/data/default.yaml               │   │
│  │    - features: baseline   ───▶ conf/features/baseline.yaml          │   │
│  │    - model: xgboost       ───▶ conf/model/xgboost.yaml              │   │
│  │                                                                      │   │
│  │  experiment_name: "ppa-fraud-detection"                             │   │
│  │  seed: 42                                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Override via CLI:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  python -m src.models.train features=quick                          │   │
│  │  python -m src.models.train model.params.n_estimators=1000          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feature Profiles

| Profile | Categories | Use Case |
|---------|------------|----------|
| `quick` | CORE only | Fast iteration |
| `standard` | CORE + GRAPH | Balanced |
| `production` | CORE + GRAPH + ENHANCED | Best performance |
| `baseline` | All (with exclusions) | Default |

---

## MLflow Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MLFLOW TRACKING STRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Experiment: ppa-fraud-detection                                            │
│  │                                                                          │
│  ├── Run: xgboost_accumulating_20251130_1430 (Parent)                      │
│  │   ├── Params: model_name, initial_window_days, feature_categories       │
│  │   ├── Metrics: mean_auc_pr, best_auc_pr, num_windows                    │
│  │   ├── Tags: model_type=xgboost, training_mode=accumulating_window       │
│  │   │                                                                      │
│  │   ├── Nested Run: window_0                                               │
│  │   │   ├── Metrics: auc_pr, auc_roc, p_at_100, latency_*                 │
│  │   │   └── Artifacts: model.pkl                                          │
│  │   │                                                                      │
│  │   ├── Nested Run: window_1                                               │
│  │   │   └── ...                                                            │
│  │   │                                                                      │
│  │   └── Nested Run: window_N                                               │
│  │       └── ...                                                            │
│  │                                                                          │
│  └── Model Registry                                                         │
│      ├── fraud-detection-xgboost (best window model)                       │
│      ├── fraud-detection-gnn-sage                                          │
│      └── fraud-detection-hybrid-sage                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Metrics Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CENTRALIZED METRICS (src/utils/metrics.py)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  calculate_metrics(y_true, y_pred, include_confusion_matrix=False)          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Always Computed:                                                    │   │
│  │  - auc_pr, auc_roc         (Ranking quality)                        │   │
│  │  - p@50, p@100, p@200      (Precision at K)                         │   │
│  │  - r@50, r@100, r@200      (Recall at K)                            │   │
│  │  - lift@50, lift@100, lift@200                                       │   │
│  │  - fraud_count, fraud_rate                                           │   │
│  │                                                                      │   │
│  │  Optional (for Seon binary classifier):                             │   │
│  │  - accuracy, precision, recall, f1_score                            │   │
│  │  - tp, tn, fp, fn                                                    │   │
│  │  - catch_rate, false_alarm_rate                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  measure_inference_latency(model, X_sample, n_iterations=100)               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  - latency_mean_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms  │   │
│  │  - latency_per_sample_ms                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GNN Hybrid Support

The trainer fully supports `embedding_generator` callback for per-window GNN embedding generation. This enables fair temporal comparison between XGBoost and GNN hybrid models.

All modules now use unified Hydra configuration directly, eliminating legacy compatibility layers.

## Recommended Improvements

1. **Add Integration Tests**: Verify full pipeline works end-to-end
2. **Optimize Embedding Generation**: Graph building per window is slow; consider caching

---

## Execution Flow

```bash
# 1. Data Pipeline (one-time setup)
make etl                 # Extract from DB → artifacts/raw_insertions.parquet
make build-graph         # Create graph → artifacts/nodes_*.parquet, edges_*.parquet
make seon-baseline       # Compute Seon metrics → artifacts/seon_baseline.json

# 2. Model Training
make train              # XGBoost with production features
make train-quick        # XGBoost with core features (fast)
make train-sage         # GNN hybrid (SAGE)
make train-hgt          # GNN hybrid (HGT)

# 3. Hyperparameter Optimization
make optimize-xgboost   # Optuna search for XGBoost

# 4. Model Comparison
make mlflow-compare-models MODEL_NAME=fraud-detection-xgboost CANDIDATE_RUN_ID=xxx
```

---

## Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Config (Input)** | Hydra | Define what to run, composable profiles |
| **Logging** | Hydra + colorlog | Colored console + auto log files |
| **Tracking (Output)** | MLflow | Record results, model registry, comparison |
| ETL | Polars | High-performance data processing |
| Graph | PyTorch Geometric | GNN-ready heterogeneous graph |
| Primary Model | XGBoost | Fraud classification |
| Alternative | GraphSAGE/HGT | GNN embeddings for hybrid |
| Baseline | Seon (static) | Production system to beat |

