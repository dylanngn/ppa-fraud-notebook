# System Architecture: Fraud Detection Framework

**Project:** Hybrid GNN + XGBoost Fraud Detection  
**Version:** 1.0  
**Last Updated:** December 17, 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration Management (Hydra)](#3-configuration-management-hydra)
4. [Experiment Tracking (MLflow)](#4-experiment-tracking-mlflow)
5. [ETL Pipeline](#5-etl-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Graph Construction](#7-graph-construction)
8. [Model Architecture](#8-model-architecture)
9. [Training Pipeline](#9-training-pipeline)
10. [Hyperparameter Optimization](#10-hyperparameter-optimization)
11. [API Service](#11-api-service)

---

## 1. Overview

This framework implements a hybrid fraud detection system combining:
- **Graph Neural Networks (GNNs)** for relational feature learning
- **XGBoost** for classification
- **Point-in-Time labels** for temporal integrity
- **Expanding window evaluation** for concept drift detection

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  insertion_events.csv    →    ETL Pipeline    →    merged_events.parquet    │
│  seon_transactions.csv   →    (Polars)        →                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  FeatureStore           →    TemporalSplitter    →    Train/Test DFs        │
│  (loading, temporal)    →    (PIT labels, gap)   →                           │
│                                                                              │
│  GraphBuilder           →    HeteroData          →    Node/Edge Tensors     │
│  (temporal edges)       →    (PyG)               →                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Baseline Models:       Logistic Regression, Random Forest                   │
│  Main Model:            XGBoost Classifier                                   │
│  Hybrid Model:          GraphSAGE Embedder + XGBoost                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  SingleTrainingPipeline:     Fixed train/test split                          │
│  ExpandingWindowPipeline:    Multiple expanding windows (concept drift)      │
│  HPO:                        Optuna sweeper for hyperparameter tuning        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVING LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI Service:       Real-time prediction with internal event store       │
│  MLflow:                Model registry and experiment tracking               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Variants Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL VARIANTS                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │  BASELINE MODELS (for comparison)                                                 │
  ├───────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                   │
  │   ┌─────────────────────────┐         ┌─────────────────────────┐                │
  │   │  Logistic Regression    │         │  Random Forest          │                │
  │   │  ─────────────────────  │         │  ─────────────────────  │                │
  │   │  • Linear baseline      │         │  • Tree ensemble        │                │
  │   │  • Fast training        │         │  • 100 trees            │                │
  │   │  • Interpretable        │         │  • max_depth: 10        │                │
  │   │  • AUC-PR: 0.49         │         │  • AUC-PR: 0.59         │                │
  │   └─────────────────────────┘         └─────────────────────────┘                │
  │                                                                                   │
  └───────────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │  MAIN MODELS                                                                      │
  ├───────────────────────────────────────────────────────────────────────────────────┤
  │                                                                                   │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐│
  │   │  Vanilla XGBoost                                                            ││
  │   │  ─────────────────────────────────────────────────────────────────────────  ││
  │   │                                                                             ││
  │   │   ┌─────────────────┐                                                       ││
  │   │   │ Base Features   │ ──────────► ┌─────────────┐ ──────► Fraud            ││
  │   │   │ (71 features)   │             │  XGBoost    │         Probability       ││
  │   │   └─────────────────┘             └─────────────┘                           ││
  │   │                                                                             ││
  │   │   • AUC-PR: 0.8250                                                          ││
  │   │   • Precision@0.5: 78.5%                                                    ││
  │   │   • Fast training (~30s)                                                    ││
  │   └─────────────────────────────────────────────────────────────────────────────┘│
  │                                                                                   │
  │   ┌─────────────────────────────────────────────────────────────────────────────┐│
  │   │  GNN + XGBoost (Hybrid)                                                     ││
  │   │  ─────────────────────────────────────────────────────────────────────────  ││
  │   │                                                                             ││
  │   │   ┌─────────────────┐                                                       ││
  │   │   │ Base Features   │─────────┐                                             ││
  │   │   │ (71 features)   │         │                                             ││
  │   │   └─────────────────┘         │                                             ││
  │   │                               ▼                                             ││
  │   │   ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐                ││
  │   │   │ Heterogeneous   │───►│  GraphSAGE  │───►│ Embeddings  │────┐           ││
  │   │   │ Graph (20M edges)│    │  (2 layers) │    │ (16 dims)   │    │           ││
  │   │   └─────────────────┘    └─────────────┘    └─────────────┘    │           ││
  │   │                                                                │           ││
  │   │                               ┌────────────────────────────────┘           ││
  │   │                               │                                             ││
  │   │                               ▼                                             ││
  │   │                    ┌───────────────────────┐    ┌─────────────┐             ││
  │   │                    │  Combined Features    │───►│  XGBoost    │──► Fraud   ││
  │   │                    │  (87 = 71 + 16)       │    │             │    Prob.   ││
  │   │                    └───────────────────────┘    └─────────────┘             ││
  │   │                                                                             ││
  │   │   • AUC-PR: 0.8233                                                          ││
  │   │   • Precision@0.5: 79.2% (+0.8%)                                            ││
  │   │   • Catches 12 more fraud cases                                             ││
  │   │   • Slower training (~5min)                                                 ││
  │   └─────────────────────────────────────────────────────────────────────────────┘│
  │                                                                                   │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Project Structure

```
ppa-fraud-notebook/
├── conf/                          # Hydra config for ETL
│   ├── config.yaml               # Main config
│   └── data/
│       └── default.yaml          # Data paths and column mappings
│
├── configs/                       # Hydra config for training
│   ├── experiment.yaml           # Main experiment config
│   ├── hpo_xgboost.yaml         # XGBoost HPO config
│   ├── hpo_gnn_xgboost.yaml     # Joint GNN+XGBoost HPO
│   ├── hpo_gnn_only.yaml        # GNN-only HPO (fixed XGBoost)
│   └── hpo_high_recall.yaml     # High-recall tuning
│
├── src/
│   ├── data/                     # ETL pipeline
│   │   ├── extract.py           # CSV loading (Polars LazyFrame)
│   │   ├── transform.py         # As-of join, label derivation
│   │   ├── load.py              # Parquet output
│   │   └── etl.py               # Pipeline orchestration
│   │
│   ├── features/                 # Feature engineering
│   │   ├── schema.py            # Feature definitions (FEATURE_SCHEMA)
│   │   ├── store.py             # FeatureStore (data loading)
│   │   ├── graph_builder.py     # Heterogeneous graph construction
│   │   └── temporal_split.py    # Point-in-Time splitting
│   │
│   ├── models/                   # Model implementations
│   │   ├── base.py              # Abstract base classes
│   │   ├── baselines.py         # LogisticRegression, RandomForest
│   │   ├── xgboost_classifier.py # XGBoost wrapper
│   │   ├── graphsage.py         # GraphSAGE encoder
│   │   └── hybrid.py            # HybridPipeline (GNN + classifier)
│   │
│   ├── training/                 # Training pipelines
│   │   ├── trainer.py           # Entry point (Hydra main)
│   │   └── pipeline.py          # Single/ExpandingWindow pipelines
│   │
│   ├── evaluation/               # Evaluation tools
│   │   ├── shap_analysis.py     # SHAP explainability
│   │   ├── seon_benchmark.py    # SEON baseline comparison
│   │   └── drift.py             # Concept drift detection
│   │
│   ├── api/                      # REST API service
│   │   ├── main.py              # FastAPI application
│   │   ├── models.py            # Pydantic schemas
│   │   ├── store.py             # In-memory event store
│   │   └── features.py          # Real-time feature engineering
│   │
│   └── utils/                    # Utilities
│       ├── anonymize.py         # PII hashing
│       ├── metrics.py           # Evaluation metrics
│       └── hydra_utils.py       # Config helpers
│
├── scripts/
│   └── run_gnn_hpo_with_best_xgb.py  # Sequential HPO script
│
├── docs/                         # Documentation
│   ├── data_overview.md         # Dataset analysis
│   ├── experiment_journal.md    # Experiment log
│   ├── architecture.md          # This file
│   └── methodology.md           # Research methodology
│
├── Makefile                      # Build targets
└── requirements.txt              # Dependencies
```

---

## 3. Configuration Management (Hydra)

### 3.1 Hydra Overview

The project uses [Hydra](https://hydra.cc/) for hierarchical configuration management:

- **Composable configs**: Base config with optional overrides
- **Command-line overrides**: `python -m trainer model.variant=gnn_xgboost`
- **Multi-run sweeps**: Built-in support for HPO with Optuna

### 3.2 ETL Configuration (`conf/`)

```yaml
# conf/data/default.yaml
sources:
  events_csv: "artifacts/insertion_events.csv"
  seon_csv: "artifacts/seon_transactions.csv"

paths:
  output: "artifacts/merged_events.parquet"

processing:
  streaming: true  # Memory-efficient processing

columns:
  events:
    id_source: "OBJECTREFERENCE"
    time_col: "DATAPIPELINE_EVENT_SENT_AT"
    fraud_flag_col: "FLAGGEDFORFRAUD"
  seon:
    id_col: "transaction_id"
    unique_col: "id"
    time_col: "date"
```

### 3.3 Training Configuration (`configs/`)

```yaml
# configs/experiment.yaml
experiment:
  name: "fraud_detection"
  tracking_uri: "sqlite:///ppa-fraud-detection-mlflow.db"

data:
  path: "artifacts/merged_events.parquet"
  train_start_date: "2024-12-01"
  train_end_date: "2025-06-01"
  test_end_date: "2025-07-01"

training:
  mode: "single"  # or "expanding"
  gap_days: 7
  run_shap: false

model:
  variant: "vanilla_xgboost"  # logistic_regression, random_forest, gnn_xgboost
  
  xgboost:
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.1
    scale_pos_weight: 1.0
    
  gnn:
    hidden_dim: 32
    output_dim: 16
    num_layers: 2
    epochs: 5
```

### 3.4 Config Overrides via CLI

```bash
# Override single parameters
python -m src.training.trainer model.variant=gnn_xgboost model.gnn.epochs=50

# Use different config file
python -m src.training.trainer --config-name=hpo_xgboost

# Multi-run sweep
python -m src.training.trainer --multirun model.xgboost.max_depth=3,5,7
```

---

## 4. Experiment Tracking (MLflow)

### 4.1 MLflow Integration

All experiments are logged to MLflow for:
- **Parameter tracking**: Model hyperparameters
- **Metric logging**: AUC-PR, AUC-ROC, precision, recall
- **Artifact storage**: Trained models, SHAP plots
- **Run comparison**: Side-by-side experiment analysis

### 4.2 Setup

```python
# src/training/trainer.py
mlflow.set_tracking_uri("sqlite:///ppa-fraud-detection-mlflow.db")
mlflow.set_experiment(cfg.experiment.name)

with mlflow.start_run(run_name=run_name):
    mlflow.log_params({...})
    result = pipeline.run()
    mlflow.log_metrics(result)
    mlflow.xgboost.log_model(model, name="model")
```

### 4.3 Viewing Results

```bash
make mlflow  # Starts MLflow UI at http://localhost:5000
```

---

## 5. ETL Pipeline

### 5.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   ETL PIPELINE                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐         ┌──────────────────────┐
  │  insertion_events.csv │         │  seon_transactions.csv│
  │  ─────────────────── │         │  ───────────────────  │
  │  • 804,717 rows      │         │  • ~140K rows         │
  │  • Listing lifecycle │         │  • SEON API responses │
  │  • Status changes    │         │  • Risk scores        │
  │  • Fraud flags       │         │  • Device/IP data     │
  └──────────┬───────────┘         └──────────┬───────────┘
             │                                 │
             │    ┌────────────────────────────┘
             │    │
             ▼    ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              EXTRACT (Polars)                                     │
  │  ─────────────────────────────────────────────────────────────────────────────── │
  │  • Events: LazyFrame (streaming for memory efficiency)                           │
  │  • SEON: Eager load (needed for join lookup)                                     │
  │  • Parse datetime columns with explicit formats                                   │
  └──────────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              TRANSFORM                                            │
  │  ─────────────────────────────────────────────────────────────────────────────── │
  │                                                                                   │
  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │
  │  │ 1. AS-OF JOIN: Match events to SEON transactions by time                   │ │
  │  │    ┌────────────────────────────────────────────────────────────────────┐  │ │
  │  │    │  Event @ T=10:30    ──────►  SEON @ T=10:45 (forward, preferred)   │  │ │
  │  │    │                     ──────►  SEON @ T=10:15 (backward, fallback)   │  │ │
  │  │    └────────────────────────────────────────────────────────────────────┘  │ │
  │  └─────────────────────────────────────────────────────────────────────────────┘ │
  │                                                                                   │
  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │
  │  │ 2. MERGE: Combine 509 columns from both sources                            │ │
  │  └─────────────────────────────────────────────────────────────────────────────┘ │
  │                                                                                   │
  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │
  │  │ 3. DERIVE LABELS: is_fraud = FLAGGEDFORFRAUD is not null                   │ │
  │  └─────────────────────────────────────────────────────────────────────────────┘ │
  │                                                                                   │
  │  ┌─────────────────────────────────────────────────────────────────────────────┐ │
  │  │ 4. ANONYMIZE: SHA-256 hash PII columns                                     │ │
  │  │    • INSERTION_ID → INSERTION_ID_hash                                      │ │
  │  │    • LISTING_LISTER_EMAIL → LISTING_LISTER_EMAIL_hash                      │ │
  │  │    • ip → ip_hash                                                          │ │
  │  └─────────────────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              LOAD                                                 │
  │  ─────────────────────────────────────────────────────────────────────────────── │
  │  • Output: merged_events.parquet                                                 │
  │  • Compression: zstd                                                             │
  │  • Size: ~8.3 GB (804,717 rows × 509 columns)                                   │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Extract (`src/data/extract.py`)

```python
def extract_data(cfg) -> Tuple[pl.LazyFrame, pl.DataFrame]:
    """
    Load events as LazyFrame (streaming) and SEON fully.
    
    Events: LazyFrame for memory efficiency (~800K rows)
    SEON: Eager load (needed for join lookup)
    """
    events_lf = pl.scan_csv(events_path, ...)
    seon_df = pl.read_csv(seon_path, ...)
    return events_lf, seon_df
```

### 5.3 Transform (`src/data/transform.py`)

**As-of Join Logic:**
```python
def correlate_events_to_seon(events_df, seon_df):
    """
    For each event, find closest SEON transaction:
    1. Forward join: closest SEON AFTER event (preferred)
    2. Backward join: closest SEON BEFORE event (fallback)
    """
    fwd = events.join_asof(seon, strategy="forward", ...)
    bwd = events.join_asof(seon, strategy="backward", ...)
    return coalesce(fwd, bwd)
```

**Label Derivation:**
```python
def derive_label(df):
    """Fraud label = FLAGGEDFORFRAUD timestamp is not null."""
    return df.with_columns(
        pl.col("FLAGGEDFORFRAUD").is_not_null().cast(pl.Int8).alias("is_fraud")
    )
```

### 5.4 Load (`src/data/load.py`)

```python
def load_data(df: pl.DataFrame, output_path: str):
    """Save to Parquet with compression."""
    df.write_parquet(output_path, compression="zstd")
```

---

## 6. Feature Engineering

### 6.1 Feature Schema (`src/features/schema.py`)

Centralized feature definitions:

```python
@dataclass(frozen=True)
class FeatureSchema:
    # Note: SEON ML prediction scores are not used as features
    # Only raw SEON signals (boolean/categorical) are used
    seon_score_features: Tuple[str, ...] = ()  # Empty
    
    # SEON boolean signals (21 total)
    seon_ip_boolean: Tuple[str, ...] = ("data_center_proxy", "residential_proxy", ...)
    seon_email_boolean: Tuple[str, ...] = ("email/domain/disposable", ...)
    
    # Listing features
    listing_numeric_features: Tuple[str, ...] = ("LISTING_PRICES_RENT_NET", ...)
    listing_categorical_features: Tuple[str, ...] = ("LISTING_OFFERTYPE", ...)
    
    # Graph identity columns (for edges)
    graph_config: GraphIdentityConfig = ...
```

### 6.2 Feature Store (`src/features/store.py`)

```python
class FeatureStore:
    """Central access point for feature data."""
    
    def load_data(self) -> pl.LazyFrame:
        """Load and preprocess data."""
        df = pl.scan_parquet(self.raw_data_path)
        df = self._create_temporal_features(df)  # hour, weekday, weekend
        df = self._cast_seon_booleans(df)        # Boolean → Int8
        return df
    
    def prepare_train_test_data(self, train_end, test_end, gap_days):
        """Create temporal train/test split."""
        ...
```

### 6.3 Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| SEON Boolean | 21 | `tor`, `vpn`, `email/domain/disposable` |
| SEON Categorical | 12 | `ip_country`, `session/device_type` |
| Listing Numeric | 10 | `LISTING_PRICES_RENT_NET`, `LISTING_CHARACTERISTICS_NUMBEROFROOMS` |
| Listing Categorical | 7 | `LISTING_OFFERTYPE`, `LISTING_CATEGORIES` |
| Temporal | 4 | `event_hour`, `event_weekday`, `event_is_weekend` |
| **Total Base** | **~71** | - |
| GNN Embeddings | 16-64 | `gnn_emb_0`, `gnn_emb_1`, ... |

### 6.4 Feature Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE ENGINEERING FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

  merged_events.parquet
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              FeatureStore.load_data()                             │
  │  ─────────────────────────────────────────────────────────────────────────────── │
  │                                                                                   │
  │    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
  │    │ Parse Datetime  │───►│ Create Temporal │───►│ Cast Booleans   │             │
  │    │ Columns         │    │ Features        │    │ to Int8         │             │
  │    └─────────────────┘    └─────────────────┘    └─────────────────┘             │
  │                                                                                   │
  │    Temporal Features Created:                                                     │
  │    • event_hour (0-23)                                                           │
  │    • event_weekday (0-6)                                                         │
  │    • event_is_weekend (0/1)                                                      │
  │    • event_is_business_hours (0/1)                                               │
  └──────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              Feature Selection by Variant                         │
  │  ─────────────────────────────────────────────────────────────────────────────── │
  │                                                                                   │
  │    ┌───────────────────────────────────────────────────────────────────────────┐ │
  │    │  VANILLA_XGBOOST                        GNN_XGBOOST                       │ │
  │    │  ─────────────────                      ───────────────                   │ │
  │    │                                                                           │ │
  │    │  ┌─────────────────────┐                ┌─────────────────────┐          │ │
  │    │  │ Base Features (66)  │                │ Base Features (66)  │          │ │
  │    │  │ • SEON Boolean (21) │                │ • SEON Boolean (21) │          │ │
  │    │  │ • SEON Categ. (12)  │                │ • SEON Categ. (12)  │          │ │
  │    │  │ • (No SEON Scores)  │                │ • (No SEON Scores)  │          │ │
  │    │  │ • Listing Num. (10) │                │ • Listing Num. (10) │          │ │
  │    │  │ • Listing Cat. (7)  │                │ • Listing Cat. (7)  │          │ │
  │    │  │ • Temporal (4)      │                │ • Temporal (4)      │          │ │
  │    │  └─────────────────────┘                └──────────┬──────────┘          │ │
  │    │           │                                        │                      │ │
  │    │           │                             ┌──────────┴──────────┐          │ │
  │    │           │                             │ + GNN Embeddings    │          │ │
  │    │           │                             │   (16 dimensions)   │          │ │
  │    │           │                             └──────────┬──────────┘          │ │
  │    │           │                                        │                      │ │
  │    │           ▼                                        ▼                      │ │
  │    │  ┌─────────────────────┐                ┌─────────────────────┐          │ │
  │    │  │ Total: 71 features  │                │ Total: 87 features  │          │ │
  │    │  └─────────────────────┘                └─────────────────────┘          │ │
  │    └───────────────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Graph Construction

### 7.1 Heterogeneous Graph (`src/features/graph_builder.py`)

```python
@dataclass
class HeteroGraphConfig:
    primary_node_type: str = "listing"
    
    entity_node_types: Dict = {
        "user": {"id_columns": ["user_id_hash"]},
        "device": {"id_columns": ["session/device_hash"]},
        "ip": {"id_columns": ["ip_hash"]},
    }
    
    temporal_identity_columns: List[str] = [
        "ip_hash", "session/device_hash", "user_id_hash",
        "LISTING_LISTER_EMAIL_hash", "LISTING_LISTER_PHONE_hash"
    ]
```

### 7.2 Node Types

| Type | Description | Count (typical) |
|------|-------------|-----------------|
| `listing` | Real estate listings | ~50K |
| `user` | User accounts | ~65K |
| `device` | Device fingerprints | ~12K |
| `ip` | IP addresses | ~35K |

### 7.3 Edge Types

```
Listing-to-Entity Edges:
  listing ──[used_user]──► user
  listing ──[used_device]──► device
  listing ──[used_ip]──► ip

Temporal Listing-to-Listing Edges:
  listing ──[shares_device]──► listing  (same device fingerprint)
  listing ──[shares_ip]──► listing      (same IP address)
  listing ──[shares_email]──► listing   (same email)
  listing ──[shares_phone]──► listing   (same phone)
  listing ──[shares_user]──► listing    (same user account)

Reverse Edges:
  user ──[rev_used_user]──► listing
  ...
```

### 7.4 Heterogeneous Graph Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         HETEROGENEOUS GRAPH STRUCTURE                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────┐
                    │  USER   │
                    │  (65K)  │
                    └────┬────┘
                         │ used_user
                         │ ▲
                         │ │ rev_used_user
                         ▼ │
  ┌─────────┐       ┌────────────┐       ┌─────────┐
  │ DEVICE  │◄──────│  LISTING   │──────►│   IP    │
  │  (12K)  │       │   (50K)    │       │  (35K)  │
  └─────────┘       └────────────┘       └─────────┘
       ▲            ▲    │    ▲               ▲
       │            │    │    │               │
  used_device       │    │    │          used_ip
                    │    │    │
                    │    ▼    │
              ┌─────┴────────────┴─────┐
              │                        │
              │  LISTING ◄──────────── LISTING
              │          shares_device │
              │          shares_ip     │
              │          shares_email  │
              │          shares_phone  │
              │          shares_user   │
              │    (TEMPORAL EDGES)    │
              │                        │
              └────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              EDGE STATISTICS                                     │
  ├─────────────────────────────────────────────────────────────────────────────────┤
  │  Edge Type          │  Count (Train)  │  Description                            │
  │  ─────────────────  │  ─────────────  │  ─────────────────────────────────────  │
  │  shares_device      │  12,171,416     │  Same device fingerprint                │
  │  shares_ip          │  2,597,653      │  Same IP address                        │
  │  shares_user        │  2,065,691      │  Same user account                      │
  │  shares_email       │  1,978,469      │  Same email                             │
  │  shares_phone       │  1,305,532      │  Same phone number                      │
  │  used_* + rev_*     │  ~388,000       │  Entity relationships                   │
  │  ─────────────────  │  ─────────────  │                                         │
  │  TOTAL              │  ~20.5 Million  │                                         │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Temporal Edge Construction

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL EDGE CONSTRUCTION                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Timeline showing how temporal edges are created between listings sharing a device:

  Device: "abc123_hash"

  Time ───────────────────────────────────────────────────────────────────────────►

       │
       │   Listing A         Listing B         Listing C
       │   T = 10:00         T = 14:00         T = 16:00
       │      │                 │                 │
       │      ▼                 ▼                 ▼
       │   ┌──────┐          ┌──────┐          ┌──────┐
       │   │  A   │─────────►│  B   │─────────►│  C   │
       │   └──────┘          └──────┘          └──────┘
       │         shares_device    shares_device
       │
       │   ┌──────┐─────────────────────────────►┌──────┐
       │   │  A   │         shares_device        │  C   │
       │   └──────┘                              └──────┘
       │
       │   ❌ NO edge from B → A (would violate temporal order)
       │   ❌ NO edge from C → A or C → B (same reason)
       │

  Rule: Edge (src → dst) only created if src_time < dst_time
        This prevents future information leakage in the graph
```

### 7.6 Temporal Integrity in Code

Edges are only created between listings where `src_time < dst_time`:
```python
for i, (src_lid, src_time) in enumerate(timed_listings):
    for j in range(i + 1, len(timed_listings)):
        dst_lid, dst_time = timed_listings[j]
        if src_time < dst_time:  # Temporal ordering
            add_edge(src_lid, dst_lid)
```

---

## 8. Model Architecture

### 8.1 Model Variants

```python
class ModelVariant(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"  # Baseline
    RANDOM_FOREST = "random_forest"              # Baseline
    VANILLA_XGBOOST = "vanilla_xgboost"          # Main model
    GNN_XGBOOST = "gnn_xgboost"                  # Hybrid model
```

### 8.2 Baseline Models (`src/models/baselines.py`)

```python
class LogisticRegressionClassifier(BaselineClassifier):
    def __init__(self, C=1.0, max_iter=1000, class_weight='balanced'):
        self.model = LogisticRegression(C=C, max_iter=max_iter, ...)

class RandomForestBaseline(BaselineClassifier):
    def __init__(self, n_estimators=100, max_depth=10, ...):
        self.model = RandomForestClassifier(n_estimators=n_estimators, ...)
```

### 8.3 XGBoost Classifier (`src/models/xgboost_classifier.py`)

```python
class XGBoostClassifier(BaseClassifier):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, ...):
        self.model = xgb.XGBClassifier(
            enable_categorical=True,  # Native categorical support
            use_label_encoder=False,
            eval_metric='logloss',
            ...
        )
```

### 8.4 GraphSAGE Embedder (`src/models/graphsage.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    GraphSAGE Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Node features (16 dims) from raw SEON signals + listing + temporal │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ HeteroConv Layer 1                                        │   │
│  │   - SAGEConv for each edge type (shares_device, etc.)    │   │
│  │   - Aggregation: mean + sum across edge types            │   │
│  │   - Output: hidden_dim (32)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│                        ReLU + Dropout (0.3)                      │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ HeteroConv Layer 2                                        │   │
│  │   - Same structure as Layer 1                            │   │
│  │   - Output: out_channels (16)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Output: Node embeddings (16 dims per listing)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Training Objective:** Self-supervised link prediction
```python
# Positive: actual edges in graph
pos_score = (z[src] * z[dst]).sum(dim=-1)

# Negative: random node pairs
neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)

loss = binary_cross_entropy_with_logits(pos_score, neg_score)
```

### 8.5 Hybrid Pipeline (`src/models/hybrid.py`)

```python
class HybridPipeline:
    """Orchestrates GNN + Classifier."""
    
    def fit(self, train_df, graph):
        # 1. Train GNN embedder (self-supervised)
        self.embedder.fit(graph, node_features)
        
        # 2. Get embeddings for training nodes
        embeddings = self.embedder.transform(graph, node_features)
        
        # 3. Combine base features + GNN embeddings
        X_train = concat([base_features, embeddings])
        
        # 4. Train classifier
        self.classifier.fit(X_train, y_train)
    
    def predict(self, df, graph):
        # 1. Get embeddings (frozen GNN)
        embeddings = self._predict_gnn(df, graph)
        
        # 2. Combine features
        X = concat([base_features, embeddings])
        
        # 3. Classify
        return self.classifier.predict_proba(X)
```

### 8.6 Hybrid Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         GNN+XGBOOST HYBRID PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              TRAINING PHASE
  ─────────────────────────────────────────────────────────────────────────────────────

  ┌─────────────────┐         ┌─────────────────┐
  │   Train Data    │         │   Train Graph   │
  │   (380K rows)   │         │   (47K nodes)   │
  └────────┬────────┘         └────────┬────────┘
           │                           │
           │                           ▼
           │               ┌───────────────────────┐
           │               │   GraphSAGE Embedder  │
           │               │   (Self-supervised)   │
           │               │   ─────────────────   │
           │               │   • Link prediction   │
           │               │   • 30 epochs         │
           │               │   • 2 layers          │
           │               └───────────┬───────────┘
           │                           │
           │                           ▼
           │               ┌───────────────────────┐
           │               │   Node Embeddings     │
           │               │   (47K × 16 dims)     │
           │               └───────────┬───────────┘
           │                           │
           ▼                           ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              FEATURE CONCATENATION                               │
  │  ─────────────────────────────────────────────────────────────────────────────  │
  │                                                                                  │
  │   ┌───────────────────────────────┐    ┌───────────────────────────────┐       │
  │   │   Base Features (66 cols)     │ +  │   GNN Embeddings (16 cols)    │       │
  │   │   • Raw SEON signals          │    │   • gnn_emb_0 ... gnn_emb_15  │       │
  │   │   • Listing attributes        │    │   • Network patterns encoded  │       │
  │   │   • Temporal features         │    │   • Fraud ring signals        │       │
  │   └───────────────────────────────┘    └───────────────────────────────┘       │
  │                                    │                                            │
  │                                    ▼                                            │
  │                    ┌───────────────────────────────┐                           │
  │                    │   Combined: 87 features       │                           │
  │                    └───────────────────────────────┘                           │
  └─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │       XGBoost Classifier      │
                       │       ───────────────────     │
                       │   • n_estimators: 350         │
                       │   • max_depth: 8              │
                       │   • learning_rate: 0.0147     │
                       │   • scale_pos_weight: 1.62    │
                       └───────────────────────────────┘
                                       │
                                       ▼
                           ┌───────────────────┐
                           │   Trained Model   │
                           └───────────────────┘


                              INFERENCE PHASE
  ─────────────────────────────────────────────────────────────────────────────────────

  ┌─────────────────┐         ┌─────────────────┐
  │   Test Data     │         │ Inference Graph │
  │   (39K rows)    │         │   (54K nodes)   │
  └────────┬────────┘         └────────┬────────┘
           │                           │
           │                           ▼
           │               ┌───────────────────────┐
           │               │   GraphSAGE Embedder  │
           │               │   (FROZEN - no train) │
           │               └───────────┬───────────┘
           │                           │
           ▼                           ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │   Base Features + GNN Embeddings → Combined Features (87)                       │
  └─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   XGBoost.predict_proba()    │
                       └───────────────────────────────┘
                                       │
                                       ▼
                           ┌───────────────────┐
                           │  Fraud Probability │
                           │    (0.0 - 1.0)    │
                           └───────────────────┘
```

---

## 9. Training Pipeline

### 9.1 Single Training Pipeline (`src/training/pipeline.py`)

```python
class SingleTrainingPipeline:
    """Fixed train/test split."""
    
    def run(self):
        # 1. Load data
        df = self.feature_store.load_data().collect()
        
        # 2. Apply temporal split with PIT labels
        train_df, test_df = splitter.split(df, train_end, test_end)
        
        # 3. Build graphs (if GNN variant)
        if self.variant == ModelVariant.GNN_XGBOOST:
            train_graph = graph_builder.build_graph(train_df, train_end)
            inference_graph = graph_builder.build_graph(inference_df, test_end)
        
        # 4. Fit pipeline
        pipeline.fit(train_df, train_graph)
        
        # 5. Predict and evaluate
        probs = pipeline.predict(test_df, inference_graph)
        metrics = calculate_metrics(labels, probs)
        
        # 6. Log to MLflow
        mlflow.log_metrics(metrics)
```

### 9.2 Expanding Window Pipeline

```python
class ExpandingWindowPipeline:
    """Multiple expanding windows for concept drift."""
    
    def run(self):
        # Generate window splits
        # Window 1: Train [start → T1], Test [T1+gap → T2]
        # Window 2: Train [start → T2], Test [T2+gap → T3]
        
        for train_df, test_df, train_end, test_end in splits:
            pipeline = self._build_pipeline()
            pipeline.fit(train_df, train_graph)
            metrics = evaluate(pipeline, test_df)
            all_metrics.append(metrics)
        
        # Aggregate: mean, std, drift_range
        return aggregate_metrics(all_metrics)
```

### 9.3 Point-in-Time Labels (`src/features/temporal_split.py`)

```python
def _apply_pit_labels(df, cutoff):
    """
    Only assign fraud label if fraud was flagged BEFORE cutoff.
    Prevents label leakage from future fraud flags.
    """
    df = df.with_columns(
        pl.when(
            pl.col("_fraud_timestamp").is_not_null() &
            (pl.col("_fraud_timestamp") <= cutoff)
        )
        .then(1)
        .otherwise(0)
        .alias("is_fraud")
    )
```

### 9.4 Temporal Split Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL TRAIN/TEST SPLIT                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Time ───────────────────────────────────────────────────────────────────────────────►

       2024-12-01                    2025-06-01         2025-06-08      2025-07-01
           │                             │                  │               │
           ▼                             ▼                  ▼               ▼
  ─────────┬─────────────────────────────┬──────────────────┬───────────────┬─────────
           │                             │                  │               │
           │◄────── TRAIN PERIOD ───────►│◄── 7-DAY GAP ──►│◄── TEST ─────►│
           │       (6 months)            │    (leakage     │   (3 weeks)   │
           │                             │     buffer)      │               │
           │   380,190 events            │                  │  39,645       │
           │   23,543 fraud (6.19%)      │                  │  events       │
           │                             │                  │  769 fraud    │
           │                             │                  │  (1.94%)      │
  ─────────┴─────────────────────────────┴──────────────────┴───────────────┴─────────


  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                         POINT-IN-TIME LABEL EXAMPLE                             │
  │  ───────────────────────────────────────────────────────────────────────────── │
  │                                                                                 │
  │  Listing X created: 2025-05-20                                                 │
  │  Fraud flagged:     2025-06-15                                                 │
  │  Train cutoff:      2025-06-01                                                 │
  │                                                                                 │
  │  Time ─────────────────────────────────────────────────────────────────►       │
  │                                                                                 │
  │       2025-05-20          2025-06-01              2025-06-15                   │
  │           │                   │                       │                         │
  │           ▼                   ▼                       ▼                         │
  │  ─────────┬───────────────────┬───────────────────────┬─────────────────       │
  │           │                   │                       │                         │
  │      Listing X          Train Cutoff           Fraud Flag Set                  │
  │       Created                 │                       │                         │
  │                               │                       │                         │
  │           │◄── TRAIN DATA ───►│                       │                         │
  │                               │                       │                         │
  │                                                                                 │
  │  Point-in-Time Label for Listing X:                                            │
  │    • In TRAINING: is_fraud = 0  (flag not yet set at cutoff)                   │
  │    • In TESTING:  is_fraud = 1  (flag was set before test_end)                 │
  │                                                                                 │
  │  ✓ This prevents label leakage - we don't use future fraud information         │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.5 Expanding Window Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         EXPANDING WINDOW STRATEGY                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

  Time ───────────────────────────────────────────────────────────────────────────────►

       Start    W1 End    W2 End    W3 End    W4 End    W5 End
         │        │         │         │         │         │
         ▼        ▼         ▼         ▼         ▼         ▼

  WINDOW 1:
  ─────────┬──────────────────┬───────┬──────────────────┬─────────
           │◄── TRAIN (2mo) ─►│ GAP   │◄── TEST (1mo) ──►│
           │                  │       │                  │
           │    103,018       │ 7 day │     61,290       │  AUC-PR: 0.7405
           │    events        │       │     events       │

  WINDOW 2:
  ─────────┬──────────────────────────────────┬───────┬──────────────────┬─────────
           │◄────────── TRAIN (3mo) ─────────►│ GAP   │◄── TEST (1mo) ──►│
           │                                  │       │                  │
           │           173,739 events         │       │     56,195       │  AUC-PR: 0.7218
           │                                  │       │     events       │

  WINDOW 3:
  ─────────┬──────────────────────────────────────────────────┬───────┬──────────────────┬─────────
           │◄────────────────── TRAIN (4mo) ─────────────────►│ GAP   │◄── TEST (1mo) ──►│
           │                                                  │       │                  │
           │                    240,660 events                │       │     54,188       │  AUC-PR: 0.7244

  ... continues with expanding training data ...

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              AGGREGATE RESULTS                                   │
  │  ───────────────────────────────────────────────────────────────────────────── │
  │                                                                                 │
  │   Mean AUC-PR:   0.7675 ± 0.058                                                 │
  │   Mean AUC-ROC:  0.9642 ± 0.008                                                 │
  │   Drift Range:   0.155 (max - min AUC-PR)                                       │
  │                                                                                 │
  │   ✓ Low standard deviation indicates stable model performance                   │
  │   ✓ Drift range < 0.2 suggests manageable concept drift                         │
  │   → Recommendation: Monthly retraining is sufficient                            │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Hyperparameter Optimization

### 10.1 Optuna Integration

HPO uses `hydra-optuna-sweeper`:

```yaml
# configs/hpo_xgboost.yaml
hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      n_startup_trials: 10
    direction: maximize  # Maximize AUC-PR
    n_trials: 50
    
    params:
      model.xgboost.n_estimators: range(50, 500, step=50)
      model.xgboost.max_depth: range(3, 12)
      model.xgboost.learning_rate: interval(0.01, 0.3)
      model.xgboost.scale_pos_weight: interval(1.0, 20.0)
```

### 10.2 HPO Variants

| Target | Config | Description |
|--------|--------|-------------|
| `make hpo` | `hpo_xgboost.yaml` | XGBoost HPO (50 trials) |
| `make hpo-high-recall` | `hpo_high_recall.yaml` | Focus on recall |
| `make hpo-gnn` | `hpo_gnn_xgboost.yaml` | Joint GNN+XGBoost |
| `make hpo-gnn-with-best-xgb` | Script | GNN-only with fixed XGBoost |

### 10.3 Sequential HPO Strategy

```bash
# 1. First, optimize XGBoost alone
make hpo

# 2. Then, optimize GNN using best XGBoost params
make hpo-gnn-with-best-xgb
```

### 10.4 HPO Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         HYPERPARAMETER OPTIMIZATION WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────────────────────┘

  PHASE 1: XGBoost Optimization (make hpo)
  ─────────────────────────────────────────────────────────────────────────────────────

                       ┌───────────────────────────────────────────────────────────┐
                       │                    Optuna TPE Sampler                     │
                       │                    ─────────────────────                  │
                       │    • 50 trials                                            │
                       │    • 10 startup trials (random)                           │
                       │    • Bayesian optimization after                          │
                       └───────────────────────────────────────────────────────────┘
                                                │
                       ┌────────────────────────┼────────────────────────┐
                       │                        │                        │
                       ▼                        ▼                        ▼
                ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
                │  Trial 1    │         │  Trial 2    │   ...   │  Trial 50   │
                │  ─────────  │         │  ─────────  │         │  ─────────  │
                │  depth: 5   │         │  depth: 8   │         │  depth: 8   │
                │  lr: 0.05   │         │  lr: 0.02   │         │  lr: 0.015  │
                │  ...        │         │  ...        │         │  ...        │
                │             │         │             │         │             │
                │  AUC: 0.79  │         │  AUC: 0.81  │         │  AUC: 0.825 │
                └─────────────┘         └─────────────┘         └──────┬──────┘
                                                                       │
                                                              ┌────────┴────────┐
                                                              │   BEST TRIAL    │
                                                              │   ───────────   │
                                                              │   AUC-PR: 0.825 │
                                                              │   depth: 8      │
                                                              │   lr: 0.0147    │
                                                              │   ...           │
                                                              └────────┬────────┘
                                                                       │
                                                                       ▼
                                                              ┌─────────────────┐
                                                              │ Save to MLflow  │
                                                              └────────┬────────┘
                                                                       │
                                                                       │
  PHASE 2: GNN Optimization (make hpo-gnn-with-best-xgb)               │
  ─────────────────────────────────────────────────────────────────────┼───────────

                                                                       │
                                                                       ▼
                       ┌───────────────────────────────────────────────────────────┐
                       │              Load Best XGBoost Params from MLflow         │
                       │              ─────────────────────────────────────────    │
                       │                                                           │
                       │    scripts/run_gnn_hpo_with_best_xgb.py                   │
                       │    • Query MLflow for best HPO run                        │
                       │    • Extract XGBoost parameters                           │
                       │    • Fix them for GNN tuning                              │
                       └───────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                       ┌───────────────────────────────────────────────────────────┐
                       │                    GNN Parameter Search                   │
                       │                    ────────────────────────               │
                       │    • hidden_dim: [32, 64, 128, 256]                       │
                       │    • output_dim: [16, 32, 64]                             │
                       │    • num_layers: [2, 3, 4]                                │
                       │    • epochs: [10, 20, 30, 40, 50]                         │
                       └───────────────────────────────────────────────────────────┘
                                                │
                                                ▼
                       ┌───────────────────────────────────────────────────────────┐
                       │                    BEST CONFIGURATION                     │
                       │                    ──────────────────────                 │
                       │                                                           │
                       │    XGBoost (fixed from Phase 1):                          │
                       │    • n_estimators: 350                                    │
                       │    • max_depth: 8                                         │
                       │    • learning_rate: 0.0147                                │
                       │                                                           │
                       │    GNN (optimized in Phase 2):                            │
                       │    • hidden_dim: 128                                      │
                       │    • output_dim: 16                                       │
                       │    • epochs: 30                                           │
                       │                                                           │
                       │    Combined AUC-PR: 0.8233                                │
                       └───────────────────────────────────────────────────────────┘
```

---

## 11. API Service

### 11.1 API Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              API SERVICE ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────────────────────┐
                         │          External System            │
                         │  (Real Estate Platform Backend)     │
                         └───────────────────┬─────────────────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         │                   │                   │
                         ▼                   ▼                   ▼
               ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
               │  POST /ingest   │ │ POST /predict   │ │ POST /bulk-     │
               │                 │ │                 │ │ ingest          │
               │  Store event    │ │  Get fraud      │ │                 │
               │  for future     │ │  probability    │ │  Load history   │
               │  predictions    │ │                 │ │                 │
               └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                        │                   │                   │
                        ▼                   ▼                   ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                              FastAPI Application                                 │
  │  ─────────────────────────────────────────────────────────────────────────────  │
  │                                                                                  │
  │   ┌─────────────────────────────────────────────────────────────────────────┐   │
  │   │                          EventStore (In-Memory)                         │   │
  │   │  ─────────────────────────────────────────────────────────────────────  │   │
  │   │  • Stores events by insertion_id                                        │   │
  │   │  • Indexes by device, IP, email, phone                                  │   │
  │   │  • Enables graph-like feature computation                               │   │
  │   │  • Optional JSON persistence                                            │   │
  │   └─────────────────────────────────────────────────────────────────────────┘   │
  │                                                                                  │
  │   ┌─────────────────────────────────────────────────────────────────────────┐   │
  │   │                        FeatureEngineer                                  │   │
  │   │  ─────────────────────────────────────────────────────────────────────  │   │
  │   │  • Extract SEON features from event payload                             │   │
  │   │  • Extract listing features                                             │   │
  │   │  • Compute graph-like features from EventStore:                         │   │
  │   │    - email_link_count (listings sharing email)                          │   │
  │   │    - device_link_count (listings sharing device)                        │   │
  │   │    - ip_link_count (listings sharing IP)                                │   │
  │   └─────────────────────────────────────────────────────────────────────────┘   │
  │                                                                                  │
  │   ┌─────────────────────────────────────────────────────────────────────────┐   │
  │   │                        XGBoost Model (Loaded at startup)                │   │
  │   │  ─────────────────────────────────────────────────────────────────────  │   │
  │   │  • Loaded from MLflow model registry                                    │   │
  │   │  • predict_proba() → fraud probability                                  │   │
  │   └─────────────────────────────────────────────────────────────────────────┘   │
  │                                                                                  │
  └─────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                         ┌─────────────────────────────────────┐
                         │         Response to Client          │
                         │  • fraud_probability: 0.0 - 1.0     │
                         │  • risk_level: LOW/MEDIUM/HIGH      │
                         │  • risk_factors: [...]              │
                         └─────────────────────────────────────┘
```

### 11.2 Prediction Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PREDICTION REQUEST FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

   POST /predict
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  EventPayload                                                                    │
  │  ───────────────────────────────────────────────────────────────────────────── │
  │  {                                                                               │
  │    "insertion_id": "abc123",                                                     │
  │    "timestamp": "2025-06-15T10:30:00Z",                                         │
  │    "status": "DRAFT",                                                            │
  │    "listing": { "offer_type": "RENT", "category": "STUDIO", ... },              │
  │    "seon": { "ip_country": "DE", "tor": false, "datacenter": false, ... },     │
  │    "session": { "device_hash": "xyz789", "ip_hash": "ip123", ... }              │
  │  }                                                                               │
  └─────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  1. FEATURE EXTRACTION                                                           │
  │  ───────────────────────────────────────────────────────────────────────────── │
  │                                                                                  │
  │    ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐     │
  │    │ Raw SEON Signals   │   │ Listing Features   │   │ Graph-like Features│     │
  │    │ ────────────────   │   │ ────────────────   │   │ ────────────────   │     │
  │    │ • ip_country: DE   │   │ • offer_type: RENT │   │ • device_links: 5  │     │
  │    │ • datacenter: false│   │ • category: STUDIO │   │ • ip_links: 2      │     │
  │    │ • tor: false       │   │ • platforms: [HG]  │   │ • email_links: 0   │     │
  │    └────────────────────┘   └────────────────────┘   └────────────────────┘     │
  │                │                       │                       │                 │
  │                └───────────────────────┼───────────────────────┘                 │
  │                                        ▼                                         │
  │                          ┌────────────────────────┐                             │
  │                          │  Combined: 71 features │                             │
  │                          └────────────────────────┘                             │
  └─────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  2. MODEL PREDICTION                                                             │
  │  ───────────────────────────────────────────────────────────────────────────── │
  │                                                                                  │
  │                    model.predict_proba(features) → [0.23, 0.77]                  │
  │                                                    ▲                             │
  │                                                    │                             │
  │                                          fraud probability = 0.77               │
  └─────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  3. RESPONSE                                                                     │
  │  ───────────────────────────────────────────────────────────────────────────── │
  │  {                                                                               │
  │    "insertion_id": "abc123",                                                     │
  │    "fraud_probability": 0.77,                                                    │
  │    "risk_level": "HIGH",                                                         │
  │    "risk_factors": [                                                             │
  │      { "feature": "ip_country", "value": "DE", "impact": "high" },              │
  │      { "feature": "category", "value": "STUDIO", "impact": "medium" },           │
  │      { "feature": "device_links", "value": 5, "impact": "medium" }              │
  │    ],                                                                            │
  │    "model_version": "1.0.0",                                                     │
  │    "processed_at": "2025-06-15T10:30:01Z"                                        │
  │  }                                                                               │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 FastAPI Application (`src/api/main.py`)

```python
app = FastAPI(title="Fraud Detection API")

@app.post("/predict")
async def predict(event: EventPayload) -> PredictResponse:
    """Predict fraud probability for an event."""
    features = feature_engineer.extract(event, store)
    probability = model.predict_proba([features])[0][1]
    return PredictResponse(probability=probability, ...)

@app.post("/ingest")
async def ingest(event: EventPayload) -> IngestResponse:
    """Store event for future predictions."""
    store.store_event(event)
    return IngestResponse(success=True)
```

### 11.5 Event Store (`src/api/store.py`)

```python
class EventStore:
    """In-memory event storage for microservice operation."""
    
    def store_event(self, event: EventPayload):
        """Store event for graph-like feature computation."""
        ...
    
    def get_related_insertions(self, attribute, value):
        """Find listings sharing an attribute (for link prediction)."""
        ...
    
    def count_related(self, attribute, value):
        """Count listings with same attribute (graph-like feature)."""
        ...
```

### 11.6 Real-time Feature Engineering (`src/api/features.py`)

```python
class FeatureEngineer:
    """Real-time feature extraction from events."""
    
    def extract(self, event: EventPayload, store: EventStore) -> dict:
        features = {}
        
        # SEON features
        features.update(self._extract_seon_features(event))
        
        # Listing features
        features.update(self._extract_listing_features(event))
        
        # Graph-like features (from store)
        features['email_link_count'] = store.count_related('email', event.email)
        features['device_link_count'] = store.count_related('device', event.device)
        
        return features
```

---

## Appendix A: Makefile Targets

```makefile
# Data
make etl                    # Run ETL pipeline

# Training
make train-lr               # Logistic Regression
make train-rf               # Random Forest
make train-vanilla          # Vanilla XGBoost
make train-gnn              # GNN + XGBoost
make compare-all            # All 4 variants

# HPO
make hpo                    # XGBoost HPO
make hpo-gnn               # Joint GNN+XGBoost HPO

# Expanding Window
make expanding-vanilla      # Concept drift validation

# API
make api                    # Start production API
make api-dev               # Start dev API with reload

# Utilities
make mlflow                # Start MLflow UI
```

---

## Appendix B: Key Dependencies

```
polars>=0.20.0              # Data processing
xgboost>=2.0.0              # Classification
torch>=2.0.0                # Deep learning
torch-geometric>=2.4.0       # Graph neural networks
hydra-core>=1.3.0           # Configuration
hydra-optuna-sweeper>=1.2.0 # HPO
mlflow>=2.9.0               # Experiment tracking
fastapi>=0.100.0            # API service
shap>=0.42.0                # Explainability
```

---

*Document Version: 1.0 | Last Updated: December 17, 2025*
