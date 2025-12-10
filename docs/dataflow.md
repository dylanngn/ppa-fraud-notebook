# Data Engineering Pipeline

**Purpose**: Document how data flows through the system from database to model predictions.

> 📁 For technical architecture, see [`architecture.md`](architecture.md)  
> 📊 For domain knowledge (fields, relations), see [`knowledge_base.md`](knowledge_base.md)  
> 📓 For experiments, see [`experiment_journal.md`](experiment_journal.md)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA ENGINEERING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Aurora DB]                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌────────────────┐                                                         │
│  │ ETL Pipeline   │  Extract, Transform, Load                               │
│  │ (src/data/etl) │                                                         │
│  └────────┬───────┘                                                         │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────────┐                                                   │
│  │ raw_insertions.parquet│  234k rows × 292 cols                            │
│  └────────┬─────────────┘                                                   │
│           │                                                                 │
│           ├───────────────────────┬──────────────────┐                      │
│           ▼                       ▼                  ▼                      │
│  ┌────────────────┐      ┌────────────────┐  ┌──────────────┐             │
│  │ Graph Artifacts│      │ Feature        │  │ SEON         │             │
│  │ Creation       │      │ Processing     │  │ Baseline     │             │
│  └────────┬───────┘      └────────┬───────┘  └──────────────┘             │
│           │                       │                                         │
│           ▼                       ▼                                         │
│  nodes_*.parquet           ~235 features                                    │
│  edges_*.parquet           (218 tabular + 17 graph)                        │
│           │                       │                                         │
│           └───────────┬───────────┘                                         │
│                       ▼                                                     │
│              ┌─────────────────┐                                            │
│              │ XGBoost Training│                                            │
│              └────────┬────────┘                                            │
│                       │                                                     │
│                       ▼                                                     │
│              ┌─────────────────┐                                            │
│              │ Predictions +   │                                            │
│              │ MLflow Tracking │                                            │
│              └─────────────────┘                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: ETL Pipeline

### Extract (Database → Memory)

**Script**: `src/data/etl/extract.py`

```python
# Connects to Aurora PostgreSQL
# Extracts ~234k listings with fraud labels
query = """
    SELECT 
        listing.*,
        fraud.flag as fraud_flag,
        COALESCE(fraud.flag, FALSE) as is_fraud
    FROM listings
    LEFT JOIN fraud_labels ON listings.id = fraud_labels.listing_id
    WHERE submission_at >= '2023-01-01'
"""
```

**Output**: Polars DataFrame in memory

### Transform (Memory → Memory)

**Script**: `src/data/etl/transform.py`

**Operations**:
1. **JSON Flattening**:
   ```python
   # Before: listing.characteristics = {"hasBalcony": true, ...}
   # After:  listing.characteristics.hasBalcony = true
   ```

2. **Type Conversion**:
   ```python
   # Dates: str → datetime[μs, UTC]
   # Booleans: {"true", "false", null} → {True, False, None}
   # Numbers: str → float64/int64
   ```

3. **Hash Generation**:
   ```python
   # PII fields → SHA256 hashes
   # email → email.hash
   # phone → phone.hash
   ```

4. **NULL Semantics**:
   ```python
   # Boolean indicators: Keep NULL as-is (NULL = FALSE)
   # Don't fill with 0 or False!
   ```

**Output**: Polars DataFrame (flattened, typed, hashed)

### Load (Memory → Disk)

**Script**: `src/data/etl/load.py`

```python
# Write to parquet with compression
df.write_parquet(
    "artifacts/raw_insertions.parquet",
    compression="snappy"
)
```

**Output**: `artifacts/raw_insertions.parquet`
- **Rows**: 234,458
- **Columns**: 292
- **Size**: ~150MB (compressed)

---

## Stage 2: Graph Artifacts Creation

### Purpose

Create node and edge tables for graph-based analysis.

**Script**: `src/data/graph/create_artifacts.py`

### Node Creation

```python
# Create nodes_listing.parquet
# Pass-through all raw columns for downstream use
nodes_listing = raw_df.select([
    "object_reference",  # Unique ID
    *[col for col in raw_df.columns]  # All raw columns
])
```

**Output**: `artifacts/nodes_listing.parquet`
- **Rows**: 234,458 (1 node per listing)
- **Columns**: Same as raw (292)
- **Purpose**: Single source for both GNN and graph feature computation

### Edge Creation

```python
# For each edge type (e.g., shared_contact_email):
edges_email = raw_df.groupby("listing.lister.billing.email.hash").agg([
    pl.col("object_reference").alias("source"),
])

# Create all pairs within each group (fraud ring detection)
# Save as: artifacts/edges_shared_contact_email.parquet
```

**Edge Types Created** (8 total):
| File | Edge Type | Coverage |
|------|-----------|----------|
| `edges_shared_contact_email.parquet` | shared_contact_email | 98% |
| `edges_shared_billing_phone.parquet` | shared_billing_phone | 98% |
| `edges_shared_ip.parquet` | shared_ip | 84% |
| `edges_shared_contact_phone.parquet` | shared_contact_phone | 70% |
| `edges_same_owner.parquet` | same_owner | 100% |
| `edges_same_ppa_person.parquet` | same_ppa_person | 94% |
| `edges_same_region.parquet` | same_region | 100% |
| `edges_same_street.parquet` | same_street | 100% |

**Output**: 8 parquet files
- **Format**: `(source, target, weight)` triples
- **Size**: ~500MB total (uncompressed ~2GB)

---

## Stage 3: Feature Processing

### FeatureProcessor (Auto Mode)

**Script**: `src/features/processor.py`

```python
from src.features.xgboost.processor import FeatureProcessor

processor = FeatureProcessor(
    categories=['base', 'graph']  # Auto mode
)

df_with_features, feature_cols = processor.process(raw_df)
```

### Step 1: Tabular Features (Auto-Selected)

**Category**: `base`

```python
# Auto-select ALL columns except exclusions
tabular_features = [
    col for col in raw_df.columns
    if col not in EXCLUDED_COLUMNS
    and not matches_excluded_pattern(col)
]
```

**Exclusions** (from `src/models/config/constants.py`):
```python
EXCLUDED_COLUMNS = {
    # Labels
    "fraud_flag", "is_fraud",
    
    # IDs
    "object_reference", "user_id", "owner_id",
    
    # Timestamps (for splitting, not features)
    "submission_at", "first_published_date",
    
    # External outputs (data leakage!)
    "seonApproved", "seonFraudScore", "seonSession",
    
    # High cardinality
    "ppaPersonId",  # Use in graph, not as feature
}

EXCLUDED_PATTERNS = [
    "*_hash",      # Hashed identifiers (use in graph only)
    "legacy.*",    # Old system fields
]
```

**Result**: ~218 tabular features

### Step 2: Graph Features (Explicit Computation)

**Category**: `graph`

**Script**: `src/features/definitions/graph.py`

```python
def compute_graph_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute graph-derived features from edge files.
    """
    # Basic features
    df = generate_graph_features(df)  # PageRank, degree, etc.
    
    # Advanced features
    df = generate_advanced_features(df)  # Clustering, betweenness
    
    return df
```

**Graph Features Computed** (17 total):
```python
GRAPH_FEATURES = [
    # Basic connectivity
    "listing_pagerank",
    "listing_in_degree",
    "listing_out_degree",
    "listing_component_size",
    
    # User-level
    "user_listing_count",
    "user_fraud_rate",
    
    # Shared contacts
    "shared_contact_email_count",
    "shared_billing_phone_count",
    "shared_ip_user_count",
    "max_shared_contact_email",
    "max_shared_billing_phone",
    
    # Advanced
    "listing_clustering_coef",
    "listing_betweenness",
    "listing_eigenvector_centrality",
    
    # Temporal
    "account_age_days",
    "user_avg_time_between_listings",
    "is_first_listing",
]
```

**Result**: 17 graph features added to DataFrame

### Step 3: Categorical Encoding

```python
def _encode_categorical_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Encode categorical columns for XGBoost.
    
    Strategy:
    1. Boolean → Int8 (0/1)
    2. String (cardinality ≤ 200) → LabelEncoder (0, 1, 2, ...)
    3. String (cardinality > 200) → Drop (too sparse)
    4. Datetime → Keep as-is (if in always_keep)
    """
    encoded_df = df.copy()
    
    for col in df.columns:
        dtype = df[col].dtype
        
        if dtype == pl.Boolean:
            encoded_df = encoded_df.with_columns(
                pl.col(col).cast(pl.Int8)
            )
        elif dtype in (pl.Utf8, pl.String):
            n_unique = df[col].n_unique()
            if n_unique <= MAX_CARDINALITY:  # 200
                # Label encode
                encoded_df = encoded_df.with_columns(
                    pl.col(col).rank("dense").cast(pl.Int32)
                )
            else:
                # Drop (too high cardinality)
                encoded_df = encoded_df.drop(col)
    
    return encoded_df
```

**MAX_CARDINALITY**: 200
- Lower = fewer features, faster training
- Higher = more information, risk of overfitting
- 200 is empirically validated balance

### Step 4: Schema Pre-Determination (For Consistency)

```python
# For train/test consistency:
# 1. Determine schema from FULL dataset ONCE
encoding_schema = FeatureProcessor.determine_encoding_schema(
    full_df, 
    keep_cols={'submission_at'}
)

# 2. Apply same schema to train and test
train_processor = FeatureProcessor(predetermined_schema=encoding_schema)
test_processor = FeatureProcessor(predetermined_schema=encoding_schema)

train_df, train_features = train_processor.process(train_df)
test_df, test_features = test_processor.process(test_df)
```

**Why This Matters**:
- Train and test must have **identical columns** for XGBoost
- Categorical encoding can vary if train/test have different values
- Pre-determination ensures consistency across temporal splits

**Final Output**:
- **~235 features** (218 tabular + 17 graph)
- All numeric (int, float, or label-encoded)
- Consistent across train/test splits
- Ready for XGBoost

---

## Stage 4: Model Training

### Temporal Split

```python
# Split by time (not random!)
cutoff_date = datetime(2024, 10, 1, tzinfo=timezone.utc)

train_df = df.filter(pl.col("submission_at") < cutoff_date)
test_df = df.filter(pl.col("submission_at") >= cutoff_date)
```

**Why Temporal**:
- Respects real-world deployment (train on past, predict future)
- Prevents data leakage (no future information in training)
- Validates concept drift mitigation

### Training Orchestration (New Architecture)

**Key Change**: Temporal splitting now in `src/features/temporal_split.py`, training orchestrated by `train.py`.

```python
# Orchestrator coordinates the pipeline
@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Load dataset
    df = load_data()
    
    # 2. Create temporal splitter (from features layer)
    splitter = AccumulatingWindowSplitter(
        df=df,
        initial_window_days=180,
        step_days=14,
        test_days=14
    )
    
    # 3. Train per window using generic trainer
    for window_idx, train_data, test_data, window_info in splitter.split():
        # Process features with temporal cutoff
        train_processed, feature_cols = feature_processor.process(
            train_data, cutoff_date=window_info['train_end']
        )
        test_processed, _ = feature_processor.process(
            test_data, cutoff_date=window_info['train_end']
        )

        # Generic trainer (no splitting logic)
        result = train_single_window(
            train_df=train_processed.to_pandas(),
            test_df=test_processed.to_pandas(),
            feature_cols=feature_cols,
            target_col="is_fraud",
            xgb_params=dict(cfg.model.params)
        )
```

**Output**:
- Trained XGBoost models (one per window)
- Per-window metrics (logged to MLflow)
- Best model registered to MLflow Model Registry

---

## Stage 5: Results & Tracking

### MLflow Logging

```python
with mlflow.start_run():
    mlflow.log_params({
        "n_features": len(feature_cols),
        "learning_rate": 0.1,
        "max_depth": 6,
    })
    
    mlflow.log_metrics({
        "auc_pr": 0.67,
        "auc_roc": 0.95,
        "precision": 0.87,
    })
    
    mlflow.sklearn.log_model(model, "model")
```

### Results Registry

```python
from src.experiments.results import save_result, ExperimentID

save_result(
    ExperimentID.EXP1_GRAPH_VALUE,
    {
        "auc_pr": 0.67,
        "auc_roc": 0.95,
        "precision": 0.87,
        "recall": 0.75,
    }
)
```

**Output**: `artifacts/results/exp01_graph_value.json`

---

## Data Quality & Validation

### Feature Quality Checks

```python
# Check for leakage
assert "seonApproved" not in feature_cols
assert "fraud_flag" not in feature_cols

# Check for consistency
assert set(train_features) == set(test_features)

# Check for missing values
assert train_df[feature_cols].null_count().sum() < 0.05 * len(train_df)
```

---

## Pipeline Execution

### Manual (Step-by-Step)

```bash
# 1. ETL (once)
python -m src.data.etl.pipeline

# 2. Graph artifacts (once or weekly)
python -m src.data.graph.create_artifacts

# 3. SEON baseline (once)
python -m src.utils.evaluate_seon

# 4. Train model
python -m src.models.train

# 5. Run experiments
python -m src.experiments.exp01_graph_value
```

### Automated (Weekly Schedule)

```bash
#!/bin/bash
# cron job: 0 2 * * 1  (Every Monday 2 AM)

# Incremental ETL
python -m src.data.etl.pipeline --incremental

# Update graph (only new nodes/edges)
python -m src.data.graph.create_artifacts --incremental

# Retrain (accumulating window)
python -m src.models.train experiment_name=weekly-retrain

# Log to MLflow
# If AUC-PR > 0.65: promote to production
```

---

## Performance Characteristics

### Pipeline Timings

| Stage | Time | Memory | Disk I/O |
|-------|------|--------|----------|
| **ETL** | ~2 min | ~4GB | 150MB write |
| **Graph artifacts** | ~5 min | ~8GB | 500MB write |
| **Feature processing** | ~30 sec | ~2GB | Read only |
| **XGBoost training** | ~3 min | ~4GB | Read only |
| **Total** | **~10 min** | **8GB peak** | **650MB** |

### Bottlenecks

| Component | Bottleneck | Solution |
|-----------|------------|----------|
| **ETL** | Database connection | Use read replica |
| **Graph creation** | Memory (10M+ edges) | Incremental updates |
| **Feature encoding** | CPU (label encoding) | Cache encoders |
| **Training** | Time (large dataset) | Use GPU XGBoost |

---

## Data Artifacts Summary

| Artifact | Size | Rows | Cols | Purpose |
|----------|------|------|------|---------|
| `raw_insertions.parquet` | 150MB | 234k | 292 | Source data |
| `nodes_listing.parquet` | 150MB | 234k | 292 | Graph nodes |
| `edges_*.parquet` (8 files) | 500MB | ~10M | 3 | Graph edges |
| `results/*.json` | <1MB | - | - | Experiment results |
| `graph.pt` (optional) | 2GB | - | - | PyG graph (Phase 2) |

**Total**: ~800MB (Phase 1), ~3GB (with GNN)

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'seonApproved'` | Column excluded but code expects it | Check exclusions in `constants.py` |
| `ValueError: feature_names mismatch` | Train/test column mismatch | Use `predetermined_schema` |
| `MemoryError` | Graph too large | Use incremental processing |

### Validation Commands

```bash
# Check ETL output
python -c "import polars as pl; print(pl.read_parquet('artifacts/raw_insertions.parquet').shape)"
# Expected: (234458, 292)

# Check graph artifacts
ls -lh artifacts/edges_*.parquet
# Expected: 8 files

# Check features
python -c "from src.features.xgboost.processor import FeatureProcessor; p = FeatureProcessor(categories=['base', 'graph']); print(len(p.get_all_features()))"
# Expected: ~235
```

---

## Future Enhancements

### Incremental Updates

```python
# Only process new listings since last run
last_run = load_last_run_timestamp()
new_listings = extract_listings_since(last_run)

# Append to existing parquet
df_old = pl.read_parquet("raw_insertions.parquet")
df_new = pl.concat([df_old, new_listings])
df_new.write_parquet("raw_insertions.parquet")
```

### Feature Store

```python
# Cache computed features
feature_store = {
    "listing_xyz": {
        "listing_pagerank": 0.042,
        "user_listing_count": 15,
        # ... all 235 features
    }
}

# Lookup during inference (no recomputation)
features = feature_store.get("listing_xyz")
```

### Real-Time Pipeline

```python
# Kafka/Pub-Sub integration
@app.route("/predict", methods=["POST"])
def predict():
    listing_data = request.json
    features = feature_processor.process_single(listing_data)
    prediction = model.predict_proba(features)[0, 1]
    return {"fraud_probability": prediction}
```

---

## References

- ETL code: `src/data/etl/`
- Graph creation: `src/data/graph/create_artifacts.py`
- Feature processing: `src/features/processor.py`
- Training orchestration: `src/models/train.py` (XGBoost), `src/models/gnn/sage.py` (Hybrid)
- Temporal splitting: `src/features/temporal_split.py`
- Generic trainer: `src/models/xgboost/trainer.py`
- Technical architecture: [`architecture.md`](architecture.md)
- Domain knowledge: [`knowledge_base.md`](knowledge_base.md)
