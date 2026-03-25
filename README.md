# Fraud Detection in Online Real Estate Marketplaces
## Utilizing a Hybrid Graph and Gradient Boosting Model

**Author:** Nguyen Hoang Minh

This repository contains the source code, data preprocessing pipelines, and
experimental framework for a master's thesis on fraud detection in real estate
marketplaces.  The research investigates graph-based fraud pattern discovery
using Swiss Marketplace Group (SMG) real estate data (804,717 listing events).

---

## Overview

Online marketplace fraud causes significant financial losses and erodes user
trust.  This project builds a **heterogeneous graph** to encode entity-sharing
relationships (device fingerprints, IP addresses, e-mails, phones) and applies
**Graph Neural Network (GNN)** embeddings combined with **XGBoost**
classification.

While GNN architectures reveal unique fraud-ring structures invisible to
standard tabular analysis, we found that a purely tabular *Vanilla XGBoost*
model trained on raw SEON signals presents superior operational trade-offs for
production deployment due to temporal distribution shift in real-world
environments.

---

## Documentation

- **[Data Overview & Discovery](docs/data_overview.md)** — dataset
  characteristics, temporal analysis, geographic distributions, device
  connectivity, and fraud-flag distribution.
- **[System Architecture](docs/architecture.md)** — ETL pipeline (Polars),
  temporal graph construction, and model implementations.
- **[Experiment Journal](docs/experiment_journal.md)** — comprehensive record
  of model evaluations, benchmark comparisons, concept drift analysis, and SHAP
  explainability findings.

---

## Research Contributions

1. **Heterogeneous Graph Pattern Discovery** — constructed a dynamic, temporal
   graph capturing over 31 million edges connecting fraudulent entities across
   shared device networks, e-mails, and IPs.
2. **Concept Drift Mitigation** — evaluated model robustness over expanding
   temporal windows, proving that frequent retraining outweighs raw topological
   complexity in production pipelines.
3. **Interpretability Integration** — incorporated SHAP globally and locally,
   attributing 82.7 % of critical predictive signals to targeted tabular signals
   extracted through SEON.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ETL (raw CSV → Parquet feature store)
make etl

# 3. Train the default model (Vanilla XGBoost)
make train-vanilla

# 4. Start the MLflow UI to inspect results
make mlflow
```

---

## Training

All training runs are tracked in MLflow under three experiments:

| Experiment | Contents |
|---|---|
| `fraud-detection` | All single-split training runs |
| `fraud-detection-expanding` | Expanding-window concept drift runs |
| `fraud-detection-hpo` | Optuna HPO sweep trials |

Runs are tagged with `model.variant`, `model.encoder`, and `training.mode`
so they can be filtered in the MLflow UI without navigating separate
experiments.

### Common commands

```bash
make train-vanilla        # Vanilla XGBoost (recommended for production)
make train-gnn            # Hybrid GNN+XGBoost (GraphSAGE encoder)
make compare-all          # Train all 4 variants on the same split
make hpo                  # XGBoost HPO (50 Optuna trials)
make expanding-vanilla    # Expanding window / concept drift analysis
```

Custom date ranges:
```bash
make train-vanilla TRAIN_START=2024-12-01 TRAIN_END=2025-06-01 TEST_END=2025-07-01
```

---

## MLflow Model Registry

After training, the best model is automatically registered to the
`fraud-detection` entry in the MLflow Model Registry.

Promote a version to production:
```bash
make register-model                  # promotes latest version
make register-model MODEL_VER=3      # promotes a specific version
```

Or use the MLflow UI (`make mlflow`) → Models → `fraud-detection` → set alias.

---

## Serving API

The FastAPI service loads the `production`-aliased model from the registry and
exposes it for real-time scoring.

```bash
make api          # Start on port 8000 (production)
make api-dev      # Start with hot-reload (development)
```

Key endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Score a listing event |
| `POST` | `/ingest` | Store an event (no prediction) |
| `POST` | `/bulk-ingest` | Bootstrap with historical events |
| `GET` | `/model-info` | Loaded model metadata & thresholds |
| `GET` | `/health` | Service health + store stats |
| `POST` | `/rebuild-graph` | Refresh graph link-count indexes |

Interactive docs: `http://localhost:8000/docs`

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///ppa-fraud-detection-mlflow.db` | MLflow backend |
| `MLFLOW_MODEL_NAME` | `fraud-detection` | Registered model name |
| `MLFLOW_MODEL_ALIAS` | `production` | Registry alias to load |
| `MODEL_PATH` | _(unset)_ | Direct XGBoost file path (bypasses registry) |
| `THRESHOLD_HIGH` | `0.7` | Probability → HIGH / DECLINE |
| `THRESHOLD_MEDIUM` | `0.3` | Probability → MEDIUM / REVIEW |
| `MAX_EVENTS` | `1000000` | In-memory event store capacity |
| `STORE_PERSIST_PATH` | `artifacts/api/event_store.json` | Store snapshot path |

---

## Production Operation Guide

### Does the model accept new insertion events as real-time input?

**It depends on the model variant.**

#### Vanilla XGBoost — fully real-time ✓

The `vanilla_xgboost` variant takes a flat feature vector as input and
produces a fraud score in milliseconds.  Every new listing-submission event can
be scored immediately:

```
Client → POST /predict (EventPayload)
              ↓
    FeatureEngineer.compute_features()
      - Raw SEON signals (IP, email, phone, session)
      - Listing attributes (price, rooms, category)
      - Temporal features (hour, weekday, …)
      - Graph-link counts (shared IP / device / email — from EventStore)
              ↓
    XGBoost.predict_proba()
              ↓
    PredictResponse (fraud_probability, risk_tier, decision)
```

The `EventStore` accumulates events in memory so that graph-link counts grow
more accurate over time.  Persist it across restarts via `STORE_PERSIST_PATH`.

#### GNN+XGBoost — batch graph rebuild required ✗

The `gnn_xgboost` variant generates node embeddings by running a GNN over the
full heterogeneous graph of all historical listings.  This graph cannot be
updated incrementally in real time because:

1. The GNN must see the *complete* neighbourhood of each node to produce stable
   embeddings.
2. Adding a new node changes the neighbourhoods of existing nodes, requiring a
   full re-inference pass.
3. The graph for the research dataset contains ~31 million edges — rebuilding
   takes minutes to hours, not milliseconds.

**Recommended production strategy for GNN+XGBoost:**

```
Offline (scheduled, e.g. nightly):
  1. Export EventStore → DataFrame
  2. Run: make train-gnn  (or a custom retraining script)
  3. Promote new model version: make register-model
  4. Restart / reload the API (picks up the new 'production' alias)

Online (real-time, between rebuilds):
  - Serve with the most recently trained model
  - New listings that arrived after the last rebuild receive a
    zero-vector GNN embedding (graceful degradation — XGBoost still
    uses all 65 tabular features)
  - Call POST /rebuild-graph after bulk-ingesting historical data
    to refresh the in-memory link-count indexes
```

**For this research, we recommend deploying the `vanilla_xgboost` variant**
in production.  It achieves comparable AUC-PR with zero retraining latency and
without the graph-rebuild dependency.

### Retraining cadence

Concept drift analysis (expanding window results, RQ3) shows that model
performance degrades measurably after ~90 days without retraining.
Recommended schedule:

- **Vanilla XGBoost**: retrain monthly, or when AUC-PR drops below 0.60.
- **GNN+XGBoost**: retrain monthly (graph rebuild is the bottleneck).

---

## Citing This Work

```bibtex
@mastersthesis{minh2024frauddetection,
  author = {Nguyen Hoang Minh},
  title  = {Fraud Detection in Online Real Estate Marketplaces:
             Utilizing a Hybrid Graph and Gradient Boosting Model},
  school = {FPT School of Business and Technology},
  year   = {2024},
  month  = {December}
}
```
