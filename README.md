# Fraud Detection in Online Real Estate Marketplaces

**A Hybrid Graph and Gradient Boosting Approach**

---

## 📊 Research Overview

This project implements a fraud detection system for real estate listings using:
- **XGBoost** as the primary classifier with auto-selected features
- **Graph-based analysis** to detect fraud rings via shared contacts
- **Temporal splitting** to prevent data leakage and validate concept drift mitigation
- **SHAP** for explainability and actionable insights

### Dataset

- **234,458 listings** from a Swiss real estate marketplace (2023-2025)
- **8.2% fraud rate** (19,217 fraud cases)
- **292 raw ETL fields** → ~235 features (218 tabular + 17 graph)
- **8 edge types** for graph-based fraud ring detection

### Baseline

- **SEON** (external fraud detection system): AUC-PR = 0.229
- **Goal**: Demonstrate that graph-based features provide significant lift over SEON

---

## 🎯 Research Questions

| RQ | Question | Approach |
|----|----------|----------|
| **RQ1** | What novel relational indicators of fraud can be identified using graph-based analysis? | Graph feature ablation study |
| **RQ2** | How can a hybrid GNN + XGBoost architecture model these indicators? | _(Deferred to Phase 2)_ |
| **RQ3** | Does periodic retraining maintain performance against concept drift? | Accumulating window evaluation (120+ time windows) |
| **RQ4** | Can SHAP translate predictions into actionable insights? | SHAP global importance + interaction analysis |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup

```bash
# 1. ETL: Extract data from database
python -m src.data.etl.pipeline

# 2. Create graph artifacts (nodes + edges)
python -m src.data.graph.create_artifacts

# 3. Compute SEON baseline (one-time)
python -m src.utils.evaluate_seon

# 4. Start MLflow tracking server (separate terminal)
mlflow ui --port 5000
```

```

### View Results

```bash
# MLflow UI
open http://127.0.0.1:5000

# Jupyter notebooks
jupyter notebook notebooks/

```

---

## 📁 Project Structure

```
ppa-fraud-notebook/
├── conf/                    # Hydra configuration
│   ├── config.yaml          # Main config (features=auto)
│   ├── data/                # Database settings
│   ├── features/            # Feature selection modes
│   └── model/               # XGBoost hyperparameters
│
├── artifacts/               # Generated data (gitignored)
│   ├── raw_insertions.parquet      # ETL output (234k rows)
│   ├── nodes_listing.parquet       # Graph nodes
│   ├── edges_*.parquet             # Graph edges (8 types)
│   └── results/                    # Experiment results (JSON)
│
├── src/
│   ├── data/                # ETL + graph creation
│   ├── features/            # Feature engineering
│   ├── models/              # XGBoost + GNN (Phase 2)
│   ├── experiments/         # 5 core experiments
│   └── utils/               # Utilities
│
├── notebooks/               # Jupyter analysis
│
├── docs/                    # Documentation
│   ├── architecture.md      # Technical architecture
│   ├── dataflow.md          # Data pipeline
│   ├── knowledge_base.md    # Domain knowledge
│   └── ...
│
└── README.md                # This file
```

---

## 🔑 Key Features

### 1. Graph-Based Fraud Ring Detection

```python
# 8 edge types for fraud ring detection:
- shared_contact_email (98% coverage)
- shared_billing_phone (98% coverage)
- shared_ip (84% coverage)
- same_owner (100% coverage)
- same_ppa_person (94% coverage)
- same_region (100% coverage)
- same_street (100% coverage)
- shared_contact_phone (70% coverage)
```

### 2. Auto Feature Selection

```yaml
# conf/features/auto.yaml
categories:
  - base   # All tabular features (auto-selected, ~218)
  - graph  # Graph-derived features (17)

# Result: ~235 features total
```

**What's excluded**: Labels, IDs, timestamps, SEON outputs (data leakage!), high-cardinality hashes

### 3. Temporal Split (No Data Leakage)

```python
# Train on past, test on future
cutoff_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
train = df.filter(pl.col("submission_at") < cutoff_date)
test = df.filter(pl.col("submission_at") >= cutoff_date)
```

### 4. MLflow Tracking

All experiments automatically logged:
- Hyperparameters (learning rate, max depth, etc.)
- Metrics (AUC-PR, AUC-ROC, Precision, Recall)
- Artifacts (feature importance, confusion matrix, SHAP plots)

---

## 📚 Documentation

### Core Docs

| Document | Purpose |
|----------|---------|
| **[docs/README.md](docs/README.md)** | Documentation navigation guide |
| **[docs/architecture.md](docs/architecture.md)** | Technical architecture |
| **[docs/dataflow.md](docs/dataflow.md)** | Data pipeline (ETL → features → models) |
| **[docs/knowledge_base.md](docs/knowledge_base.md)** | Domain knowledge (fields, relations) |

### For Developers

1. [`docs/architecture.md`](docs/architecture.md) - Understand technical setup
2. [`docs/dataflow.md`](docs/dataflow.md) - Understand data pipeline
3. [`docs/knowledge_base.md`](docs/knowledge_base.md) - Understand domain

---

## 🛠️ Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| **ML Framework** | XGBoost 2.0+ | Gradient boosting classifier |
| **Graph ML** | PyTorch Geometric 2.4+ | GNN (Phase 2) |
| **Data** | Polars 0.19+ | Fast DataFrame operations |
| **Config** | Hydra 1.3+ | Configuration management |
| **Tracking** | MLflow 2.8+ | Experiment tracking |
| **DB** | psycopg2 2.9+ | PostgreSQL driver |
| **Explainability** | SHAP 0.43+ | Feature importance |

---

## 📈 Expected Results (Post-SEON Fix)

| Metric | SEON Baseline | Our Model (Expected) | Improvement |
|--------|---------------|----------------------|-------------|
| **AUC-PR** | 0.229 | **0.65-0.70** | **+187%** |
| **AUC-ROC** | 0.805 | **~0.95** | +18% |
| **Precision@100** | 0.268 | **~0.85** | **+217%** |

**Note**: Lower than previous (invalid) results due to SEON data leakage fix. These are **honest** metrics!

---

## 🚨 Common Issues

### Issue: "SEON baseline not found"
```bash
python -m src.utils.evaluate_seon
```

### Issue: "Graph artifacts not found"
```bash
python -m src.data.graph.create_artifacts
```

---

## 📝 License

This project is part of a master's thesis. Contact the author for usage permissions.

---

## 👤 Author

Senior Data Scientist  
Master's Thesis: Fraud Detection in Real Estate Marketplaces  
2025

---

## 🔗 Quick Links

- **MLflow UI**: http://127.0.0.1:5000 (after `mlflow ui`)
- **Architecture**: [`docs/architecture.md`](docs/architecture.md)
