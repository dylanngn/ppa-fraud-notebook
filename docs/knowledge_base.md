# Knowledge Base: Fraud Detection Feature Engineering

This document tracks our understanding of the data, feature definitions, and engineering logic.

---

## Data Quality Overview

### Dataset Summary

| Metric | Value |
|--------|-------|
| **Source** | `artifacts/raw_insertions.parquet` |
| **Total Rows** | 234,458 |
| **Total Columns** | 292 |
| **Usable Fields** (<50% null) | 82 (28%) |
| **Unusable Fields** (≥95% null) | 175 (60%) |
| **Account Date Range** | 2020-12-17 → 2025-11-25 |
| **Listing Date Range** | 2023-01-01 → 2025-11-11 |

### Contact Field Coverage

| Field Type | Primary Field | Coverage | Unique Values |
|------------|---------------|----------|---------------|
| **Email (lister)** | `listing.lister.email.hash` | 99.98% | 130,947 |
| **Email (billing)** | `listing.lister.billing.email.hash` | 98.00% | 129,918 |
| **Phone (billing)** | `listing.lister.billing.phoneData.hash` | 97.99% | 127,770 |
| **Phone (contact)** | `listing.lister.phone.hash` | 69.98% | 89,614 |
| **Address** | `listing.address.address_hash` | 100.0% | ~234k |
| **Coordinates** | `listing.address.geoCoordinates.lat/lng` | 99.40% | 107,168 |

> ⚠️ **Key Insight:** Billing phone (98%) has much higher coverage than contact phone (70%). Use billing phone for graph edges.

### Data Integrity

| Check | Result |
|-------|--------|
| Duplicate `object_reference` | ✅ 100% unique |
| Empty `object_reference` | ✅ None |
| Flattened listing coverage | ✅ 98.21% |

### Data Type Distribution

| Type | Count | Notes |
|------|-------|-------|
| String | 171 | Mostly hashed identifiers |
| Boolean | 54 | Characteristics flags |
| Int64 | 29 | IDs, counts |
| Float64 | 20 | Prices, coordinates |
| Datetime | 5 | Timestamps |
| Null | 12 | Completely empty columns |

> 📊 Full data quality report: [`docs/data_quality_reports/data_quality_report.md`](data_quality_reports/data_quality_report.md)

---

## Project Structure

```
ppa-fraud-notebook/
├── conf/                    # Hydra configuration
│   ├── config.yaml          # Main config (defaults)
│   ├── data/default.yaml    # Data extraction settings
│   ├── features/            # Feature profiles (quick, standard, production)
│   └── model/xgboost.yaml   # Model hyperparameters
├── src/
│   ├── data/                # ETL and graph building
│   ├── features/            # Feature engineering
│   ├── models/              # XGBoost and GNN training
│   └── utils/               # Metrics, MLflow helpers
├── artifacts/               # Generated data files
└── docs/                    # Documentation
```

### Key Commands

```bash
# Data Pipeline (run once)
make etl              # Extract, transform, load
make build-graph      # Build graph artifacts
make seon-baseline    # Generate static Seon baseline

# Model Training
make train-baseline   # XGBoost with production features
make train-quick      # XGBoost with core features (fast iteration)
make train-sage       # Train SAGE hybrid (GNN + XGBoost)
```

---

## 1. Data Sources & Identifiers

| Field | Type | Description |
|-------|------|-------------|
| `object_reference` | Text | Primary business ID for listing |
| `owner_id` | Text | User identifier |
| `platform` | Text | Brand (Homegate, ImmoScout24, SMG) |
| `submission_at` | Datetime | When listing was submitted |
| `fraud_flag` | Datetime | If present, listing was marked as fraud |

---

## 2. Fraud Logic & Labels

- **Fraud Flag**: `fraud_flag` timestamp indicates listing was marked as fraud
- **Seon Baseline**: `auto_approval_criteria` contains production Seon check result
- **Target**: Catch "slip-through" fraud (approved by Seon but later caught)

---

## 3. Feature Tiers (Data Quality Based)

### Tier 1: Core Features (High Coverage >80%)

| Feature | Coverage | Source Field | Description |
|---------|----------|--------------|-------------|
| `account_age_days` | 100% | `account_created_at` | Days since account creation |
| `payment_type` | 98% | `listing.selectedBundle` | DIRECT vs INVOICE (17.5x fraud lift for new+invoice) |
| `bundle_tier` | 97.7% | `listing.selectedBundle.name` | basic, premium, top |
| `bundle_period` | 77.25% | `listing.selectedBundle.duration` | Duration in days |
| `log_price` | 80% | `listing.prices.sell/rent` | Log-transformed price |
| `latitude` | 99.4% | `listing.address.geoCoordinates.lat` | Property latitude |
| `longitude` | 99.4% | `listing.address.geoCoordinates.lng` | Property longitude |
| `offer_type` | 100% | `listing.offerType` | BUY vs RENT |
| `living_space` | 84.7% | `listing.characteristics.livingSpace` | Square meters |
| `rooms` | 92.7% | `listing.characteristics.numberOfRooms` | Number of rooms |

### Tier 2: Graph Features (Computed)

| Feature | Description |
|---------|-------------|
| `shared_contact_email_count` | Emails shared with other listings |
| `listing_component_size` | Size of connected component |
| `listing_pagerank` | PageRank centrality |
| `is_isolated` | Boolean: no graph connections |
| `neighbor_overlap_score` | Clustering coefficient proxy |

### Tier 3: Time-Weighted Features (Computed)

| Feature | Description |
|---------|-------------|
| `email_time_spread` | Days between first and last email reuse |
| `email_recency_weighted` | Exponentially-weighted email connections |
| `combined_recency_weighted` | Email + phone combined |

### Deprecated Features (Low Coverage)

| Feature | Coverage | Reason |
|---------|----------|--------|
| `is_new` | 0.04% | Almost always NULL |
| `has_elevator` | 40% | Below 50% threshold |
| `hasSwimmingPool` | 4.75% | 95.25% null |
| `isSmokerFriendly` | 4.88% | 95.12% null |
| `cubage` | 4.92% | 95.08% null |
| `yearBuilt` | 53.05% | Borderline coverage, low signal |
| Contact phone fields | 69.98% | Use billing phone (97.99%) instead |
| Burst detection | N/A | No impact in experiments |
| Interaction features | N/A | XGBoost learns automatically |

> 📉 **175 columns** (60% of dataset) have ≥95% null values and are excluded from feature engineering.

---

## 4. Training Strategy

### Accumulating Window (Production-Realistic)

```
Window 1: Train on [2023-01-01, 2023-06-30] → Test [Jul 1-14]
Window 2: Train on [2023-01-01, 2023-07-07] → Test [Jul 8-21]
...
Window N: Train on [2023-01-01, current] → Test [next 14 days]
```

**Why Accumulating?**
- Reflects production continuous learning
- GNNs need complete graph history
- Better fraud pattern coverage

### Configuration

```yaml
# conf/model/xgboost.yaml
training:
  initial_window_days: 180  # Start with 6 months
  step_days: 7              # Weekly evaluation
  max_windows: null         # All windows (or set for debugging)
```

---

## 5. Graph Structure

### Node Types

| Node Type | ID Field | Unique Count | Features |
|-----------|----------|--------------|----------|
| **User** | `owner_id` | 133,811 | `account_created_at` |
| **Listing** | `insertion_id` | 234,458 | All listing features + embeddings |
| **Email** | Email hash | 130,947 | (constant) |
| **Phone** | Phone hash | 127,770 | (constant) |
| **Address** | `Country_Zip_City_Street` | ~234k | `latitude`, `longitude` |
| **IP** | IP hash | — | (constant) |

### Edge Types (8 total, simplified)

| Edge | Source → Target | Coverage | Source Field |
|------|-----------------|----------|--------------|
| `posts` | User → Listing | 100% | `user_id` |
| `uses` | User → IP | 94.7% | `meta.ip` |
| `has_email` | User → Email | 99.98% | `listing.lister.email.hash` |
| `has_contact_email` | Listing → Email | 99.98% | `listing.lister.email.hash` |
| `has_billing_email` | Listing → Email | 98.00% | `listing.lister.billing.email.hash` |
| `has_billing_phone` | Listing → Phone | 97.99% | `listing.lister.billing.phoneData.hash` |
| `located_at` | Listing → Address | 99.99% | `listing.address.city_hash` + `zip_hash` |
| `billing_address` | Listing → Address | 98% | `listing.lister.billing.address` |

**Note**: Use billing phone (98% coverage) over contact phone (70%) for graph edges.

---

## 6. Model Comparison Framework

### Models

| Model | Type | Command |
|-------|------|---------|
| **Seon** | Production Baseline | Pre-computed: `artifacts/seon_baseline.json` |
| **XGBoost** | Primary Model | `make train` |
| **GNN Hybrid** | Alternative | `make train-sage` or `make train-hgt` |

### XGBoost (Primary Model)

```bash
make train  # or: python -m src.models.train features=production
```

- Uses Tier 1-3 handcrafted features
- No GPU required
- Training: ~15-20 minutes
- Interpretable via SHAP

### GNN Hybrid (Alternative Approach)

```bash
make train-sage  # or make train-hgt
```

- Trains GNN on graph structure
- Generates 64-dim embeddings
- Feeds embeddings + tabular to XGBoost
- GPU required, 2-3 hours training

### Comparison Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| AUC-PR | Area under precision-recall curve | >0.70 (beat Seon) |
| P@100 | Precision at top 100 predictions | >0.75 |
| Training Time | Wall-clock minutes | <20 min (XGBoost) |
| Inference Latency | Milliseconds per prediction | <100ms |

---

## 7. MLflow Integration

### Experiment Tracking

```python
# Automatically tracked:
- Hyperparameters
- Per-window metrics (AUC-PR, P@100)
- Aggregate metrics (mean, best)
- Feature importance
- Models (registered to Model Registry)
```

### Model Registry

```bash
# Compare candidate vs production
make mlflow-compare-models MODEL_NAME=fraud-detection-xgboost CANDIDATE_RUN_ID=xxx

# Get deployment recommendation
make mlflow-deployment-recommendation MODEL_NAME=fraud-detection-xgboost CANDIDATE_RUN_ID=xxx
```

---

## 8. Key Research Findings

### Fraud Pattern Insights

1. **Payment Type is Critical**: New accounts + invoice payment = 28% fraud rate (17.5x lift)
2. **Fraud is Isolated**: Fraudsters avoid creating large networks (breaks GNN homophily assumption)
3. **Long-Term Patterns**: Fraud infrastructure reused over 12+ months

### Model Architecture Insights

1. **Handcrafted features are explicit**: XGBoost sees exact counts directly
2. **GNNs need complete history**: Sliding windows hurt GNN performance
3. **XGBoost learns interactions**: Explicit interaction features don't help tree models

### Best Practices

1. Use **accumulating windows** for production deployment
2. Prioritize **billing phone** over contact phone (98% vs 70% coverage)
3. Exclude **low-coverage features** (<50% coverage) to avoid noise
4. Focus on **82 high-coverage fields** for feature engineering
5. Drop **175 unusable columns** (≥95% null) to reduce dimensionality
6. Use **email hash** (99.98% coverage) as primary contact identifier for graph
7. Validate **data integrity** before training (no duplicates, no empty IDs)