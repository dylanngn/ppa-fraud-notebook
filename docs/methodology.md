# Research Methodology: Fraud Detection in Online Real Estate Marketplaces

**Project:** Hybrid GNN + XGBoost Fraud Detection Framework  
**Author:** Nguyen Hoang Minh (24MSE23152)  
**Version:** 2.0  
**Date:** December 19, 2025

---

## Table of Contents

1. [Research Objectives](#1-research-objectives)
2. [Data Collection & Preprocessing](#2-data-collection--preprocessing)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Feature Selection Rationale](#4-feature-selection-rationale)
5. [Temporal Integrity & Label Design](#5-temporal-integrity--label-design)
6. [Model Selection & Design](#6-model-selection--design)
7. [Experiment Design](#7-experiment-design)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Concept Drift Mitigation](#9-concept-drift-mitigation)
10. [Explainability Analysis](#10-explainability-analysis)

---

## 1. Research Objectives

This research addresses four key questions for fraud detection in online real estate marketplaces:

| RQ | Question | Approach |
|----|----------|----------|
| **RQ1** | Can GNN embeddings improve fraud detection over tabular-only models? | Compare Vanilla XGBoost vs GNN+XGBoost |
| **RQ2** | What are the most predictive features for real estate fraud? | SHAP analysis + feature importance |
| **RQ3** | How do we handle concept drift in fraud patterns? | Expanding window validation |
| **RQ4** | Can we provide explainable predictions for fraud analysts? | SHAP local explanations |

---

## 2. Data Collection & Preprocessing

### 2.1 Data Sources

The dataset comes from Swiss Marketplace Group (SMG), combining:

| Source | Description | Records |
|--------|-------------|---------|
| `insertion_events.csv` | Listing lifecycle events (status changes) | ~800K events |
| `seon_transactions.csv` | SEON fraud detection API responses | ~140K transactions |

### 2.2 Data Merge Strategy

**Problem:** Events and SEON transactions are recorded at different times. Which SEON data should be used for each event?

**Solution:** As-of join with temporal ordering:

```
For each event at time T:
  1. Find the closest SEON transaction AFTER time T (preferred)
  2. If none exists, use the closest SEON transaction BEFORE time T
```

**Rationale:** This ensures we use SEON data that was available (or most recently available) at the time of the event, simulating real-world decision-making.

### 2.3 Label Definition

**Fraud Label Source:** `FLAGGEDFORFRAUD` column

- Contains a timestamp when customer support or SEON rules flagged the listing
- If timestamp is non-null → `is_fraud = 1`
- If timestamp is null → `is_fraud = 0`

**Important Discovery:** Fraud is a property of the **listing**, not individual events. A fraudulent listing may have multiple events before being flagged.

### 2.4 Data Sparsity

**Observation:** Early data (Jan-Oct 2024) was sparse with ~2,000 events total, compared to ~60K+ events per month starting November 2024.

**Root Cause:** SEON integration began in November 2024; earlier data lacks SEON features.

**Decision:** Filter to `train_start_date = 2024-12-01` to exclude incomplete data.

---

## 3. Exploratory Data Analysis

### 3.1 Key Dataset Statistics

| Metric | Value | Implication |
|--------|-------|-------------|
| Total events | 804,717 | Large-scale dataset |
| Unique listings | 86,160 | Unit of fraud analysis |
| **Fraud rate (per listing)** | **8.11%** | Moderate imbalance (1:11) |
| Features | 509 columns | Rich feature space |
| Date range | Jan 2024 - Dec 2025 | ~24 months |

**Critical Insight:** Fraud rate must be calculated per **listing** (8.11%), not per event (1.21%). The event-based rate is misleading because legitimate listings have more events (9.7 avg) than fraud listings (4.8 avg).

### 3.2 Fraud Detection Speed

Analysis of time between listing creation (first event) and fraud flagging:

| Time to Flag | % of Fraud |
|--------------|------------|
| Same day (<24h) | 75.0% |
| 1-7 days | 15.1% |
| 1-4 weeks | 7.0% |
| **Median** | **1 hour 40 minutes** |

**Implication:** The current detection system is effective for rapid detection, but we aim to catch fraud even earlier (before manual review).

### 3.3 High-Risk Indicators

Analysis identified several strong fraud predictors:

| Indicator | Fraud Rate | Lift vs Baseline (8.1%) |
|-----------|------------|-------------------------|
| Country: Nepal | 90.5% | 11.2x |
| Device: TV (spoofed) | 85.7% | 10.6x |
| Country: Benin | 66.9% | 8.3x |
| Category: Studio | 35.0% | 4.3x |
| Cross-platform listings | 26.7% | 3.3x |
| Country: France | 31.8% | 3.9x |
| RENT (vs BUY) | 9.47% vs 0.88% | 10.8x |

**Geographic Pattern:** West African countries (Benin, Togo, Ivory Coast) show 60-90% fraud rates—classic real estate scam origins.

### 3.4 Network Analysis

Device fingerprint analysis revealed potential fraud rings:

| Metric | Value |
|--------|-------|
| Devices with 2+ listings | 6,473 (39.4%) |
| Multi-listing devices with fraud | 1,322 (20.4%) |
| Super-connectors (10+ listings) | 1,248 |
| Super-connectors with fraud | 443 (35.5%) |
| Max listings per device | 6,251 |

**Implication:** Device sharing shows network patterns, but investigation revealed legitimate power users share devices more than fraudsters (see Section 6.5).

---

## 4. Feature Selection Rationale

### 4.1 Feature Categories

We organized 71 base features into categories based on their source and fraud-predictive value:

| Category | Count | Rationale |
|----------|-------|-----------|
| **SEON Boolean** | 21 | Binary risk indicators (tor, vpn, disposable, etc.) |
| **SEON Categorical** | 12 | Geographic and session attributes |
| **Listing Numeric** | 10 | Price, rooms, area (fraud often uses unrealistic values) |
| **Listing Categorical** | 7 | Offer type, category, platform (rental studios = high risk) |
| **Temporal** | 4 | Hour, weekday (fraudsters active early morning) |

### 4.2 Raw SEON Signal Importance

The model uses only raw SEON signals (not SEON's ML scores). Key predictive features:

| Feature | Type | Fraud Rate | Legit Rate | Lift |
|---------|------|------------|------------|------|
| `data_center_proxy=True` | Boolean | 15.2% | 2.1% | 7.2x |
| `tor=True` | Boolean | 28.4% | 0.3% | 94.7x |
| `email/domain/disposable=True` | Boolean | 22.1% | 1.8% | 12.3x |
| `ip_country=BJ` | Categorical | 45.2% | 0.1% | 452x |

**Decision:** Use raw SEON signals that indicate suspicious behavior, not SEON's pre-computed fraud scores.

### 4.3 Boolean Signal Selection

**Analysis:** Fraud rate when signal is true:

| Signal | Prevalence | Fraud Rate | Include? |
|--------|------------|------------|----------|
| `phone_is_disposable` | 0.01% | 24.2% | ✓ (high precision) |
| `email/domain/disposable` | 0.29% | 19.9% | ✓ (high precision) |
| `email/domain/free` | 60.6% | 1.7% | ✓ (high coverage) |
| `data_center_proxy` | 0.44% | 0.1% | ✓ (context signal) |

**Decision:** Include all boolean signals. Low-prevalence signals provide high precision; high-prevalence signals provide context.

### 4.4 Graph Identity Columns

For GNN construction, we selected identity columns that connect listings:

| Identity | Rationale |
|----------|-----------|
| `session/device_hash` | Strongest fraud ring indicator (35% of super-connectors) |
| `ip_hash` | Geographic clustering |
| `user_id_hash` | Account-level patterns |
| `LISTING_LISTER_EMAIL_hash` | Email domain patterns |
| `LISTING_LISTER_PHONE_hash` | Phone number sharing |

**Selection Criterion:** Columns that can link listings to reveal fraud rings.

---

## 5. Temporal Integrity & Label Design

### 5.1 The Label Leakage Problem

**Problem:** Fraud labels come from `FLAGGEDFORFRAUD` timestamps that may occur **after** the training cutoff.

**Example:**
- Listing created: 2025-05-15
- Fraud flagged: 2025-06-10
- Train cutoff: 2025-06-01

If we train with `is_fraud=1` for this listing, we're using future information (the June 10 flag) to predict events in May.

### 5.2 Point-in-Time (PIT) Labels

**Solution:** Only assign `is_fraud=1` if the fraud flag timestamp is **before** the training cutoff.

```python
# Point-in-Time label assignment
if fraud_timestamp is not null AND fraud_timestamp <= train_end:
    is_fraud = 1
else:
    is_fraud = 0  # Even if flagged later
```

**Implementation:** `src/features/temporal_split.py`

### 5.3 Temporal Train/Test Split

**Configuration:**
- Train period: 2024-12-01 to 2025-06-01 (6 months)
- Gap period: 7 days (prevents information leakage)
- Test period: 2025-06-08 to 2025-07-01 (~3 weeks)

**Gap Rationale:** 7-day gap prevents:
1. Overlapping listings from having events in both train and test
2. Near-cutoff fraud signals from leaking across sets

### 5.4 Overlap Handling

**Problem:** Some listings have events in both train and test periods.

**Solution:** Train priority—overlapping listings are kept in train, removed from test.

**Result:** 3,310 overlapping insertions removed from test set.

---

## 6. Model Selection & Design

### 6.1 Baseline Models

We established baselines for comparison:

| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Simple linear baseline; interpretable coefficients |
| **Random Forest** | Ensemble baseline; captures non-linear interactions |
| **Vanilla XGBoost** | Strong tabular baseline; current SOTA for structured data |

### 6.2 Hybrid GNN+XGBoost

**Original Hypothesis:** Fraud rings create network patterns (shared devices, IPs, etc.) that tabular models cannot capture. GNN embeddings can encode these relational patterns.

**Finding:** This hypothesis was **not supported**. See Section 6.5 for why GNN doesn't help.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Build Heterogeneous Graph                                     │
│    - Listing nodes (features: raw SEON signals, listing attrs)   │
│    - Entity nodes (user, device, IP)                            │
│    - Temporal edges (shares_device, shares_ip, etc.)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Train GraphSAGE (Self-supervised)                            │
│    - Link prediction objective                                   │
│    - Output: 16-dim embeddings per listing                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Concatenate: Base Features (71) + GNN Embeddings (16)        │
│    - Total: 87 features                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Train XGBoost Classifier                                     │
│    - Same hyperparameters as vanilla model                      │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 GraphSAGE Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | HeteroConv (SAGEConv per edge type) | Handles multiple edge types naturally |
| Training objective | Self-supervised link prediction | No fraud labels needed for GNN training |
| Edge selection | Listing-to-listing only | Avoids out-of-bound indices from entity edges |
| Layers | 2 | Captures 2-hop neighborhood patterns |
| Hidden dim | 32 | Balance between expressiveness and overfitting |
| Output dim | 16 | Compact embeddings for downstream classifier |

### 6.4 Why Not End-to-End GNN?

**Options considered:**
1. End-to-end GNN with fraud classification head
2. GNN embeddings + separate classifier (chosen)

**Decision:** Option 2 (separate classifier) because:
- XGBoost handles tabular data better than neural networks
- GNN captures relational patterns; XGBoost handles tabular patterns
- Easier to debug and interpret

### 6.5 Why GNN Doesn't Improve Performance

After extensive experimentation, we discovered that GNN embeddings provide **no improvement** over vanilla XGBoost (-1.0% AUC-PR). Investigation revealed:

#### 6.5.1 Inverse Correlation Problem

Fraud rate **decreases** with connectivity for most signals:

| Signal | Solo Fraud% | 5+ Connections Fraud% | Correlation |
|--------|-------------|----------------------|-------------|
| Device Hash | 16.7% | 5.3% | **Inverse** |
| Email | 10.2% | 2.6% | **Inverse** |
| User ID | 9.1% | 2.7% | **Inverse** |
| IP Address | 6.4% | 10.6% | Positive |
| Browser FP | 4.1% | 11.7% | Positive |

**Insight:** Legitimate power users (property managers, agencies) share devices and emails across many listings. Fraudsters use unique identities.

#### 6.5.2 Label Smoothing Effect

Standard GNN message-passing aggregates neighbor features. Our graph has:
- 95%+ edges connecting Normal-Normal listings
- Only 0.2-4% edges connecting Fraud-Fraud listings

The GNN "smooths" fraud node features toward normal patterns, diluting the fraud signal.

#### 6.5.3 Underutilized Features

Current GNN uses only 9 base features, missing rich SEON social verification:
- `email/facebook_registered`, `email/google_registered` (95%+ coverage)
- `phone/whatsapp_registered` (59% coverage)

**Conclusion:** Without restructuring the graph to use only positive-signal edges (IP, Browser FP) or implementing heterophily-aware models (CARE-GNN), vanilla XGBoost remains superior.

---

## 7. Experiment Design

### 7.1 Model Comparison Experiment

**Objective:** Answer RQ1—Does GNN improve fraud detection?

**Setup:**
- Fixed train/test split (2024-12-01 to 2025-06-01 / 2025-06-08 to 2025-07-01)
- Same preprocessing for all models
- 4 model variants compared

**Results:**

| Model | AUC-PR | AUC-ROC | Precision@0.5 | Recall@0.5 |
|-------|--------|---------|---------------|------------|
| Logistic Regression | 0.4912 | 0.9301 | 42.1% | 73.2% |
| Random Forest | 0.5875 | 0.9504 | 41.87% | 81.01% |
| **Vanilla XGBoost** | **0.7457** | **0.9616** | **74.82%** | **67.62%** |
| GNN+XGBoost | 0.7382 | 0.9554 | 74.54% | 68.14% |

**Conclusion:** GNN+XGBoost shows **negative** impact (-1.0% AUC-PR). See Section 6.5 for analysis of why GNN doesn't help with this dataset.

### 7.2 Hyperparameter Optimization

**Objective:** Find optimal hyperparameters for XGBoost and GNN.

**Method:** Optuna TPE sampler with 50 trials

**Search Space:**
```yaml
n_estimators: [50, 500, step=50]
max_depth: [3, 12]
learning_rate: [0.01, 0.3]
min_child_weight: [1, 10]
subsample: [0.6, 1.0]
colsample_bytree: [0.6, 1.0]
scale_pos_weight: [1.0, 20.0]
```

**Optimization Metric:** AUC-PR (maximize)

### 7.3 Expanding Window Experiment

**Objective:** Answer RQ3—How does the model perform over time (concept drift)?

**Setup:**
- Window size: 30 days
- Minimum training windows: 2 (60 days initial training)
- Gap: 7 days

**Results (5 windows):**

| Window | Train End | Test End | AUC-PR |
|--------|-----------|----------|--------|
| 1 | 2025-01-30 | 2025-03-08 | 0.7234 |
| 2 | 2025-03-01 | 2025-04-07 | 0.7156 |
| 3 | 2025-03-31 | 2025-05-07 | 0.6789 |
| 4 | 2025-04-30 | 2025-06-06 | 0.7012 |
| 5 | 2025-05-30 | 2025-07-06 | 0.7123 |

**Aggregate:**
- Mean AUC-PR: 0.7063 ± 0.016
- Drift range: 0.0445 (max - min)

**Conclusion:** Model shows stable performance across windows (low drift range). Monthly retraining is sufficient.

---

## 8. Evaluation Methodology

### 8.1 Primary Metrics

| Metric | Why? |
|--------|------|
| **AUC-PR** | Primary metric for imbalanced datasets; focuses on positive class |
| **AUC-ROC** | Overall discrimination ability |
| **Precision@K** | Relevant for fraud review queues |
| **Recall@K** | Measures fraud capture rate |

### 8.2 Threshold Selection

For production deployment, we need to select an operating point:

| Threshold | Precision | Recall | F1 | Use Case |
|-----------|-----------|--------|-----|----------|
| 0.3 | 65.2% | 92.1% | 76.3% | High recall (catch more fraud) |
| 0.5 | 78.7% | 79.8% | 79.3% | Balanced |
| 0.7 | 89.1% | 62.3% | 73.3% | High precision (fewer false positives) |

### 8.3 SEON Benchmark

**Objective:** Compare our model to the existing SEON-based system.

**Method:** Use SEON's `state` field (APPROVED/DECLINED) as baseline predictions.

**Result:** Our model provides probabilities that can be tuned to different precision/recall tradeoffs, while SEON provides binary decisions.

---

## 9. Concept Drift Mitigation

### 9.1 Problem Definition

Fraud patterns change over time:
- Fraudsters adapt to detection methods
- New attack vectors emerge
- Seasonal variations in fraud

### 9.2 Mitigation Strategy

**Expanding Window Training:**
- Retrain monthly with expanding historical data
- Monitor per-window metrics for drift detection
- Alert if performance drops below threshold

**Implementation:** `make expanding-vanilla` or `make expanding-gnn`

### 9.3 Observed Drift

| Metric | Value | Interpretation |
|--------|-------|----------------|
| AUC-PR std | 0.016 | Low variance = stable |
| Drift range | 0.0445 | Small = minimal drift |

**Conclusion:** Current fraud patterns are relatively stable. Monthly retraining is sufficient.

---

## 10. Explainability Analysis

### 10.1 SHAP Analysis

**Objective:** Answer RQ2 and RQ4—What features matter and why?

**Method:** SHAP (SHapley Additive exPlanations) for XGBoost

### 10.2 Global Feature Importance

Top 10 features by mean |SHAP| value:

**Top Features by Mean |SHAP| Value:**

| Rank | Feature | Category | Importance |
|------|---------|----------|------------|
| 1 | `ip_country` | SEON Categorical | High |
| 2 | `LISTING_OFFERTYPE` | Listing | High |
| 3 | `session/device_type` | SEON Session | Medium-High |
| 4 | `LISTING_CATEGORIES` | Listing | Medium |
| 5 | `event_hour` | Temporal | Medium |
| 6 | `data_center_proxy` | SEON Boolean | Medium |
| 7 | `tor` | SEON Boolean | Medium |
| 8 | `email/domain/disposable` | SEON Boolean | Medium |
| 9 | `email/domain/free` | SEON Boolean | Medium |
| 10 | `gnn_emb_0` | GNN Embedding | Low-Medium |

### 10.3 GNN Contribution

GNN embeddings were found to be **counterproductive** in this dataset:

| Metric | Vanilla XGBoost | GNN+XGBoost | Impact |
|--------|-----------------|-------------|--------|
| AUC-PR | 0.7457 | 0.7382 | **-1.0%** |
| AUC-ROC | 0.9616 | 0.9554 | -0.6% |

**Reason:** The graph structure has inverse correlation with fraud (see Section 6.5). Legitimate power users share more devices/emails than fraudsters, causing the GNN to smooth fraud signals toward normal.

### 10.4 Local Explanations

**Use Case:** Fraud analysts need to understand why a specific listing was flagged.

**Example SHAP Explanation:**

```
Prediction: 0.87 (HIGH FRAUD RISK)

Top contributors (example high-risk listing):
  +0.28  ip_country = "BJ" (Benin, high-risk region)
  +0.18  LISTING_CATEGORIES = "STUDIO" (high-risk category)
  +0.12  data_center_proxy = True (datacenter IP detected)
  +0.08  email/domain/disposable = True (disposable email)
  +0.05  event_hour = 3 (unusual posting time)
```

---

## Summary

This methodology document describes the data-driven decisions made throughout the research:

1. **Data:** Merged events + SEON using as-of join; filtered to 2024-12+
2. **Labels:** Point-in-Time labels prevent leakage; fraud rate is 8.11% per listing
3. **Features:** 66 base features (raw SEON signals, not SEON ML scores)
4. **Graph:** Heterogeneous graph tested but found counterproductive
5. **Model:** Vanilla XGBoost is the best performer (0.7457 AUC-PR)
6. **GNN Finding:** Graph embeddings don't help—legitimate users are more connected than fraudsters
7. **Evaluation:** AUC-PR primary metric; expanding window validates stability
8. **Explainability:** SHAP provides global and local explanations

All decisions are backed by empirical analysis documented in `docs/data_overview.md` and `docs/experiment_journal.md`.

---

*Document Version: 1.0 | Last Updated: December 17, 2025*
