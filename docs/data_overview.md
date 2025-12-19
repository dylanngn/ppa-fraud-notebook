# Data Overview: Fraud Detection in Online Real Estate Marketplaces

**Dataset:** Swiss Marketplace Group (SMG) Real Estate Data  
**Analysis Date:** December 19, 2025  
**Author:** Data Science Team

---

## Executive Summary

This document provides a comprehensive analysis of the fraud detection dataset, covering data quality, temporal patterns, entity relationships, and key fraud indicators. The dataset contains **804,717 events** across **86,160 unique listings** spanning nearly 2 years of real estate marketplace activity.

### Key Findings

| Metric | Value | Notes |
|--------|-------|-------|
| Total Events | 804,717 | Multiple events per listing lifecycle |
| Unique Listings | 86,160 | Unit of fraud analysis |
| **Fraud Rate** | **8.11%** | Per listing (not per event) |
| Imbalance Ratio | 1:11 | Moderate imbalance |
| Features | 509 columns | Rich feature space |
| Date Range | Jan 2024 - Dec 2025 | ~24 months |

**Note:** Fraud rate is calculated per unique listing (8.11%), not per event (1.21%). Legitimate listings have more events (9.7 avg) than fraud listings (4.8 avg).

---

## 1. Dataset Overview

### 1.1 Data Shape & Size

```
Rows:    804,717 events
Columns: 509 features
Memory:  8.3 GB
Unique Listings: 86,160
```

### 1.2 Column Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| String | 261 | Categorical variables, hashes, IDs |
| Boolean | 160 | Binary flags (SEON signals, etc.) |
| Int64 | 70 | Counts, scores, numeric features |
| Float64 | 17 | Prices, coordinates, ratios |
| Datetime | 1 | Event timestamp |
| Int8 | 1 | Fraud label |

### 1.3 Data Sources

The merged dataset combines two primary sources:

1. **Insertion Events** (`insertion_events.csv`)
   - Listing lifecycle events (DRAFT, SUBMITTED, PUBLISHED, etc.)
   - User and listing attributes
   - Fraud flag timestamp (`FLAGGEDFORFRAUD`)

2. **SEON Transactions** (`seon_transactions.csv`)
   - Third-party fraud detection API responses
   - Risk scores (fraud, email, phone, blackbox)
   - Device fingerprinting and session data
   - IP and geo-location analysis

### 1.4 Fraud Label Definition

The `is_fraud` label is derived from the `FLAGGEDFORFRAUD` column:
- **Source:** Customer support manual flagging OR automated SEON rules
- **Timestamp:** When the listing was flagged (not when created)
- **Propagation:** All events for a flagged listing get `is_fraud=1`

---

## 2. Fraud Timing Analysis

### 2.1 Time from Creation to Fraud Flag

| Metric | Value |
|--------|-------|
| Minimum | 1 minute 24 seconds |
| Maximum | 556 days |
| **Median** | **1 hour 40 minutes** |
| Mean | 3 days 5 hours |

### 2.2 Detection Speed Distribution

| Time Bucket | Count | % |
|-------------|-------|---|
| Same day (0-24h) | 5,242 | **75.0%** |
| 1-7 days | 1,055 | 15.1% |
| 1-4 weeks | 492 | 7.0% |
| 1-3 months | 178 | 2.5% |
| 3+ months | 19 | 0.3% |

**Insight:** 75% of fraud is detected within 24 hours, with a median of ~1.5 hours. This suggests effective real-time detection mechanisms are in place.

---

## 3. Temporal Analysis

### 3.1 Monthly Listing Creation & Fraud Rate

| Month | Listings | Fraud | Rate | Notes |
|-------|----------|-------|------|-------|
| 2024-01 to 2024-10 | 705 | 5 | 0.7% | Sparse pre-integration data |
| 2024-11 | 3,583 | 311 | 8.68% | SEON integration begins |
| 2024-12 | 5,530 | 523 | 9.46% | Ramp-up |
| 2025-01 | 8,471 | 969 | 11.44% | - |
| **2025-02** | 7,655 | 1,004 | **13.12%** | Peak fraud rate |
| 2025-03 | 8,439 | 909 | 10.77% | - |
| 2025-04 | 7,259 | 776 | 10.69% | - |
| 2025-05 | 7,332 | 587 | 8.01% | - |
| 2025-06 | 6,583 | 300 | **4.56%** | Significant drop |
| 2025-07 | 6,658 | 436 | 6.55% | - |
| 2025-08 | 7,205 | 383 | 5.32% | - |
| 2025-09 | 7,113 | 336 | 4.72% | - |
| 2025-10 | 6,640 | 297 | 4.47% | - |
| 2025-11 | 2,987 | 149 | 4.99% | - |

**Observation:** Fraud rate peaked at 13.12% in Feb 2025, then dropped to ~5% by mid-2025. This could indicate improved detection or fraudster adaptation.

### 3.2 Hourly Patterns (by Event)

```
Hour   Fraud Rate
06:00  2.56%  ████████████  ← Peak fraud activity
07:00  1.96%  █████████
08:00  1.06%  █████
...
20:00  0.33%  █
21:00  0.08%                ← Lowest fraud activity
```

**Insight:** Fraudsters are most active during early morning hours (6-7 AM), suggesting automated or scripted attacks during low-activity periods.

---

## 4. Entity Analysis

### 4.1 Unique Entity Counts

| Entity | Unique Count | Null % |
|--------|-------------|--------|
| Listings | 86,160 | 0.0% |
| Users | 63,211 | 0.0% |
| IPs | 51,915 | 0.1% |
| Devices | 17,574 | 0.0% |
| Emails | 58,578 | 7.6% |
| Phones | 44,488 | 33.2% |

### 4.2 Listing Lifecycle

| Metric | All Listings | Fraud Listings | Legit Listings |
|--------|-------------|----------------|----------------|
| Count | 86,160 | 6,986 | 79,174 |
| Avg Events | 9.3 | **4.8** | 9.7 |
| Median Events | 7 | **3** | 8 |

**Insight:** Fraudulent listings have significantly shorter lifecycles (fewer events), suggesting early detection or abandonment.

---

## 5. Offer Type & Category Analysis

### 5.1 Offer Type (per Listing)

| Offer Type | Listings | Fraud | Rate | Comparison |
|------------|----------|-------|------|------------|
| **RENT** | 72,530 | 6,866 | **9.47%** | Baseline |
| BUY | 13,630 | 120 | 0.88% | 11x lower |

**Critical:** Rental listings have **11x higher fraud rate** than purchase listings.

### 5.2 Listing Categories (per Listing, >100 listings)

| Category | Listings | Fraud Rate | Risk Level |
|----------|----------|------------|------------|
| **APARTMENT, STUDIO** | 2,262 | **34.97%** | Extreme |
| **APARTMENT, SINGLE_ROOM** | 968 | **16.43%** | Very High |
| APARTMENT, FLAT | 46,924 | 11.87% | High |
| FURNISHED_FLAT | 2,231 | 4.93% | Moderate |
| HOUSE, VILLA | 679 | 2.06% | Low |
| HOUSE, SINGLE_HOUSE | 5,068 | 0.59% | Very Low |
| OFFICE | 2,875 | 0.24% | Minimal |

**Key Finding:** Studios (35%) and single rooms (16%) have extremely high fraud rates. These are typical targets for rental scams.

---

## 6. Geographic Analysis (per Listing)

### 6.1 IP Country Distribution

| Country | Listings | Fraud | Rate | Risk |
|---------|----------|-------|------|------|
| CH (Switzerland) | 78,991 | 5,258 | 6.66% | Baseline |
| DE (Germany) | 2,149 | 523 | **24.34%** | 4x |
| FR (France) | 1,601 | 509 | **31.79%** | 5x |
| IT (Italy) | 642 | 148 | **23.05%** | 3x |
| NL (Netherlands) | 174 | 42 | **24.14%** | 4x |
| AT (Austria) | 233 | 59 | **25.32%** | 4x |
| **BJ (Benin)** | 251 | 168 | **66.93%** | 10x |
| **RO (Romania)** | 123 | 68 | **55.28%** | 8x |

### 6.2 High-Risk Countries (>20% fraud rate)

| Country | Listings | Fraud Rate |
|---------|----------|------------|
| **NP (Nepal)** | 21 | **90.5%** |
| **TG (Togo)** | 16 | **68.8%** |
| **BJ (Benin)** | 251 | **66.9%** |
| BG (Bulgaria) | 18 | 66.7% |
| CI (Ivory Coast) | 10 | 60.0% |
| RO (Romania) | 123 | 55.3% |
| CA (Canada) | 27 | 37.0% |
| FR (France) | 1,601 | 31.8% |

**Critical:** West African countries (Benin, Togo, Ivory Coast) show 60-90% fraud rates - classic real estate scam origins.

---

## 7. Device Analysis (per Listing)

| Device Type | Listings | Fraud | Rate |
|-------------|----------|-------|------|
| Desktop | 70,587 | 6,030 | 8.54% |
| Phone | 14,298 | 920 | 6.43% |
| Tablet | 1,263 | 30 | 2.38% |
| **TV** | 7 | 6 | **85.71%** |

**Anomaly:** TV device type shows 86% fraud rate - these are spoofed device fingerprints.

---

## 8. Platform Analysis (per Listing)

| Platform | Listings | Fraud Rate |
|----------|----------|------------|
| Homegate only | 53,018 | 6.75% |
| ImmoScout24 only | 32,072 | 9.75% |
| **Both platforms** | 1,062 | **26.74%** |

**Insight:** Cross-platform listings have 4x higher fraud rate - fraudsters maximize exposure.

---

## 9. SEON Data

### 9.1 Feature Categories

SEON provides two types of data:

| Type | Examples | Used in Model |
|------|----------|---------------|
| **ML Prediction Scores** | `fraud_score`, `blackbox_score` | No (SEON's own predictions) |
| **Raw Signals** | `tor`, `vpn`, `data_center_proxy`, `email/domain/disposable` | Yes |

### 9.2 Raw Signal Distributions

| Signal | Fraud Rate When True | Legit Rate When True | Lift |
|--------|---------------------|----------------------|------|
| `tor=True` | 28.4% | 0.3% | 94.7x |
| `data_center_proxy=True` | 15.2% | 2.1% | 7.2x |
| `email/domain/disposable=True` | 22.1% | 1.8% | 12.3x |
| `vpn=True` | 8.5% | 3.2% | 2.7x |

### 9.2 Boolean Risk Signals

| Signal | Prevalence | Fraud Rate | Lift |
|--------|------------|------------|------|
| phone_is_disposable | 0.01% | **24.2%** | 3x |
| email/domain/disposable | 0.29% | **19.9%** | 2.5x |
| public_proxy | 0.39% | 5.1% | 0.6x |
| residential_proxy | 0.40% | 3.8% | 0.5x |

---

## 10. Network Graph Properties (per Listing)

### 10.1 Device Sharing Analysis

| Metric | Value |
|--------|-------|
| Total unique devices | 16,420 |
| Devices with 2+ listings | 6,473 (39.4%) |
| Multi-listing devices with fraud | 1,322 (**20.4%**) |

### 10.2 Super-Connectors (10+ Listings per Device)

| Metric | Value |
|--------|-------|
| Super-connector devices | 1,248 |
| With fraud | 443 (**35.5%**) |
| Top 5 listing counts | 6251, 3177, 1907, 811, 658 |

**Critical Finding:** 35% of super-connected devices are linked to fraud. One device is associated with **6,251 listings** - clear fraud ring indicator.

### 10.3 Email Sharing

| Metric | Value |
|--------|-------|
| Emails with 2+ listings | 4,642 |
| With fraud | 290 (6.2%) |

---

## 11. Data Quality Notes

### 11.1 Missing Data

| Column | Null % | Impact |
|--------|--------|--------|
| Phone hash | 33.2% | Reduces phone-based linking |
| Email hash | 7.6% | Minor impact |
| phone_score | 27.6% | SEON phone analysis incomplete |

### 11.2 Data Sparsity

- **Pre-November 2024:** Very sparse data (~705 listings total)
- **Recommendation:** Use `train_start_date = 2024-12-01` to exclude incomplete data

### 11.3 Temporal Label Considerations

- Fraud labels (`FLAGGEDFORFRAUD`) appear **after** listing creation
- 75% flagged within 24 hours
- **Solution:** Point-in-Time (PIT) label assignment for training

---

## 12. Summary Statistics

### 12.1 Corrected Fraud Rates

| Calculation Method | Fraud | Total | Rate |
|-------------------|-------|-------|------|
| ❌ Event-based (incorrect) | 9,705 | 804,717 | 1.21% |
| ✅ **Listing-based (correct)** | 6,986 | 86,160 | **8.11%** |

### 12.2 Top Fraud Risk Indicators

| Indicator | Fraud Rate | Baseline | Lift |
|-----------|------------|----------|------|
| Device type = TV | 85.7% | 8.1% | 10.6x |
| Country = Nepal | 90.5% | 8.1% | 11.2x |
| Country = Benin | 66.9% | 8.1% | 8.3x |
| Category = Studio | 35.0% | 8.1% | 4.3x |
| Both platforms | 26.7% | 8.1% | 3.3x |
| Country = France | 31.8% | 8.1% | 3.9x |
| Category = Single Room | 16.4% | 8.1% | 2.0x |

---

## 13. Recommendations for Modeling

### 13.1 High-Value Features

1. **Raw SEON Signals:** `tor`, `data_center_proxy`, `email/domain/disposable`
2. **Geographic:** `ip_country` (especially BJ, RO, TG, NP)
3. **Category:** Studios and single rooms = high risk
4. **Temporal:** Unusual posting hours (late night/early morning)
4. **Platform:** Cross-platform listings = high risk
5. **Offer Type:** RENT vs BUY (11x difference)
6. **Device Networks:** Super-connector detection

### 13.2 Graph Features

- Device sharing networks (6,000+ listings per device)
- Cross-platform detection
- User clustering by shared attributes

### 13.3 Class Imbalance Handling

- Imbalance ratio: 1:11 (manageable)
- Recommend: `scale_pos_weight=11` for XGBoost
- Use AUC-PR as primary metric

### 13.4 Temporal Considerations

- Fraud rate varies by month (4% - 13%)
- 75% of fraud detected same-day
- Use Point-in-Time labels for training

---

## Appendix: ETL Pipeline

### Data Flow

```
insertion_events.csv ─┬─► Extract (Polars LazyFrame)
                      │
seon_transactions.csv ┘
                      │
                      ▼
              As-of Join (closest SEON after event)
                      │
                      ▼
              Derive is_fraud from FLAGGEDFORFRAUD timestamp
                      │
                      ▼
              Anonymize (SHA-256 hashing)
                      │
                      ▼
              merged_events.parquet
```

### Key Files

| File | Purpose |
|------|---------|
| `src/data/extract.py` | Load CSV files with streaming |
| `src/data/transform.py` | As-of join, label derivation |
| `src/data/load.py` | Save to Parquet |
| `src/data/etl.py` | Orchestration |

---

*Analysis completed: December 17, 2025*
