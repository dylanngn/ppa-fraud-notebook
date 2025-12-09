# Knowledge Base: Fraud Detection Data & Features

**Purpose**: Single source of truth for data quality, feature definitions, and engineering decisions.

> 📁 For project structure and architecture, see [`architecture.md`](architecture.md)  
> 📓 For experiment tracking and results, see [`experiment_journal.md`](experiment_journal.md)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| **Source** | `artifacts/raw_insertions.parquet` |
| **Total Rows** | 234,458 |
| **Total Columns** | 292 |
| **Usable Fields** | 127 (≥50% semantic coverage) |
| **Boolean Indicators** | 50 (100% semantic coverage via NULL=FALSE) |
| **Truly Unusable** | 146 (≥95% null, excluding boolean indicators) |
| **Account Date Range** | 2020-12-17 → 2025-11-25 |
| **Listing Date Range** | 2023-01-01 → 2025-11-11 |
| **Fraud Rate** | 8.2% (19,217 fraud flags) |

### Data Integrity Checks

| Check | Result |
|-------|--------|
| Duplicate `object_reference` | ✅ 100% unique |
| Empty `object_reference` | ✅ None |
| Flattened listing coverage | ✅ 98.21% |

### Data Type Distribution

| Type | Count | Notes |
|------|-------|-------|
| String | 171 | Hashed identifiers |
| Boolean | 54 | NULL = FALSE semantics |
| Int64 | 29 | IDs, counts |
| Float64 | 20 | Prices, coordinates |
| Datetime | 5 | Timestamps |
| Null | 12 | Completely empty |

---

## 2. Critical Data Quality Insight: NULL = FALSE

> ⚠️ **Boolean indicator fields use NULL to mean FALSE, not "missing data".**

This insight significantly impacts feature engineering:

| Pattern | Examples | NULL Semantics |
|---------|----------|----------------|
| `has*` | `hasBalcony`, `hasElevator` | Property does NOT have this feature |
| `is*` | `isQuiet`, `isChildFriendly` | Property is NOT this |
| `are*` | `arePetsAllowed` | Feature is NOT allowed |

**Implication**: Fields like `hasElevator` (40% non-null) actually have **100% semantic coverage**:
- 40% have elevators (TRUE)
- 60% don't have elevators (NULL = FALSE)

---

## 3. Contact Field Coverage

| Field Type | Primary Field | Coverage | Unique Values | Notes |
|------------|---------------|----------|---------------|-------|
| **Email (lister)** | `listing.lister.email.hash` | 99.98% | 130,947 | Primary email |
| **Email (billing)** | `listing.lister.billing.email.hash` | 98.00% | 129,918 | Secondary |
| **Phone (billing)** | `listing.lister.billing.phoneDay.hash` | 97.99% | 127,770 | **Use this** |
| **Phone (contact)** | `listing.lister.phone.hash` | 69.98% | 89,614 | Lower coverage |
| **Address** | `listing.address.address_hash` | 100.0% | ~234k | Unique per listing |
| **Coordinates** | `listing.address.geoCoordinates.lat/lng` | 99.40% | 107,168 | High precision |

> 💡 **Decision**: Use billing phone (98%) instead of contact phone (70%) for graph edges.

---

## 4. Boolean Indicator Fields (Complete List)

All 50 boolean indicator fields with their TRUE rates (sorted by frequency):

### High TRUE Rate (>40%) - Common Features

| Feature | TRUE % | Source Field | In Model |
|---------|--------|--------------|----------|
| `isFromNewInsertionFunnel` | 99.96% | `listing.meta.isFromNewInsertionFunnel` | No (meta) |
| `seonApproved` | 98.83% | `auto_approval_criteria.criteria.seonApproved` | No (label leak) |
| `hasBalcony` | 71.19% | `listing.characteristics.hasBalcony` | ✅ Yes |
| `hasParking` | 55.14% | `listing.characteristics.hasParking` | ✅ Yes |
| `hasNiceView` | 47.72% | `listing.characteristics.hasNiceView` | ✅ Yes |
| `hasGarage` | 44.19% | `listing.characteristics.hasGarage` | ✅ Yes |
| `isChildFriendly` | 43.52% | `listing.characteristics.isChildFriendly` | ✅ Yes |
| `isQuiet` | 42.74% | `listing.characteristics.isQuiet` | ✅ Yes |
| `hasElevator` | 40.88% | `listing.characteristics.hasElevator` | ✅ Yes |

### Moderate TRUE Rate (20-40%)

| Feature | TRUE % | Source Field | In Model |
|---------|--------|--------------|----------|
| `hasWashingMachine` | 32.40% | `listing.characteristics.hasWashingMachine` | ✅ Yes |
| `arePetsAllowed` | 28.43% | `listing.characteristics.arePetsAllowed` | ✅ Yes |
| `isAPMEnabled` | 26.18% | `listing.lister.billing.payment.isAPMEnabled` | No (billing) |
| `isWheelchairAccessible` | 25.69% | `listing.characteristics.isWheelchairAccessible` | ✅ Yes |

### Low TRUE Rate (5-20%) - Potentially Discriminative

| Feature | TRUE % | Source Field | In Model |
|---------|--------|--------------|----------|
| `isOldBuilding` | 16.67% | `listing.characteristics.isOldBuilding` | ✅ Yes |
| `isNewBuilding` | 16.25% | `listing.characteristics.isNewBuilding` | ✅ Yes |
| `hasCableTv` | 14.76% | `listing.characteristics.hasCableTv` | ⚠️ Experiment |
| `hasFireplace` | 10.71% | `listing.characteristics.hasFireplace` | ⚠️ Experiment |
| `isMinergieGeneral` | 9.20% | `listing.characteristics.isMinergieGeneral` | ⚠️ Experiment |
| `isMinergieCertified` | 6.81% | `listing.characteristics.isMinergieCertified` | ⚠️ Experiment |

### Very Low TRUE Rate (<5%) - Rare but Valid

| Feature | TRUE % | Source Field | Notes |
|---------|--------|--------------|-------|
| `isSmokingAllowed` | 4.88% | `listing.characteristics.isSmokingAllowed` | Rare amenity |
| `hasSwimmingPool` | 4.75% | `listing.characteristics.hasSwimmingPool` | Luxury feature |
| `hasDishwasher` | 3.86% | `listing.characteristics.hasDishwasher` | Appliance |
| `hasStoreRoom` | 3.82% | `listing.characteristics.hasStoreRoom` | Storage |

> 💡 **Hypothesis**: Low TRUE rate features may be discriminative for fraud if fraudsters over-claim or under-claim certain amenities.

---

## 5. Feature Configuration (Simplified)

Features are now **auto-selected** from ETL output. See `src/models/config/constants.py` for exclusions.

### Category: `base` (Auto-Selected)

All columns from `raw_insertions.parquet` are included **except**:
- ID columns: `object_reference`, `owner_id`, `insertion_id`, `user_id`
- Target/labels: `is_fraud`, `fraud_flag`, `seon_approved`
- Timestamps: `submission_at`, `first_published_date`, `account_created_at`
- Hash columns: `*_hash`, `*_hashes`
- Text: `description_text`

This auto-selection includes:
- All numerical features (prices, coordinates, living space, rooms, etc.)
- All boolean indicators (NULL = FALSE semantics)
- Categorical features (offer_type, bundle_tier, payment_type, etc.)

### Category: `graph` (Explicit Computation)

17 graph-derived features computed from edge relationships:

| Feature | Description |
|---------|-------------|
| `contact_email_count` | Listings sharing contact email |
| `shared_contact_email_count` | Other users with same email |
| `listing_component_size` | Size of connected component |
| `listing_pagerank` | PageRank centrality |
| `listing_degree` | Total edge connections |
| `degree_total` | Sum of all edge types |
| `is_isolated` | No graph connections |
| `unique_identifier_count` | Distinct identifiers used |
| `neighbor_overlap_score` | Similarity to neighbors |
| ... | (see `constants.py` for full list) |

> **Note**: Experiment 9 validated that auto-selection (all columns) matches manually-curated feature sets in AUC-PR performance, justifying this simplification.

---

## 6. Graph Structure

### Entity Identification (Source of Truth)

**Naming Convention:**
- **External name**: `listing_id` (used in logs, configs, documentation)
- **Internal storage**: `insertion_id` (database field, parquet column)
- **Graph indexing**: 0..N-1 based on FULL dataframe order (critical for GNN embedding lookup)

| Entity | Raw Column | Internal Alias | Join Logic |
|--------|------------|----------------|------------|
| **Listing** | `i.object_reference` | `insertion_id` | Primary key (externally called `listing_id`) |
| **User** | `u.owner_id` | `user_id` | Joined via `i.listing->'legacy'->>'personId' = u.owner_id` |

### Node Types (6)

| Node Type | ID Field | Count | Features |
|-----------|----------|-------|----------|
| **User** | `owner_id` | 133,811 | `account_created_at` |
| **Listing** | `insertion_id` | 234,458 | All listing features + embeddings |
| **Email** | Email hash | 130,947 | (constant) |
| **Phone** | Phone hash | 127,770 | (constant) |
| **Address** | Composite hash | ~234k | `latitude`, `longitude` |
| **IP** | IP hash | — | (constant) |

### Edge Types (8)

| Edge | Source → Target | Coverage | Notes |
|------|-----------------|----------|-------|
| `posts` | User → Listing | 100% | Primary relationship |
| `uses` | User → IP | 94.7% | IP tracking |
| `has_email` | User → Email | 99.98% | User email |
| `has_contact_email` | Listing → Email | 99.98% | Lister email |
| `has_billing_email` | Listing → Email | 98.00% | Billing email |
| `has_phone` | Listing → Phone | ~99% | **Unified** billing+lister (coalesced) |
| `located_at` | Listing → Address | 99.99% | Property location |
| `billing_address` | Listing → Address | 98% | Billing address |

> 💡 **Simplification**: Phone edges unified from 2 → 1 by coalescing billing (98%) and lister (70%) phone.

---

## 7. Training Strategy

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
  max_windows: null         # All windows
```

---

## 8. Fraud Detection Insights

### High-Signal Patterns

| Pattern | Fraud Rate | Lift | Notes |
|---------|-----------|------|-------|
| New account + Invoice payment | 28% | 17.5x | Strongest signal |
| Isolated listings (no graph connections) | Higher | TBD | Fraudsters avoid networks |
| Email reuse across listings | Higher | TBD | Fraud rings |

### Anti-Patterns (What Doesn't Work)

| Feature/Approach | Finding | Reason |
|------------------|---------|--------|
| Burst detection | No impact | Fraudsters vary timing |
| Interaction features | No lift | XGBoost learns these automatically |
| Sliding windows (for GNN) | Performance drops | GNNs need complete history |
| Low-coverage exclusion | **Wrong** | Boolean indicators have 100% semantic coverage |

---

## 9. Feature Configuration Design

```yaml
# conf/features/auto.yaml (default)
categories:
  - base   # Auto-selects all ETL columns (minus exclusions)
  - graph  # Computes graph-derived features
```

### Configuration Approach

| Aspect | Design | Rationale |
|--------|--------|-----------|
| Tabular features | Auto-select all | Experiment 9 showed no benefit to manual curation |
| Graph features | Explicit compute | Requires edge data, computed on-demand |
| Exclusions | Defined in constants.py | IDs, labels, hashes, timestamps |

> This simplified approach eliminates the need for multiple feature profiles while maintaining the same AUC-PR performance.

---

## 10. Key Decisions & Rationale

| Decision | Rationale | Date |
|----------|-----------|------|
| Use billing phone over contact phone | 98% vs 70% coverage | 2025-11-30 |
| Unify phone edges (coalesce) | Reduce complexity, ~99% coverage | 2025-11-30 |
| Include `has_elevator` in model | 100% semantic coverage, was incorrectly excluded | 2025-12-01 |
| Switch to `include_groups` config | Explicit selection prevents silent failures | 2025-12-01 |
| Add low TRUE rate booleans | Potentially discriminative for fraud detection | 2025-12-01 |
| Remove interaction features | XGBoost learns these automatically; eliminates dependency ordering issue | 2025-12-01 |
| Simplify to auto feature selection | Experiment 9 validated that auto-selection matches manual curation | 2025-12-09 |
| Remove time_weighted & text features | Marginal impact (<0.5% AUC-PR); reduces complexity | 2025-12-09 |
| Merge graph & advanced_graph | Single `graph` category for all graph-derived features | 2025-12-09 |

### ⚠️ Technical Debt: Feature Category Ordering

**Status**: Resolved

**Context**: The `FeatureRegistry` executes feature categories in the order listed in YAML config.

**Current State**: Safe. Only two categories remain (`base`, `graph`), which are independent. The `base` category auto-selects all ETL columns, and `graph` computes derived features from edges.

---

## Appendix: Field Coverage Reference

### Core Identifiers (100% coverage)

- `object_reference` - Listing business ID → aliased to `insertion_id` in graph artifacts
- `owner_id` - User ID → aliased to `user_id` in graph artifacts
- `submission_at` - Submission timestamp
- `account_created_at` - Account creation timestamp
- `listing_platform` / `user_platform` - Platform brand

> ⚠️ **User-Listing Relationship**: The true connection between listings and users is:
> ```sql
> i.listing->'legacy'->>'personId' = u.owner_id
> ```
> This joins on the `personId` inside the listing JSON to the user's `owner_id`, NOT on `insertions.user_id`.

### High Coverage Fields (>90%)

- `listing.offerType` - 100%
- `listing.lister.email.hash` - 99.98%
- `listing.address.geoCoordinates` - 99.40%
- `bundle.tier` - 97.73%
- `listing.characteristics.numberOfRooms` - 92.71%

### Moderate Coverage Fields (50-90%)

- `listing.characteristics.livingSpace` - 84.69%
- `bundle.period` - 77.26%
- `listing.lister.phone.hash` - 69.98%
- `listing.characteristics.yearBuilt` - 53.05%

### Excluded Fields (Truly Unusable)

146 fields with ≥95% null that are NOT boolean indicators. Examples:
- `listing.valueAddedServices` - 100% null
- `listing.lister.website` - 100% null
- `listing.characteristics.cubage` - 95.08% null (numeric, truly missing)
