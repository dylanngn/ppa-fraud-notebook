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

## 5. Feature Groups & Configuration

Features are organized into explicit groups for ablation experiments. See `src/models/config/constants.py` for definitions.

### Group: `core_numerical` (Always Include)

| Feature | Coverage | Source | Description |
|---------|----------|--------|-------------|
| `account_age_days` | 100% | Derived | Days since account creation |
| `payment_type` | 98% | `bundle.paymentType` | DIRECT vs INVOICE |
| `bundle_tier` | 97.7% | `bundle.tier` | basic, premium, top |
| `bundle_period` | 77.25% | `bundle.period` | Duration in days |
| `log_price` | 80% | Derived | Log-transformed price |
| `latitude` | 99.4% | `listing.address.geoCoordinates.latitude` | Property latitude |
| `longitude` | 99.4% | `listing.address.geoCoordinates.longitude` | Property longitude |
| `offer_type` | 100% | `listing.offerType` | BUY vs RENT |
| `living_space` | 84.7% | `listing.characteristics.livingSpace` | Square meters |
| `rooms` | 92.7% | `listing.characteristics.numberOfRooms` | Room count |

### Group: `boolean_high` (>40% TRUE rate)

7 features: `has_balcony`, `has_parking`, `has_nice_view`, `has_garage`, `is_child_friendly`, `is_quiet`, `has_elevator`

### Group: `boolean_moderate` (20-40% TRUE rate)

3 features: `has_washing_machine`, `are_pets_allowed`, `is_wheelchair_accessible`

### Group: `boolean_low` (<20% TRUE rate)

2 features: `is_old`, `is_new_building`

### Group: `graph_basic`

12 features including: `contact_email_count`, `shared_contact_email_count`, `listing_component_size`, `listing_pagerank`, etc.

### Group: `graph_advanced`

5 features: `degree_total`, `is_isolated`, `unique_identifier_count`, `neighbor_overlap_score`, `avg_neighbor_degree`

### Group: `time_weighted_core`

11 features including: `email_time_spread`, `email_recency_weighted`, `phone_time_spread`, `combined_recency_weighted`, etc.

### Group: `text`

10 features: `description_length`, `description_word_count`, `description_has_url`, etc.

---

## 6. Graph Structure

### Entity Identification (Source of Truth)

| Entity | Raw Column | Graph Alias | Join Logic |
|--------|------------|-------------|------------|
| **Listing** | `i.object_reference` | `insertion_id` | Primary key |
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
include_groups:
  - core_numerical
  - boolean_all
  - graph_all
```

### Available Profiles

| Profile | Groups Included | Use Case |
|---------|-----------------|----------|
| `quick` | `core_numerical`, `boolean_high` | Fast iteration |
| `standard` | `core_numerical`, `boolean_all`, `graph_all` | Balanced |
| `production` | All core + graph + time_weighted + text | Best performance |

---

## 10. Key Decisions & Rationale

| Decision | Rationale | Date |
|----------|-----------|------|
| Use billing phone over contact phone | 98% vs 70% coverage | 2025-11-30 |
| Unify phone edges (coalesce) | Reduce complexity, ~99% coverage | 2025-11-30 |
| Include `has_elevator` in model | 100% semantic coverage, was incorrectly excluded | 2025-12-01 |
| Switch to `include_groups` config | Explicit selection prevents silent failures | 2025-12-01 |
| Add low TRUE rate booleans | Potentially discriminative for fraud detection | 2025-12-01 |
| Remove interaction features | XGBoost learns these automatically (Exp 10); eliminates dependency ordering issue | 2025-12-01 |

### ⚠️ Technical Debt: Feature Category Ordering

**Status**: Deferred (acceptable for experiment phase)

**Context**: The `FeatureRegistry` executes feature categories in the order listed in YAML config. If a category depends on columns from another category, incorrect ordering causes silent failures (returns df unchanged with warning).

**Current State**: Safe. Interaction features (the only category with dependencies) were removed. Remaining categories (`base`, `graph`, `advanced_graph`, `time_weighted`, `text`) are independent.

**Future Risk**: If adding new feature categories with dependencies:
1. Document dependencies in the category's docstring
2. Consider adding explicit dependency declarations to `FeatureRegistry`
3. Or: auto-detect and compute dependencies before dependent categories

**Reference**: Code review feedback Issue #11 (2025-12-01)

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
