# Knowledge Base: Domain Knowledge

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| **Source** | `artifacts/raw_insertions.parquet` |
| **Total Rows** | 234,458 listings |
| **Total Columns** | 292 raw ETL fields |
| **Usable Fields** | 127 (≥50% coverage) |
| **Boolean Indicators** | 50 (NULL = FALSE semantics) |
| **Truly Unusable** | 146 (≥95% null) |
| **Account Date Range** | 2020-12-17 → 2025-11-25 |
| **Listing Date Range** | 2023-01-01 → 2025-11-11 |
| **Fraud Rate** | 8.2% (19,217 fraud cases) |

### Data Integrity Checks

| Check | Result |
|-------|--------|
| Duplicate `object_reference` | ✅ 100% unique |
| Empty `object_reference` | ✅ None |
| Flattened listing coverage | ✅ 98.21% |

### Data Type Distribution

| Type | Count | Notes |
|------|-------|-------|
| String | 171 | Hashed identifiers, text |
| Boolean | 54 | NULL = FALSE semantics |
| Int64 | 29 | IDs, counts |
| Float64 | 20 | Prices, coordinates |
| Datetime | 5 | Timestamps |
| Null | 12 | Completely empty columns |

---

## 2. Critical Domain Insight: NULL = FALSE Semantics

> ⚠️ **Boolean indicator fields use NULL to mean FALSE, not "missing data".**

This is a **domain-specific convention** in the real estate platform:

| Pattern | Examples | NULL Means |
|---------|----------|------------|
| `has*` | `hasBalcony`, `hasElevator`, `hasParking` | Property does NOT have this feature |
| `is*` | `isQuiet`, `isChildFriendly`, `isSmokerFriendly` | Property is NOT this |
| `are*` | `arePetsAllowed` | Feature is NOT allowed |

**Example**: `hasElevator`
- 40% non-null (TRUE) → Building has an elevator
- 60% null → Building does **NOT** have an elevator
- **Semantic coverage**: 100%!

**Impact on Feature Engineering**:
- Don't impute NULL as missing
- NULL carries information (absence of feature)
- 50 boolean fields have 100% semantic coverage despite appearing "sparse"

---

## 3. Field Definitions & Coverage

### Contact Fields

| Field Type | Raw ETL Column | Coverage | Unique | Notes |
|------------|----------------|----------|--------|-------|
| **Email (primary)** | `listing.lister.email.hash` | 99.98% | 130,947 | Primary lister email |
| **Email (billing)** | `listing.lister.billing.email.hash` | 98.00% | 129,918 | Secondary email |
| **Phone (billing)** | `listing.lister.billing.phoneDay.hash` | 97.99% | 127,770 | **Preferred for graph** |
| **Phone (contact)** | `listing.lister.phone.hash` | 69.98% | 89,614 | Lower coverage |
| **Address** | `listing.address.address_hash` | 100.0% | ~234k | Unique per listing |
| **Coordinates** | `listing.address.geoCoordinates.lat`, `.lng` | 99.40% | 107,168 | High precision |

> 💡 **Domain Decision**: Use billing phone (98% coverage) instead of contact phone (70%) for fraud ring detection via shared contact graph edges.

### Boolean Indicator Fields

All 50 boolean indicators sorted by TRUE rate:

#### High TRUE Rate (>40%) - Common Features

| Feature | TRUE % | Raw ETL Column |
|---------|--------|----------------|
| `isFromNewInsertionFunnel` | 99.96% | `listing.meta.isFromNewInsertionFunnel` |
| `hasBalcony` | 71.19% | `listing.characteristics.hasBalcony` |
| `hasParking` | 55.14% | `listing.characteristics.hasParking` |
| `hasNiceView` | 47.72% | `listing.characteristics.hasNiceView` |
| `hasGarage` | 44.19% | `listing.characteristics.hasGarage` |
| `isChildFriendly` | 43.52% | `listing.characteristics.isChildFriendly` |
| `isQuiet` | 42.74% | `listing.characteristics.isQuiet` |
| `hasElevator` | 40.88% | `listing.characteristics.hasElevator` |

#### Moderate TRUE Rate (20-40%)

| Feature | TRUE % | Raw ETL Column |
|---------|--------|----------------|
| `hasWashingMachine` | 32.40% | `listing.characteristics.hasWashingMachine` |
| `hasGarden` | 32.30% | `listing.characteristics.hasGarden` |
| `isWheelchairAccessible` | 29.77% | `listing.characteristics.isWheelchairAccessible` |
| `arePetsAllowed` | 27.54% | `listing.characteristics.arePetsAllowed` |
| `hasBuiltInKitchen` | 26.92% | `listing.characteristics.hasBuiltInKitchen` |
| `hasCellar` | 23.93% | `listing.characteristics.hasCellar` |
| `hasCustomLinkInDescription` | 21.86% | `listing.meta.hasCustomLinkInDescription` |

#### Low TRUE Rate (5-20%) - Potentially Discriminative

| Feature | TRUE % | Raw ETL Column | Notes |
|---------|--------|----------------|-------|
| `hasLift` | 18.92% | `listing.characteristics.hasLift` | Similar to elevator |
| `isFurnished` | 17.31% | `listing.characteristics.isFurnished` | Rental type signal |
| `hasSwimmingPool` | 11.75% | `listing.characteristics.hasSwimmingPool` | Luxury signal |
| `hasCableConnection` | 11.40% | `listing.characteristics.hasCableConnection` | |
| `hasTerrace` | 10.92% | `listing.characteristics.hasTerrace` | |
| `isSmokerFriendly` | 10.40% | `listing.characteristics.isSmokerFriendly` | |
| `hasSauna` | 7.15% | `listing.characteristics.hasSauna` | Luxury signal |
| `hasDishwasher` | 6.83% | `listing.characteristics.hasDishwasher` | |

#### Very Low TRUE Rate (<5%) - Rare but Valid

| Feature | TRUE % | Raw ETL Column |
|---------|--------|----------------|
| `hasHomeDelivery` | 4.92% | `listing.characteristics.hasHomeDelivery` |
| `hasBathtub` | 4.07% | `listing.characteristics.hasBathtub` |
| `isBarrierFree` | 2.56% | `listing.characteristics.isBarrierFree` |
| `hasAlarmSystem` | 1.68% | `listing.characteristics.hasAlarmSystem` |
| `hasConcierge` | 1.09% | `listing.characteristics.hasConcierge` |
| `hasLibrary` | 0.54% | `listing.characteristics.hasLibrary` |
| `hasCornerBath` | 0.12% | `listing.characteristics.hasCornerBath` |
| `hasAirConditioning` | 0.03% | `listing.characteristics.hasAirConditioning` |

### Predictive Power Analysis (Deep Dive Results)

Based on correlation and fraud rate spread analysis:

#### Top Numeric Signals
| Feature | Correlation | Insight |
|---------|-------------|---------|
| `bundle.initialPrice` | 0.12 | Higher prices associated with fraud (targeting premium). |
| `listing.characteristics.numberOfRooms` | 0.13 | Anomalous room counts signal fake listings. |
| `listing.characteristics.yearBuilt` | 0.14 | Newer/Older distribution differences. |
| `listing.characteristics.numberOfFloors` | 0.10 | Detail level correlates with legitimacy. |

#### Production Baselines (Benchmarks)
| Feature | Type | Insight |
|---------|------|---------|
| `auto_approval_criteria.criteria.seonApproved` | Binary | **Primary Baseline**. If True (Approved) → Legit. If False → Fraud. Our model must beat this manual rule logic. |

#### Top Categorical Signals (Fraud Rate Spread)
| Feature | Max Spread | Insight |
|---------|------------|---------|
| `listing.lister.billing.phoneDay.area_hash` | 89% | **Critical**. Certain area codes are almost 100% fraud. |
| `listing.lister.email.domain_hash` | 86% | Disposable domains vs corporate. |
| `auto_approval_criteria.meta.state` | 75% | Proxy for previous automated decisions. |
| `listing.address.city_hash` | 64% | Geographic hotspots. |
| `bundle.tier` | 28% | Premium bundles targeted by fraudsters. |
| `listing.platforms` | 27% | Cross-posting behavior. |

---

## 4. Field Aliases & Naming Conventions

### Column Naming Strategy

All artifacts use **raw ETL column names** (no aliases):

| Convention | Example | When Used |
|------------|---------|-----------|
| **Raw ETL names** | `listing.address.geoCoordinates.latitude` | `raw_insertions.parquet`, `nodes_*.parquet` |
| **No aliases** | ❌ Not `lat` → Use full name | Everywhere (after cleanup) |
| **Local transformations** | `description_text` (coalesced from multiple fields) | Only in `graph_node_features.py` for embeddings |

**Rationale**: Single naming convention eliminates confusion and mapping errors.

---

## 5. Graph Relationships

### Entity Identification (Source of Truth)

| Entity | Unique ID | Raw ETL Column |
|--------|-----------|----------------|
| **Listing** | `object_reference` | `listing.objectReference` |
| **User** | `owner_id` | `listing.ownerId` |
| **Person** | `ppa_person_id` | `listing.lister.ppaPersonId` |
| **Region** | `region_code` | `listing.address.regionCode` |

### Edge Types & Coverage

| Edge Type | Source → Target | Coverage | Unique | Cardinality | Notes |
|-----------|-----------------|----------|--------|-------------|-------|
| **shared_contact_email** | Listing → Listing | 98.0% | 129,918 emails | Many-to-many | Primary fraud ring signal |
| **shared_billing_phone** | Listing → Listing | 97.99% | 127,770 phones | Many-to-many | Secondary fraud ring signal |
| **shared_ip** | Listing → Listing | 84.17% | 145,044 IPs | Many-to-many | Network-level connection |
| **shared_contact_phone** | Listing → Listing | 69.98% | 89,614 phones | Many-to-many | Lower coverage fallback |
| **same_owner** | Listing → Listing | 100% | 96,516 owners | Many-to-many | User repeat behavior |
| **same_ppa_person** | Listing → Listing | 94.01% | 94,887 persons | Many-to-many | Identity linkage |
| **same_region** | Listing → Listing | 100% | 26 regions | Many-to-many | Geographic clustering |
| **same_street** | Listing → Listing | 100% | 95,845 streets | Many-to-many | Hyper-local patterns |

**Graph Statistics**:
- **Nodes**: 234,458 listings
- **Edges**: ~10M+ (depends on edge type filtering)
- **Average Degree**: ~45 edges/node
- **Connected Components**: ~15k (many isolated nodes)

---

## 6. Domain-Specific Business Rules

### Fraud Indicators (Domain Knowledge)

Based on platform's fraud detection experience:

| Pattern | Why It's Suspicious | Coverage |
|---------|---------------------|----------|
| **Shared billing email** | Different accounts, same payment | 98% |
| **Shared phone number** | Coordinated fraud ring | 98% |
| **Premium tier + RENT** | Fraudsters target high-value rentals | Association rule: 5.72x lift |
| **INVOICE payment** | Bypass credit card verification | Association rule: 4.93x lift |
| **New accounts (<30 days)** | Cold start exploitation | Account age is top SHAP feature |
| **Low price outliers** | Too-good-to-be-true scams | Price-based anomaly detection |

### Legitimate Patterns (Not Fraud)

| Pattern | Why It's Legitimate | Notes |
|---------|---------------------|-------|
| **Real estate agents** | Many listings, same email | `customer_segment` field differentiates |
| **Property managers** | Shared company phone | Detected via high listing volume |
| **Family accounts** | Shared billing address | Context-dependent |

---

## 7. Data Quality Decisions

### Fields Excluded from Features

| Category | Reason | Examples |
|----------|--------|----------|
| **Labels** | Target variable | `fraud_flag`, `is_fraud` |
| **IDs** | No predictive value | `object_reference`, `user_id`, `owner_id` |
| **Timestamps** | Used for splitting only | `submission_at`, `first_published_date` |
| **External outputs** | Data leakage | `seonApproved`, `seonFraudScore`, `seonSession` |
| **High cardinality** | Too sparse | `*_hash` (except contacts), `ppaPersonId` |
| **Always NULL** | No information | 12 completely empty columns |

### Fields with Special Handling

| Field | Issue | Solution |
|-------|-------|----------|
| `ppaPersonId` | High cardinality (94k) | Use as graph edge, not feature |
| Boolean indicators | NULL = FALSE | No imputation needed |
| Contact hashes | Privacy-sensitive | Use for graph edges, not direct features |

## 8. Domain Glossary

### Real Estate Terms

| Term | Definition |
|------|------------|
| **Insertion** | A listing submission to the platform |
| **Object Reference** | Unique listing identifier |
| **PPA Person ID** | Platform's internal person identifier |
| **Bundle Tier** | Listing visibility package (basic, premium, top) |
| **Payment Type** | How user pays for listing (INVOICE, CREDIT_CARD, etc.) |
| **Region Code** | Swiss canton/region identifier (26 total) |

### Fraud Terms

| Term | Definition |
|------|------------|
| **Fraud Ring** | Group of coordinated fraudulent accounts |
| **Cold Start** | Brand new account with no history |
| **Concept Drift** | Fraud patterns changing over time |
| **SEON** | External fraud detection system (baseline) |

## 9. Contact Coverage Summary

**Decision Tree for Graph Edges**:

```
Need to link listings by contact?
├─ Email? → Use BOTH (Contact + Billing). Billing has higher coverage (98%).
├─ Phone? → Use BOTH (Contact + Billing). Billing has higher coverage (98% vs 70%).
└─ Address? → Use address hash (100% coverage but too unique)
```

**Why billing fields > contact fields**:
- Billing email: 98% vs contact email 70%
- Billing phone: 98% vs contact phone 70%
- Billing info harder to fake (payment verification)
- More stable (less likely to change)

---