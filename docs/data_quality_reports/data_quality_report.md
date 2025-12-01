# Data Quality Report

**Dataset:** `artifacts/raw_insertions.parquet`  
**Total Rows:** 234,458  
**Total Columns:** 292  
**Generated:** 2025-11-30 17:54:56

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Unusable Fields (≥95.0% null) | 175 |
| High Coverage Fields (<50.0% null) | 82 |
| Total Fields Analyzed | 292 |

---

## Unusable Fields (>95.0% Null)

Found **175 fields** that are mostly null.

| Column | Null % | Non-Null Count | dtype |
|--------|--------|----------------|-------|
| `listing.valueAddedServices` | 100.0% | 0 | Null |
| `listing.prices.rent` | 100.0% | 0 | Null |
| `listing.prices.buy` | 100.0% | 0 | Null |
| `listing.lister.website` | 100.0% | 0 | Null |
| `listing.lister.contacts.viewing...` | 100.0% | 0 | Null |
| ... | ... | ... | ... |
| `listing.meta.listingHash` | 95.79% | 9,879 | String |
| `listing.meta.softwareName` | 95.41% | 10,750 | String |
| `listing.characteristics.hasSwimmingPool` | 95.25% | 11,134 | Boolean |
| `listing.characteristics.isSmokerFriendly` | 95.12% | 11,430 | Boolean |
| `listing.characteristics.cubage` | 95.08% | 11,541 | Float64 |

> *Full list available in [01_all_fields_null_stats.csv](01_all_fields_null_stats.csv)*

---

## High Coverage Fields (<50.0% Null)

All **82 fields** with high coverage (sorted by coverage, highest first):

| Column | Null % | Non-Null Count | Coverage % | dtype |
|--------|--------|----------------|------------|-------|
| `account_created_at` | 0.0% | 234,458 | 100.0% | Datetime |
| `listing.address.address_hash` | 0.0% | 234,458 | 100.0% | String |
| `listing.address.country_hash` | 0.0% | 234,458 | 100.0% | String |
| `listing.contactForm.deliveryFormat` | 0.0% | 234,458 | 100.0% | String |
| `listing.contactForm.size` | 0.0% | 234,458 | 100.0% | String |
| ... | ... | ... | ... | ... |
| `listing.lister.phone.number_hash` | 30.02% | 164,070 | 69.98% | String |
| `listing.lister.phone.area_hash` | 30.05% | 164,008 | 69.95% | String |
| `listing.lister.phone.country_hash` | 31.14% | 161,443 | 68.86% | String |
| `listing.characteristics.hasParking` | 44.86% | 129,285 | 55.14% | Boolean |
| `listing.characteristics.yearBuilt` | 46.95% | 124,376 | 53.05% | Int64 |

> *Full list available in [03_high_coverage_fields.csv](03_high_coverage_fields.csv)*

---

## Phone Fields Analysis

Found **21 phone fields**:

| Field | Null % | Coverage % | Unique Count | dtype |
|-------|--------|------------|--------------|-------|
| `listing.lister.billing.phoneData.hash` | 2.00% | 97.99% | 127,770 | String |
| `listing.lister.billing.phoneData.area_hash` | 2.00% | 97.99% | 14,959 | String |
| `listing.lister.billing.phoneData.number_hash` | 2.03% | 97.97% | 7,059 | String |
| `listing.lister.billing.phoneData.country_hash` | 4.24% | 95.76% | 203 | String |
| `listing.lister.phone.hash` | 30.02% | 69.98% | 89,614 | String |
| ... | ... | ... | ... | ... |
| `listing.lister.billing.phoneMobile...` | 99.78% | 0.22% | 17 | String |
| `listing.lister.contacts.inquiry.phone.hash` | 99.89% | 0.11% | 111 | String |
| `listing.lister.contacts.inquiry.phone.number_hash` | 99.89% | 0.11% | 111 | String |
| `listing.lister.contacts.inquiry.phone.area_hash` | 99.89% | 0.11% | 73 | String |
| `listing.lister.contacts.inquiry.phone.country_hash` | 99.98% | 0.02% | 8 | String |

> *Full list available in [04_phone_fields.csv](04_phone_fields.csv)*

---

## Email Fields Analysis

Found **16 email fields**:

| Field | Null % | Coverage % | Unique Count | dtype |
|-------|--------|------------|--------------|-------|
| `listing.lister.email.hash` | 0.02% | 99.98% | 130,947 | String |
| `listing.lister.email.domain_hash` | 0.02% | 99.98% | 24,442 | String |
| `listing.lister.email.user_hash` | 0.02% | 99.98% | 116,353 | String |
| `auto_approval_criteria.criteria.email` | 1.17% | 98.83% | 3 | Boolean |
| `listing.lister.billing.email.hash` | 2.00% | 98.00% | 129,918 | String |
| ... | ... | ... | ... | ... |
| `listing.lister.contacts.viewing.email.hash` | 99.99% | 0.01% | 11 | String |
| `listing.lister.contacts.viewing.email.domain_hash` | 99.99% | 0.01% | 15 | String |
| `listing.lister.contacts.viewing.email.user_hash` | 99.99% | 0.01% | 15 | String |
| `listing.lister.billing.directPayment.email` | 99.99% | 0.01% | 15 | String |
| `listing.autoApprovalCriteria.criteria.email` | 100.00% | 0.00% | 2 | Boolean |

> *Full list available in [05_email_fields.csv](05_email_fields.csv)*

---

## Address Fields Analysis

Found **38 address fields**:

| Field | Null % | Coverage % | Unique Count | dtype |
|-------|--------|------------|--------------|-------|
| `listing.address.country_hash` | 0.0% | 100.0% | 95 | String |
| `listing.address.zip_hash` | 0.01% | 99.99% | 4,531 | String |
| `listing.address.city_hash` | 0.01% | 99.99% | 5,982 | String |
| `listing.address.region` | 0.29% | 99.71% | 127 | String |
| `listing.address.geoCoordinates.lat` | 0.60% | 99.40% | 107,168 | Float64 |
| ... | ... | ... | ... | ... |
| `listing.address.geoCoordinates.isManual` | 99.998% | 0.002% | 3 | Boolean |
| `listing.address.geoDistances` | 99.998% | 0.002% | 4 | String |
| `listing.address.geoTags` | 99.998% | 0.002% | 4 | String |
| `listing.characteristics.craneCapacity` | 99.998% | 0.002% | 4 | Int64 |
| `listing.lister.billing.address...` | 100.0% | 0.0% | 1 | Null |

> *Full list available in [06_address_fields.csv](06_address_fields.csv)*

---

## Data Type Distribution

| Data Type | Count |
|-----------|-------|
| String | 171 |
| Boolean | 54 |
| Int64 | 29 |
| Float64 | 20 |
| Null | 12 |
| Datetime | 5 |
| Int32 | 1 |

---

## Numeric Fields Summary

**50 numeric fields** sorted by coverage:

| Column | Null % | Coverage % | Min | Max | Median | Std | Unique | dtype |
|--------|--------|------------|-----|-----|--------|-----|--------|-------|
| `listing.legacy.personId` | 0% | 100% | 28.0 | — | 1.66e15 | 8.76e14 | 133,821 | Int64 |
| `user_id` | 0% | 100% | 3.0 | — | 83,5140 | 324,809.9 | 133,811 | Int32 |
| `listing.address.geoCoordinates.lat` | 0.6% | 99.4% | -89.94 | — | 47.29 | 1.33 | 107,168 | Float64 |
| `listing.address.geoCoordinates.lng` | 0.6% | 99.4% | -151.74 | — | 8.35 | 2.90 | 107,664 | Float64 |
| `auto_approval_criteria.meta.seonScore` | 1.5% | 98.5% | 0.0 | — | 0.0 | 20.76 | 5,104 | Float64 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

> *Full list available in [08_numeric_fields.csv](08_numeric_fields.csv)*

---

## Categorical Fields - Top Values

### `listing.address.address_hash`

| Value (hash) | Count |
|--------------|-------|
| `ecbad8a5...` | 11 |
| `8bcca3c2...` | 7 |
| `14fc0a5c...` | 7 |
| `6b14cfdd...` | 4 |
| `0da890a4...` | 4 |
| `b62a2366...` | 2 |
| *other values* | 1 each |

### `listing.address.city_hash`

| Value (hash) | Count |
|--------------|-------|
| `acc476e5...` | 202 |
| `bcc4443e...` | 50 |
| `1dddaa05...` | 18 |
| `9a5b90e1...` | 11 |
| `2ac501de...` | 6 |
| `b4634af8...` | 4 |
| *other values* | 1 each |

### `listing.address.country_hash`

| Value (hash) | Count |
|--------------|-------|
| `ad52dcae...` | 1,012 |
| `13eba3a1...` | 477 |
| `21e43304...` | 80 |
| `fe0fef20...` | 50 |
| `f9a486b9...` | 5 |
| `132fdd4c...` | 2 |
| `752fc0fd...` | 2 |
| *other values* | 1-2 each |

### `listing.address.zip_hash`

| Value (hash) | Count |
|--------------|-------|
| `e8d347cc...` | 29 |
| `2c2c7899...` | 17 |
| `f6160158...` | 14 |
| `eda609ea...` | 11 |
| `ff5966c5...` | 10 |
| `042a23b7...` | 4 |
| `603081ac...` | 2 |
| *other values* | 1 each |

> *Full categorical analysis available in [09_categorical_fields/](09_categorical_fields/)*

---

## Data Consistency Checks

| Check | Metric | Value | Notes |
|-------|--------|-------|-------|
| Duplicate object_reference | unique_pct | 100.0% | ✅ All unique |
| Empty object_reference | valid_pct | 100.0% | ✅ No empty values |
| Flattened listing coverage | coverage_pct | 98.21% | ✅ High coverage |
| Date range: account_created_at | date_range | — | 2020-12-17 → 2025-11-25 |
| Date range: first_published_date | date_range | — | 2023-01-01 → 2025-11-11 |

---

## Quick Reference

### Key Observations

1. **High Sparsity:** 175 out of 292 columns (60%) have ≥95% null values
2. **Core Identifiers:** `user_id`, `account_created_at`, and address fields have 100% coverage
3. **Contact Info Coverage:**
   - Email: ~99.98% coverage
   - Phone: ~70% coverage for lister phone, ~98% for billing phone
4. **Date Range:** Data spans ~5 years (Dec 2020 - Nov 2025)
5. **No Duplicates:** All records have unique object references

### Recommended Actions

- Focus feature engineering on the 82 high-coverage fields
- Consider imputation strategies for phone fields (~30% missing)
- Drop the 175 unusable fields to reduce dimensionality


