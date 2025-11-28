# Field Usage Analysis & Recommendations

Based on data quality report (234,458 rows, 292 columns) and code review.

## Summary

- **Unusable fields (≥95% null):** 175 fields
- **High coverage fields (<50% null):** 82 fields
- **Currently used fields:** Need optimization

## Fields to REMOVE (Very Low Coverage)

### 1. Contact Inquiry/Viewing Fields (99.89%+ null)
**Location:** `src/data/create_graph_artifacts.py`

**Remove:**
- `listing.lister.contacts.inquiry.email.hash` (99.89% null, only 254 values)
- `listing.lister.contacts.viewing.email.hash` (99.99% null, only 22 values)
- `listing.lister.contacts.inquiry.phone.hash` (99.89% null, only 254 values)
- `listing.lister.contacts.inquiry.mobile.hash` (99.89% null, only 254 values)
- `listing.lister.contacts.viewing.phone.hash` (99.89% null, only 249 values)
- `listing.lister.contacts.viewing.mobile.hash` (99.89% null, only 249 values)
- `listing.lister.contacts.inquiry.name_hash` (99.98% null, only 57 values)

**Impact:** These fields contribute almost nothing (0.1% coverage) and add noise.

### 2. Billing Phone Mobile (99.78% null)
**Location:** `src/data/create_graph_artifacts.py:138`

**Remove:**
- `listing.lister.billing.phoneMobile.hash` (99.78% null, only 512 values)

**Impact:** Minimal - use `phoneDay.hash` instead (97.99% coverage).

### 3. Auto Approval Criteria (100% null)
**Location:** `src/data/create_graph_artifacts.py:99`

**Remove:**
- `auto_approval_criteria.criteria.criteria.seonApproved` (100% null)

**Impact:** Field doesn't exist in data.

## Fields to KEEP (High Coverage)

### Email Fields (Excellent Coverage)
✅ **KEEP:**
- `listing.lister.email.hash` - 99.976% coverage (130,947 unique) ⭐
- `listing.lister.email.domain_hash` - 99.976% coverage (24,442 unique)
- `listing.lister.email.user_hash` - 99.976% coverage (116,353 unique)
- `listing.lister.billing.email.hash` - 98.002% coverage (129,918 unique) ⭐

### Phone Fields (Good Coverage)
✅ **KEEP:**
- `listing.lister.billing.phoneDay.hash` - 97.995% coverage (127,770 unique) ⭐ BEST
- `listing.lister.phone.hash` - 69.978% coverage (89,614 unique) ⭐ GOOD

**Note:** The report shows multiple `phoneDay.hash` entries with different coverage. Need to verify exact field name.

### Address Fields (Excellent Coverage)
✅ **KEEP:**
- `listing.address.country_hash` - 100% coverage ⭐
- `listing.address.zip_hash` - 99.99% coverage ⭐
- `listing.address.city_hash` - 99.99% coverage ⭐
- `listing.address.street_hash` - Need to check in report
- `listing.address.geoCoordinates.latitude` - 99.404% coverage
- `listing.address.geoCoordinates.longitude` - 99.404% coverage

## Fields to ADD (High Coverage, Not Currently Used)

### 1. Email Component Hashes
**Add to graph features:**
- `listing.lister.email.domain_hash` - 99.976% coverage
- `listing.lister.email.user_hash` - 99.976% coverage

**Rationale:** Domain-based features can detect shared email providers (fraud indicator).

### 2. Additional Phone Fields
**Check coverage and consider:**
- `listing.lister.mobile.hash` - Need to verify coverage from report
- Other `phoneDay.hash` variants with good coverage

### 3. Address Region
**Add:**
- `listing.address.region` - 99.71% coverage (127 unique values)

**Rationale:** Regional patterns can be useful for fraud detection.

## Code Changes Required

### 1. `src/data/create_graph_artifacts.py`

**Remove from email_cols (lines 110-115):**
```python
# REMOVE these:
"listing.lister.contacts.inquiry.email.hash",
"listing.lister.contacts.viewing.email.hash",
```

**Remove from phone_cols (lines 135-143):**
```python
# REMOVE these:
"listing.lister.billing.phoneMobile.hash",  # 99.78% null
"listing.lister.contacts.inquiry.phone.hash",  # 99.89% null
"listing.lister.contacts.inquiry.mobile.hash",  # 99.89% null
"listing.lister.contacts.viewing.phone.hash",  # 99.89% null
"listing.lister.contacts.viewing.mobile.hash",  # 99.89% null
```

**Remove person node creation (lines 192-205):**
- `listing.lister.contacts.inquiry.name_hash` is 99.98% null
- Consider removing person nodes entirely or finding alternative source

**Remove from edges (lines 242-245):**
```python
# REMOVE:
edges_listing_inquiry_email = create_edge_df(
    "object_reference",
    "listing.lister.contacts.inquiry.email.hash"  # 99.89% null
)
```

**Remove from edges (lines 278-282):**
```python
# REMOVE or make optional:
edges_listing_person = df_listings.select([
    pl.col("object_reference").alias("source"),
    get_col("listing.lister.contacts.inquiry.name_hash").alias("target")  # 99.98% null
]).filter(pl.col("target").is_not_null()).unique()
```

### 2. `src/data/graph_builder.py`

**Check edge file references:**
- `edges_listing_inquiry_email.parquet` - Will be empty/removed
- `edges_listing_has_person.parquet` - Will be mostly empty

**Make these optional** (already handled with `_ensure_artifact` checks).

### 3. Feature Files

**No changes needed** - Feature files use edge parquet files, which will automatically exclude low-coverage edges.

## Recommended Priority

### High Priority (Do First)
1. ✅ Remove inquiry/viewing contact fields (99.89%+ null)
2. ✅ Remove `phoneMobile.hash` (99.78% null)
3. ✅ Remove `auto_approval_criteria.seonApproved` (100% null)

### Medium Priority
4. Consider adding `listing.lister.email.domain_hash` as separate node type
5. Add `listing.address.region` to listing nodes

### Low Priority
6. Review if person nodes are needed (99.98% null)
7. Consider email component hashes (domain/user) for graph features

## Expected Impact

**After cleanup:**
- Cleaner graph structure (fewer empty edges)
- Faster graph building (less processing of null fields)
- More accurate features (no noise from 0.1% coverage fields)
- Better model performance (focus on high-signal fields)

## Verification Steps

1. Run ETL to regenerate artifacts
2. Check edge parquet file sizes (should be larger for kept fields)
3. Verify graph building completes without warnings
4. Compare feature counts before/after

