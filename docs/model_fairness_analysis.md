# Model Fairness Analysis: Baseline vs Hybrid Models

## Executive Summary

This analysis evaluates whether the baseline and hybrid models receive fair treatment in terms of:
1. **Data Richness**: Feature availability and quality
2. **Accumulated Window**: Temporal data access strategy
3. **Evaluation Strategy**: Test set fairness and temporal consistency

## Critical Issues Found

### 🔴 **CRITICAL: Data Leakage in Hybrid Models**

**Problem**: GNN embeddings are generated using the **FULL graph** (including future data), then used in accumulating window training.

**Location**: 
- `train_hybrid_sage.py:254-257` - Generates embeddings on full graph
- `train_hybrid_hgt.py:337-345` - Generates embeddings on full graph

**Impact**: 
- Hybrid models have access to future information through embeddings
- This creates an unfair advantage over baseline models
- Results are not comparable

**Evidence**:
```python
# train_hybrid_sage.py:254-257
# Generate embeddings for all nodes (reuse full_data from evaluation)
print("\nGenerating Full Graph Embeddings...")
model.eval()
with torch.no_grad():
    z_listing = model(full_data.x_dict, full_data.edge_index_dict)
```

The `full_data` contains ALL listings, including those in the test set.

---

### 🟡 **MODERATE: Graph Features Temporal Inconsistency**

**Problem**: Graph features are computed on the **ENTIRE dataset** without temporal filtering.

**Location**:
- `src/features/graph_features.py` - No temporal filtering
- `src/features/advanced_graph_features.py` - No temporal filtering  
- `src/features/time_weighted_features.py` - Uses all historical data (including future)
- `src/features/interaction_features.py` - Uses all historical data

**Impact**:
- When baseline uses `include_graph_features=True`, it gets features computed with future information
- This affects accumulating window training where test data is in the future
- However, this affects BOTH baseline and hybrid equally if both use graph features

**Evidence**:
```python
# time_weighted_features.py:193
# Days difference: positive = other is older
days_diff = (ltime - other_time).total_seconds() / 86400.0

if days_diff < 0:
    # Other listing is in the future, skip
    continue
```

While `time_weighted_features.py` skips future listings, other graph features don't have this protection.

---

### 🟢 **MINOR: Feature Set Differences**

**Problem**: Baseline and hybrid models use different feature sets.

**Baseline (with graph features)**:
- Base tabular features (account_age_days, log_price, etc.)
- Graph features (contact_email_count, shared_contact_email_count, etc.)
- Advanced graph features (degree_total, is_isolated, etc.)
- Time-weighted features (email_count_7d, phone_velocity_7d, etc.)
- Interaction features (new_account_high_email_reuse, etc.)

**Hybrid Models**:
- Base tabular features (same as baseline)
- GNN embeddings (64-dim vectors from SAGE/HGT)
- **NO graph features** (explicitly excluded in `feature_engineering()`)

**Location**: `train_hybrid_sage.py:346` and `train_hybrid_hgt.py:434`
```python
# Step 3: Feature engineering
df = feature_engineering(df)  # include_graph_features=False by default
```

**Impact**:
- Hybrid models miss rich graph-derived features (shared counts, component sizes, etc.)
- Only get learned embeddings, which may not capture all graph patterns
- This is actually a **disadvantage** for hybrid models

---

## Detailed Analysis

### 1. Accumulated Window Strategy

**Status**: ✅ **FAIR** - All models use the same strategy

**Implementation**: `train_accumulating_window()` in `train_baseline.py:274`

**Strategy**:
- Initial window: 180 days
- Step size: 7 days
- Test window: 14 days
- Training data: ALL data from start to `train_end` (accumulating)
- Test data: Next 14 days after `train_end`

**Code**:
```python
# train_baseline.py:405-410
# ACCUMULATING WINDOW: Use ALL data from start to train_end
train_data = df.filter(pl.col("submission_at") < train_end)
test_data = df.filter(
    (pl.col("submission_at") >= train_end) & 
    (pl.col("submission_at") < test_end)
)
```

**Fairness**: ✅ All models use this same function, so window strategy is identical.

---

### 2. Evaluation Strategy

**Status**: ✅ **FAIR** - All models use identical test sets

**Implementation**: Same `train_accumulating_window()` function for all models

**Test Set Construction**:
- Same temporal split logic
- Same minimum size requirements (50 test samples, 1000 train samples)
- Same fraud rate tracking

**Fairness**: ✅ Test sets are identical across models.

---

### 3. Data Richness Comparison

#### Baseline Model (with graph features)

**Features**:
1. **Base Tabular** (13 features):
   - account_age_days, log_price, living_space, rooms
   - is_new, has_balcony, has_elevator, has_parking
   - bundle_period, bundle_tier_score
   - is_direct_payment, is_buy
   - latitude, longitude

2. **Graph Features** (12 features):
   - contact_email_count, shared_contact_email_count, max_shared_contact_email
   - contact_phone_count, shared_contact_phone_count, max_shared_contact_phone
   - user_listing_count, user_unique_ip_count
   - shared_ip_user_count, max_shared_ip_users
   - listing_component_size, listing_pagerank

3. **Advanced Graph Features** (5 features):
   - degree_total, is_isolated, unique_identifier_count
   - neighbor_overlap_score, avg_neighbor_degree

4. **Time-Weighted Features** (25 features):
   - Email: total_historical, count_7d, count_30d, recent_weighted, velocity_7d, acceleration, is_burst, is_dormant_reactivation, recency_weighted, time_spread
   - Phone: (same 10 features)
   - Combined: velocity_7d, acceleration, any_burst, any_dormant_reactivation, recency_weighted

5. **Interaction Features** (14 features):
   - Binary: new_account_high_email_reuse, new_account_high_phone_reuse, new_account_high_reuse_any, new_account_invoice_payment, very_new_account_invoice, is_small_listing, new_account_small_listing, in_large_component, new_account_large_component, new_account_isolated
   - Continuous: account_age_risk_score, email_reuse_intensity, component_risk_score, suspicious_combo_score

**Total**: ~69 features

#### Hybrid Models (SAGE/HGT)

**Features**:
1. **Base Tabular** (13 features): Same as baseline
2. **GNN Embeddings** (64 features): Learned graph representations
3. **NO graph features**: Explicitly excluded

**Total**: ~77 features (but different information content)

**Analysis**:
- Hybrid models have more features (77 vs 69), but miss explicit graph patterns
- GNN embeddings are learned end-to-end, but may not capture all patterns that explicit features do
- This is a **trade-off**, not necessarily an advantage

---

### 4. Temporal Consistency Issues

#### Issue 1: GNN Embedding Generation

**Problem**: Embeddings generated on full graph (including test data)

**Current Flow**:
1. Train GNN on 80% split (temporal)
2. Evaluate on 20% split
3. **Generate embeddings on FULL graph** ← Problem here
4. Use embeddings in accumulating window training

**Correct Flow Should Be**:
1. For each accumulating window:
   - Filter graph to only include edges/nodes before `train_end`
   - Generate embeddings on filtered graph
   - Use embeddings for that window's training

**Impact**: High - This is data leakage

#### Issue 2: Graph Feature Computation

**Problem**: Graph features computed on entire dataset

**Current Flow**:
- Graph features are pre-computed once on all data
- Used in accumulating window training

**Correct Flow Should Be**:
- For each accumulating window:
  - Compute graph features using only data before `train_end`
  - This is computationally expensive but temporally correct

**Impact**: Medium - Affects both baseline and hybrid if graph features are used

---

## Recommendations

### Priority 1: Fix Data Leakage in Hybrid Models

**Action**: Generate embeddings per accumulating window

**Implementation**:
```python
def train_hybrid_with_temporal_embeddings():
    # For each accumulating window:
    for train_end in window_dates:
        # Filter graph to only include data before train_end
        filtered_graph = filter_graph_by_time(full_graph, train_end)
        
        # Generate embeddings on filtered graph
        embeddings = generate_embeddings(filtered_graph)
        
        # Train XGBoost on this window with these embeddings
        train_xgboost_window(embeddings, train_end)
```

**Files to Modify**:
- `train_hybrid_sage.py`: Modify `main()` to generate embeddings per window
- `train_hybrid_hgt.py`: Same modification

### Priority 2: Make Graph Features Temporal

**Action**: Compute graph features per accumulating window

**Implementation**:
- Add temporal filtering to graph feature computation
- Or compute features on-the-fly during accumulating window training

**Files to Modify**:
- `src/features/graph_features.py`: Add temporal filtering
- `src/features/advanced_graph_features.py`: Add temporal filtering
- `src/features/time_weighted_features.py`: Already has some temporal awareness, but needs window filtering
- `src/features/interaction_features.py`: Add temporal filtering

### Priority 3: Feature Set Alignment

**Action**: Give hybrid models access to graph features too

**Rationale**: 
- Hybrid models currently miss explicit graph features
- Adding them would make comparison fairer
- GNN embeddings + explicit features might be complementary

**Implementation**:
```python
# In train_hybrid_sage.py and train_hybrid_hgt.py
df = feature_engineering(df, include_graph_features=True)  # Add graph features
```

---

## Summary Table

| Aspect | Baseline | Hybrid | Fair? | Issue |
|--------|----------|--------|-------|-------|
| **Accumulating Window** | ✅ Yes | ✅ Yes | ✅ Yes | None |
| **Test Set** | ✅ Same | ✅ Same | ✅ Yes | None |
| **Base Features** | ✅ 13 | ✅ 13 | ✅ Yes | None |
| **Graph Features** | ✅ 69 | ❌ 0 | ⚠️ No | Hybrid missing features |
| **GNN Embeddings** | ❌ No | ✅ 64 | ⚠️ No | Baseline doesn't have |
| **Temporal Consistency** | ⚠️ Partial | ❌ No | ❌ No | **Data leakage in hybrid** |
| **Future Data Access** | ⚠️ Graph features | ❌ Embeddings | ❌ No | Both have issues |

---

## Conclusion

**Main Finding**: The hybrid models have **data leakage** through embeddings generated on the full graph. This makes performance comparisons invalid.

**Secondary Finding**: Graph features are computed on the full dataset, affecting temporal consistency for both models.

**Recommendation**: 
1. **Immediate**: Fix embedding generation to be temporal per window
2. **Short-term**: Make graph features temporal
3. **Long-term**: Align feature sets for fair comparison

The evaluation strategy (accumulating window) is fair, but the data preparation is not temporally consistent.

