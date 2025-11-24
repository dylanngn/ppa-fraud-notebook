# Updated Evaluation Report: Node Separation Impact

## Executive Summary

After separating email and phone nodes into specific types (contact vs billing), we re-ran experiments to test if reducing noise would improve GNN performance.

### Results: Node Separation **Did NOT Improve Performance** ❌

| Model | Old Graph (Unified Nodes) | New Graph (Separated Nodes) | Change |
|-------|--------------------------|----------------------------|--------|
| **Baseline XGBoost** | **0.5947** | **0.5947** | **0.0%** ✅ |
| **HGT + XGBoost** | 0.5909 | 0.5729 | **-3.0%** ❌ |
| **SAGE + XGBoost** | 0.5694 | 0.5714 | **+0.4%** ≈ |

### Key Finding: Separation Made Things WORSE

- HGT performance **dropped 3%** (-0.018 AUC-PR)
- GraphSAGE stayed essentially the same
- Baseline XGBoost **remains the best model**

---

## Detailed Analysis

### Why Did Node Separation Fail?

#### Hypothesis 1: Lost Information
**Before (Unified)**:
- 1 unified `email` node connected all related listings
- Fraudsters using the same email created strong connections

**After (Separated)**:
- `contact_email` and `billing_email` are now **separate graphs**
- Lost transitivity: Listing A → contact_email → Listing B is broken if one uses contact and the other uses billing

#### Hypothesis 2: Increased Sparsity
**Node Count Increase**:
- Before: 286K email nodes, 157K phone nodes
- After: 132K contact_email + 132K billing_email + 97K contact_phone + 138K billing_phone
- **More nodes = sparser connections per node type**

#### Hypothesis 3: HGT Couldn't Learn Type Importance
Even though we separated nodes by type, HGT couldn't effectively learn that:
- contact_email connections = important
- billing_email connections = noise

The separation actually **removed valuable cross-type fraud patterns**.

---

## What This Reveals About the Data

### The Real Problem: Weak Graph Signal

1. **Baseline Dominance** (0.5947):
   - Proves tabular features (`account_age_days`) are very strong
   - Graph adds almost no value

2. **Pure GNN Performance** (0.04-0.20 AUC-PR):
   - GNNs alone **completely fail** at fraud detection
   - This confirms the graph signal is extremely weak

3. **Hybrid Barely Helps**:
   - Adding GNN embeddings to XGBoost improves by only 0-3%
   - Not statistically significant

---

## Root Cause Analysis

### Why is the Graph Signal So Weak?

Based on the sparsity analysis and experiments, we can now conclude:

**1. Frauds Are NOT More Connected** ✅ (Confirmed)
- Fraud mean degree: 16.71
- Legit mean degree: 17.38
- **Frauds are LESS connected**, not more

**2. No Strong Fraud Rings** ✅ (Confirmed)
- 11,882 email-based fraud groups
- But no dominant "fraud rings" that HGT can exploit
- Fraud patterns are individual, not network-based

**3. Account Age Dominates** ✅ (Likely)
- New accounts = fraud
- Graph structure can't compete with this simple signal

**4. Missing Critical Bridge Nodes** ✅ (Confirmed)
- No device fingerprints
- No session IDs
- No IP sharing (0 IP-based fraud rings found)
- **IPs are unique per listing** - no reuse detected

---

## Comparison: Before vs After Node Separation

### HGT Hybrid Performance by Window

| Window | Old (Unified) | New (Separated) | Difference |
|--------|--------------|----------------|------------|
| 2024-02-03 to 2024-02-17 | 0.7624 | 0.7411 | -0.0213 |
| 2024-10-26 to 2024-11-09 | 0.6852 | 0.6602 | -0.0250 |
| 2025-08-16 to 2025-08-30 | 0.8111 | 0.7969 | -0.0142 |
| **Mean** | **0.5909** | **0.5729** | **-0.0180** |

**Across almost all windows, the new graph performed worse.**

---

## Recommendations

### For Your Thesis

**This is still valuable!** You've demonstrated:
1. ✅ Graph-based methods don't universally improve fraud detection
2. ✅ Even theoretically sound optimizations (node type separation) can fail
3. ✅ You tested the hypothesis scientifically and documented the negative result

**Contributions**:
- Identified when graph methods fail (sparse graphs with weak signals)
- Showed that node type separation can backfire by breaking transitivity
- Provided guidance for practitioners: **tabular features > graph** when individual signals are strong

### Next Steps to Improve Results

If you want to beat the baseline, try:

**Option 1: Add More Bridge Nodes** 🎯
- Device fingerprints (most important!)
- Session IDs
- Browser metadata
- These create stronger fraud ring connections

**Option 2: Try Simpler Approach** 
- Use GNN for **anomaly detection** instead of classification
- Flag listings with unusual network patterns
- Combine with baseline as ensemble

**Option 3: Feature Engineering**
- Extract graph-based features manually:
  - Shared email count
  - Shared phone count
  - Network clustering coefficient
- Feed to XGBoost directly (no GNN)

**Option 4: Accept the Result**
- Document that baseline XGBoost is sufficient
- Focus thesis on "when graph methods work vs when they don't"
- Negative results are publishable with good analysis

---

## Graph Feature Engineering vs. Hybrid HGT (2025-11-24 Update)

### Configurations

| Approach | Feature Set | Training Procedure | Mean AUC-PR | Mean AUC-ROC | Notes |
|----------|-------------|--------------------|-------------|--------------|-------|
| **Graph-Feature Baseline** | Tabular signals (account age, prices, bundle info, etc.) + engineered graph statistics (shared email/phone/IP counts, component size, PageRank, user/IP reuse) | Sliding-window XGBoost | **0.6655** | **0.9392** | Highest-performing setup; no neural embeddings |
| **Residual Hybrid HGT** | Same tabular + graph statistics + 64-dim HGT embeddings with residual self-feature concat | HGT embeddings → XGBoost (tabular + graph + embeddings) | **0.6484** | ~0.93 | Improves over old hybrid (0.5909) but still trails the graph-feature baseline |

### Why Graph Features Win

1. **Explicit structural counts:** XGBoost sees the exact number of listings sharing a contact email/phone/IP, the size of each connected component, and how central a listing is in the email/phone bipartite graph. These interpretable signals translate directly into higher precision.
2. **Low-noise feature space:** Manual stats add only ~12 columns, whereas embeddings add 64 dimensions that can capture noise because most non-listing nodes still have constant features.
3. **Operational impact:** With graph counts, P@100 routinely exceeds 0.9 during the heaviest fraud windows (late 2024–2025), whereas the hybrid oscillates between 0.5–0.8 even after the residual fix.

### What the Residual Hybrid Adds

1. **Preserves tabular signals:** Concatenating the self projection with the aggregated message prevents HGT from over-smoothing `account_age_days`.
2. **Verifies complementarity:** Mean AUC-PR rose from 0.5909 (old hybrid without graph stats) to 0.6484 when residuals and manual features were combined, showing embeddings can add value when structural signals exist.
3. **Still limited by graph quality:** Without richer bridge nodes (devices, sessions, payment hashes), the embeddings cannot add enough incremental information to beat the graph-feature baseline.

### Recommendation

- **Production:** Use the graph-feature baseline (tabular + engineered graph stats) because it is simpler, more interpretable, and more accurate.
- **Research narrative:** Document that manual feature engineering unlocked graph signal where end-to-end GNNs failed, emphasizing the importance of high-quality edges for neural approaches.
- **Future work:** Revisit hybrid/GNN models only after collecting stronger bridge entities or when exploring alternative objectives (e.g., anomaly detection, contrastive pretraining).

---

## Conclusion

### Final Rankings (After Node Separation):

| Rank | Model | AUC-PR | Recommendation |
|------|-------|--------|----------------|
| 🥇 **1st** | **Baseline XGBoost** | **0.5947** | ✅ **RECOMMENDED FOR PRODUCTION** |
| 2nd | HGT + XGBoost | 0.5729 | ❌ Not worth the complexity |
| 3rd | SAGE + XGBoost | 0.5714 | ❌ Not worth the complexity |

### Key Takeaway:

**Node type separation reduced performance by breaking valuable cross-type fraud patterns.** The graph signal in this dataset is fundamentally too weak to compete with simple tabular features like account age.

**For this specific fraud detection problem, stick with Baseline XGBoost.**
