# Experiment Journal

## Experiment 1: Node Separation (COMPLETED)

**Date**: 2025-11-24
**Goal**: Reduce noise by separating contact vs billing emails/phones
**Hypothesis**: HGT will learn to weight contact edges higher than billing edges

### Implementation:
- Separated `email` into: `account_email`, `contact_email`, `billing_email`
- Separated `phone` into: `contact_phone`, `billing_phone`
- Total: 5 node types instead of 2

### Results:
| Configuration | AUC-PR | Change |
|--------------|--------|--------|
| Baseline XGBoost | 0.5947 | - |
| HGT Unified | 0.5909 | -0.6% |
| HGT Separated | 0.5729 | -3.7% |

### Findings:
❌ **FAILED** - Performance dropped 3%

**Root Cause**: Breaking emailsphones into separate types destroyed transitivity:
- Before: Listing A → email → Listing B (connected)
- After: Listing A → contact_email, Listing B → billing_email (not connected)
- Fraudsters reuse same email for both contact AND billing
- Separation broke fraud ring detection

### Ablation Test:
Removed all billing edges entirely:
- **Clean Graph** (no billing): 0.5723
- **Full Separated**: 0.5729
- **Difference**: 0.06% (negligible)

**Conclusion**: Billing edges are noise that HGT already ignores. The problem was node separation, not billing edges.

---

## Decision: Stop Fine-Tuning & Revert to Unified Graph (COMPLETED)

**Date**: 2025-11-24  
**Reasoning**:
- Sliding-window results confirmed that deeper/longer HGT training could not beat the baseline (mean AUC-PR stayed < 0.57).
- The only graph features with signal already live on listing nodes; all other node types still carry constant features, so no additional supervision exists for a larger model to exploit.
- Maintaining separate builders/scripts (typed graph, clean-graph ablation, fine-tuned trainer) created operational overhead without improving outcomes.

**Actions Taken**:
- Removed the fine-tuned training script and associated artifacts (`src/models/train_finetuned_hgt.py` and `artifacts/results/hybrid_hgt_finetuned_results.csv` are no longer part of the workflow).
- Deleted the separated-node graph builder and the clean-graph ablation helper to lock the project to the original unified schema.
- Updated documentation to reflect that experiments now focus on understanding *why* GNNs underperform rather than chasing marginal tuning gains.

**Outcome**: From this point on, all experiments should start from the unified graph (`src/data/graph_builder.py`) and either (a) handcraft graph-derived features for the baseline or (b) collect richer bridge signals before reconsidering GNNs.

---

## Experiment 3: Residual Listing Features (COMPLETED)

**Date**: 2025-11-24  
**Goal**: Mitigate over-smoothing so high-signal listing attributes (especially `account_age_days`) survive message passing.  
**Implementation**:
- Updated `UnifiedGNNWrapper` so the final listing embedding concatenates the pre-convolution self representation with the aggregated neighborhood message before projection.
- This acts as an explicit skip connection that preserves raw listing context alongside graph-derived information.

**Results**:
- Regenerated HGT embeddings and retrained the hybrid after enabling the skip connection and joining the new graph-feature columns.
- Hybrid mean AUC-PR improved from the old **0.5909 → 0.6484**, showing the residual preserved tabular signal and let the embeddings contribute slightly when paired with engineered graph stats.
- However, it still trails the graph-feature baseline (0.6655), so manual features remain the best option for now.

**Conclusion**: Residual connections reduce over-smoothing and help the hybrid catch up, but until we find stronger bridge nodes or better embeddings, XGBoost + engineered graph statistics is still the top performer.

---

## Experiment 4: Graph Feature Engineering (COMPLETED)

**Date**: 2025-11-24  
**Goal**: Quantify whether structural graph signals exist by exporting hand-crafted statistics and feeding them into the baseline/hybrid XGBoost models.  
**Implementation**:
- Added `src/features/graph_features.py` plus a new CLI command `python src/cli.py graph-features` to compute:
  - Contact email/phone degrees and “shared contact” counts (degree centrality).
  - User/IP reuse metrics (shared IP hypothesis).
  - Listing-level component sizes across email/phone unions (community size proxy).
  - Listing PageRank scores on the email/phone bipartite graph (importance/centrality).
- `train_baseline.py` and `train_hybrid.py` now auto-join `artifacts/listing_graph_features.parquet` and include any available graph columns in their feature lists.

**Results**:
- `make graph-features` succeeded after adding the compatibility shims for Polars.
- Re-running the baseline with the new columns yielded **0.6655 mean AUC-PR** (vs. 0.5947 previously) and **0.9392 mean AUC-ROC**.
- Early windows (e.g., April–May 2023) now achieve P@100 between 0.49–0.75, and high-volume windows in late 2024–2025 regularly exceed 0.9 precision—evidence that the manual graph statistics surface genuine signal that was invisible to the pure tabular model.

**Interpretation**:
- Graph-derived counts (shared emails/phones/IPs, component sizes, PageRank) *do* improve fraud detection, even though end-to-end GNN embeddings struggled.
- This bridges the gap between “GNN adds no value” and “graph structure is useless”: the structure matters, but it needs to be distilled into targeted features.

**Next Steps**:
1. Optionally feed the same statistics into the hybrid pipeline to test whether embeddings + manual features beat 0.6655.
2. Document the feature definitions and their impact in the evaluation report / thesis chapter.

## Timeline

- [x] **2025-11-24 08:00** - Node Separation Experiment (Failed)
- [x] **2025-11-24 12:00** - Ablation Test (Confirmed billing edges are noise)
- [x] **2025-11-24 13:00** - Decision: Revert to unified graph & remove fine-tuning
- [x] **2025-11-24 13:30** - Residual listing features experiment results
- [x] **2025-11-24 14:00** - Graph feature engineering experiment results

---

## Lessons Learned

1. ✅ **Negative results have value** - Understanding why GNNs fail is publishable
2. ✅ **Test assumptions rigorously** - Ablation tests reveal true causes
3. ✅ **Graph structure matters** - Transitivity >> Node type separation
4. ✅ **Baseline features are strong** - `account_age_days` dominates
5. ⚠️ **GNNs need strong graph signal** - Weak signals = tabular features win

---

## Next Experiments (If Current Fails)

### Fallback Option 1: Manual Feature Engineering
Extract graph-based features without GNN:
- `shared_contact_email_count`
- `contact_email_fraud_rate`
- `phone_reuse_count`
- Feed to XGBoost directly

**Expected**: +5-15% over baseline

### Fallback Option 2: Ensemble
- Train multiple models (HGT, SAGE, GAT)
- Ensemble predictions
- May capture different patterns

**Expected**: +2-5% over best single model

### Fallback Option 3: Accept Baseline
- Document comprehensive analysis
- Focus thesis on "when graph methods work vs don't work"
- Publishable negative result
