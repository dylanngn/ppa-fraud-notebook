# Fraud Model Investigation: GNN Performance

## ✅ RESOLVED (2025-12-19)

The GNN underperformance issue has been **resolved**. GNN+XGBoost now **outperforms** vanilla XGBoost by +5.3% AUC-PR.

### Key Changes Made
1. **Supervised GNN training**: Switched from unsupervised link prediction to supervised node classification with fraud labels
2. **Expanded graph**: All identity columns now enabled (device, IP, user, email, phone) with ~31M edges
3. **Richer node features**: 44 input features (36 base + 8 temporal encodings)
4. **Optimized hyperparameters**: hidden_dim=32, num_layers=2, epochs=10 (shallow and efficient)

### Results After Fix

| Model | AUC-PR | Change |
|-------|--------|--------|
| Vanilla XGBoost | 0.6639 | Baseline |
| **GNN+XGBoost** | **0.6990** | **+5.3%** |

---

## Historical Analysis: Why Unsupervised GNN Failed

### Summary (Original Issue)
- GNN embeddings added little value because they were trained without fraud supervision, used sparse edges, and relied on low-signal node features.
- Data joins and filtering shrunk the usable graph; HPO ignored key graph/supervision knobs.

### Evidence (code-linked)
- **Unsupervised objective**: GNN trained with link prediction only; fraud labels never influenced embeddings, so they captured connectivity rather than fraud signal. [src/models/graphsage.py#L140:L158](src/models/graphsage.py#L140)
- **Sparse identity edges**: Only `ip_hash` and `session/similarity_hash` formed temporal edges; entities with <2 occurrences were dropped. [src/features/graph_builder.py#L44:L125](src/features/graph_builder.py#L44)
- **Low-signal node features**: Node inputs were just three proxy booleans plus first six listing numerics. [src/features/schema.py#L356:L365](src/features/schema.py#L356)

### Recommendations Applied
1. ✅ **Supervise the GNN**: Implemented node classification loss with fraud labels
2. ✅ **Strengthen graph signal**: Enabled all identity columns (~31M edges)
3. ✅ **Richer node features**: Expanded to 44 features (36 base + 8 temporal)
4. ✅ **Optimized HPO**: Found best params (hidden=32, layers=2, epochs=10)
