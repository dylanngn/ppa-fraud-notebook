# Model Architectures

This document provides detailed specifications for all models in the fraud detection system.

## 1. Baseline: XGBoost

### Overview
A gradient boosting decision tree model that serves as the baseline for comparison.

### Input
**Features** (13 dimensions):
- `account_age_days`: Float (Days since account creation)
- `log_price`: Float (Log-transformed price)
- `living_space`: Float (Square meters)
- `rooms`: Float (Number of rooms)
- `is_new`: Binary (New construction flag)
- `has_balcony`: Binary
- `has_elevator`: Binary
- `has_parking`: Binary
- `bundle_period`: Integer (Days)
- `bundle_tier_score`: Ordinal (0=basic, 1=premium, 2=top)
- `is_direct_payment`: Binary
- `is_buy`: Binary (vs. rent)
- `latitude`, `longitude`: Float (Location)

### Architecture
```
XGBoost Classifier
├── n_estimators: 100
├── max_depth: 6
├── learning_rate: 0.1
├── objective: binary:logistic
├── eval_metric: aucpr
└── scale_pos_weight: Auto (class imbalance ratio)
```

### Output
- **Probabilities**: Float [0, 1] representing fraud likelihood

### Training Strategy
- **Method**: Sliding Window Backtesting
- **Window**: 90 days training, 14 days testing
- **Step**: 14 days
- **Loss**: LogLoss (Binary Cross-Entropy)

---

## 2. Graph Attention Network (GAT)

### Overview
Uses attention mechanisms to learn importance weights for neighbors in the graph.

### Input
**Node Features** (per node type):
- `User`: 32-dim (email domain, phone, account features)
- `Listing`: 128-dim (price, size, location, description embedding)
- `IP`: 32-dim (hashed IP address)
- `Email`: 16-dim (domain embedding)
- `Phone`: 16-dim (country code, hashed)
- `Location`: 8-dim (lat/long + geo hash)

**Graph Structure**:
- Heterogeneous edges with timestamps
- Edge types: `posts`, `uses`, `located_at`

### Architecture
```
UnifiedGNNWrapper (GAT)
├── Input Projection Layer (per node type)
│   └── Linear: [input_dim] → 64
├── GAT Core (converted via to_hetero)
│   ├── Layer 1: GATConv(64, 16, heads=4) → 64
│   │   ├── Attention mechanism per edge type
│   │   └── 4 attention heads
│   ├── Layer 2: GATConv(64, 16, heads=4) → 64
│   └── ReLU activations
├── Output Projection
│   └── Linear: 64 → 64 (embeddings)
└── Classification Head (training only)
    └── Linear: 64 → 1
```

### Output
- **Embeddings**: 64-dim vector per listing
- **Predictions**: Float [0, 1] (during training)

### Training Details
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross-Entropy with Logits
- **Epochs**: 20 (default)
- **Device**: MPS/CUDA if available, else CPU

---

## 3. Graph Convolutional Network (GCN)

### Overview
Uses spectral graph convolutions for efficient neighborhood aggregation.

### Input
Same as GAT (heterogeneous node features + graph structure)

### Architecture
```
UnifiedGNNWrapper (GCN)
├── Input Projection Layer (per node type)
│   └── Linear: [input_dim] → 64
├── GCN Core (converted via to_hetero)
│   ├── Layer 1: GCNConv(64, 64) → 64
│   │   └── Normalized adjacency aggregation
│   ├── Layer 2: GCNConv(64, 64) → 64
│   └── ReLU activations
├── Output Projection
│   └── Linear: 64 → 64 (embeddings)
└── Classification Head
    └── Linear: 64 → 1
```

### Output
- **Embeddings**: 64-dim vector per listing
- **Predictions**: Float [0, 1]

### Training Details
Same as GAT

### Key Differences from GAT
- No attention mechanism (simpler, faster)
- Uses normalized adjacency matrix for aggregation
- Better for homophilic graphs (similar nodes connect)

---

## 4. Heterogeneous Graph Transformer (HGT)

### Overview
Designed specifically for heterogeneous graphs with type-aware attention.

### Input
Same as GAT/GCN

### Architecture
```
UnifiedGNNWrapper (HGT)
├── Input Projection Layer (per node type)
│   └── Linear: [input_dim] → 64
├── HGT Core
│   ├── Layer 1: HGTConv(64, 64, num_heads=4)
│   │   ├── Type-specific attention weights
│   │   ├── Separate parameters per edge type
│   │   └── Message passing per relation
│   ├── Layer 2: HGTConv(64, 64, num_heads=4)
│   └── No activation between layers (HGT handles internally)
├── Output Projection
│   └── Linear: 64 → 64 (embeddings)
└── Classification Head
    └── Linear: 64 → 1
```

### Output
- **Embeddings**: 64-dim vector per listing
- **Predictions**: Float [0, 1]

### Training Details
Same as GAT

### Key Advantages
- **Type-aware**: Separate parameters for each node/edge type
- **Relation-specific**: Learns different transformations per edge type
- **Designed for heterogeneity**: No need for `to_hetero` conversion

---

## 5. HGT with Relative Temporal Encoding (HGT+RTE)

### Overview
Extends HGT with time-awareness to capture temporal patterns in fraud behavior.

### Input
Same as HGT, **plus**:
- **Edge Timestamps**: Int64 (nanoseconds since epoch)

### Architecture
```
HGTWrapperWithRTE
├── Input Projection Layer (per node type)
│   └── Linear: [input_dim] → 64
├── Time Encoding Layer
│   └── Sinusoidal Encoding: Δt → 64-dim
│       ├── period = 24h, 7d, 30d, 365d
│       └── Captures multi-scale time patterns
├── HGT Core with RTE
│   ├── Layer 1: HGTConv(64, 64, heads=4)
│   │   ├── Attention = f(node_i, node_j, edge_type, Δt_encoding)
│   │   └── Time-modulated attention weights
│   ├── Layer 2: HGTConv(64, 64, heads=4)
│   └── Temporal message passing
├── Output Projection
│   └── Linear: 64 → 64
└── Classification Head
    └── Linear: 64 → 1
```

### Time Encoding Formula
```
Δt = timestamp_target - timestamp_edge
RTE(Δt) = [sin(Δt / period_1), cos(Δt / period_1), ..., sin(Δt / period_k), cos(Δt / period_k)]
```

### Output
- **Embeddings**: 64-dim temporally-aware vector
- **Predictions**: Float [0, 1]

### Key Advantages
- **Velocity Detection**: Can identify "rapid-fire" fraud attacks
- **Time-decay**: Recent connections weighted differently than old ones
- **Multi-scale**: Captures hourly, daily, weekly, and monthly patterns

---

## 6. Hybrid Models (GNN + XGBoost)

### Overview
Combines GNN embeddings with tabular features for final classification.

### Input
**Tabular Features** (13-dim): Same as Baseline XGBoost  
**GNN Embeddings** (64-dim): From any GNN variant (GAT/GCN/HGT/HGT+RTE)

**Total**: 77-dim feature vector

### Architecture
```
Hybrid Model
├── Stage 1: GNN (GAT/GCN/HGT/HGT+RTE)
│   └── Output: 64-dim embeddings per listing
├── Stage 2: Feature Concatenation
│   └── [Tabular (13) + Embeddings (64)] → 77-dim
└── Stage 3: XGBoost Classifier
    ├── n_estimators: 100
    ├── max_depth: 6
    ├── learning_rate: 0.1
    └── Input: 77-dim features
```

### Output
- **Probabilities**: Float [0, 1]

### Training Strategy
1. **Phase 1**: Train GNN on historical data (80% split)
2. **Phase 2**: Extract embeddings for all listings
3. **Phase 3**: Train XGBoost on Tabular + Embeddings (90-day sliding window)

### Variants
- `hybrid_gat_results.csv`
- `hybrid_gcn_results.csv`
- `hybrid_hgt_results.csv`
- `hybrid_hgt_rte_results.csv`

---

## Comparison Summary

| Model | Parameters | Training Time | Best For |
|-------|-----------|---------------|----------|
| **Baseline XGBoost** | ~10K | Fast (~1 min) | Tabular-only baseline |
| **GAT** | ~50K | Medium (~10 min) | Attention-based learning |
| **GCN** | ~40K | Fast (~5 min) | Simple spectral aggregation |
| **HGT** | ~60K | Medium (~12 min) | Heterogeneous graphs |
| **HGT+RTE** | ~70K | Slow (~15 min) | Temporal fraud patterns |
| **Hybrid (Any)** | GNN + 10K | GNN + Fast | Best overall performance |

## Hyperparameter Tuning Suggestions

### GNN Models
- **hidden_channels**: {32, 64, 128} - Trade-off between capacity and overfitting
- **num_layers**: {1, 2, 3} - Deeper networks capture longer-range connections
- **num_heads**: {2, 4, 8} - More heads = more diverse attention patterns
- **learning_rate**: {0.0001, 0.001, 0.01} - Lower for stability, higher for speed

### XGBoost
- **max_depth**: {4, 6, 8} - Deeper trees = more complex rules
- **n_estimators**: {50, 100, 200} - More trees = better fit (but slower)
- **learning_rate**: {0.05, 0.1, 0.2} - Lower = more stable convergence

### Temporal Windows
- **Training Window**: {30d, 60d, 90d} - Longer = more data, but concept drift
- **Test Window**: {7d, 14d, 30d} - Matches operational cadence
- **Step Size**: {7d, 14d} - Smaller = more frequent updates
