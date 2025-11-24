# Model Experiments Plan (Updated)

This document details the specific experiments to run to validate the hybrid fraud detection architecture.

## 1. Experiment A: The Tabular Baseline (XGBoost)
**Goal**: Establish a strong baseline using traditional methods. If the GNN cannot beat this, it adds no value.

*   **Model**: XGBoost (Gradient Boosted Decision Trees).
*   **Input Data**: `nodes_listing.parquet` joined with `nodes_user.parquet`.
*   **Features**:
    *   **Tabular**: Price, Living Space, Rooms, Zip Code.
    *   **Booleans**: New, Balcony, Elevator, Parking.
    *   **Categorical**: Bundle Tier, Payment Type, Offer Type.
    *   **Location**: Latitude, Longitude.
    *   **Engineered**: `account_age_days`.
*   **Pros**: Fast, interpretable, industry standard.
*   **Cons**: Cannot easily capture complex "guilt-by-association".

## 2. Experiment B: The Hybrid Architecture (GNN + XGBoost)
**Goal**: Combine the best of both worlds. Use a GNN for representation learning and XGBoost for tabular classification.

### Workflow
1.  **Stage 1: Representation Learning (GNN)**
    *   Train a GNN on the heterogeneous graph to classify fraud.
    *   **Output**: Extract dense vector embeddings (64-dim) for each listing.
2.  **Stage 2: Classification (XGBoost)**
    *   **Input**: [Original Tabular Features] + [GNN Embeddings].
    *   **Model**: XGBoost.
    *   **Hypothesis**: The embeddings summarize the graph neighborhood (e.g., "connected to a known fraudster"), acting as powerful new features.

### Variants (Ablation Study)
We will test 4 different GNN architectures to find the best embedding generator:

1.  **GAT (Graph Attention Network)**:
    *   Uses attention mechanisms to weigh neighbors.
    *   Adapted for heterogeneous graphs via `to_hetero`.
2.  **GraphSAGE (Graph Sample and Aggregate)**:
    *   Inductive learning via neighborhood sampling.
    *   Natively supports bipartite/heterogeneous graphs.
3.  **HGT (Heterogeneous Graph Transformer)**:
    *   Designed specifically for heterogeneous graphs.
    *   Uses type-specific attention.
4.  **HGT + RTE (Relative Temporal Encoding)**:
    *   Adds time-awareness to HGT.
    *   Encodes the time difference between nodes and edges.

## 3. Thesis Experiments (The "Why")

To scientifically demonstrate the value of each component, we define three specific experiments:

### Experiment A: The Necessity of Heterogeneity
*   **Comparison**: **HGT** vs. **GAT/GraphSAGE**.
*   **Hypothesis**: HGT should outperform because it explicitly models the schema (User vs. Listing vs. IP), whereas GAT/GraphSAGE (even with `to_hetero`) treats connections more generically.

### Experiment B: The Necessity of Relative Temporal Encoding (RTE)
*   **Comparison**: **HGT** vs. **HGT + RTE**.
*   **Hypothesis**: RTE should detect "high-velocity" attacks (rapid posting) better than standard HGT, as it captures the *time difference* between edges.

### Experiment C: The Necessity of the Hybrid Architecture (Cold Start)
*   **Comparison**: **Pure GNN** vs. **Hybrid (GNN + XGBoost)**.
*   **Setup**: Evaluate the GNN's direct predictions against the Hybrid model's predictions.
*   **Hypothesis**:
    *   **Pure GNN**: Will fail on new listings (Degree 0) because they have no graph connections yet.
    *   **Hybrid**: Will remain robust on new listings by falling back on tabular features (Price, Location, etc.).
*   **Visualization**: Plot **Performance (AUC) vs. Node Degree**. We expect the Hybrid line to stay high at Degree 0, while the GNN line drops.

## 4. Evaluation Strategy

*   **Method**: Sliding Window Backtesting.
*   **Configuration**:
    *   **Training Window**: 90 Days (Ensures sufficient data density).
    *   **Test Window**: 14 Days (Simulates bi-weekly model updates).
    *   **Step Size**: 14 Days.
*   **Metrics**:
    *   **AUC-PR (Average Precision)**: Primary metric (focus on minority class).
    *   **Precision@100**: Operational metric (how many of the top 100 flagged are actually fraud?).
    *   **AUC-ROC**: Secondary metric.

## 4. Implementation Roadmap (Completed)

1.  **`src/models/gnn_variants.py`**: Implemented GAT, GCN, HGT, HGT-RTE.
2.  **`src/models/train_embeddings.py`**: Unified script to train any GNN and save embeddings.
3.  **`src/models/train_hybrid.py`**: Unified script to train XGBoost on embeddings + tabular features.
4.  **`notebooks/03_model_comparison.ipynb`**: Notebook to visualize and compare results.

## 5. Running Experiments

Use the `Makefile` to run the full pipeline for each variant:

```bash
make exp-gat      # Run GAT experiment
make exp-sage     # Run GraphSAGE experiment
make exp-hgt      # Run HGT experiment
make exp-hgt-rte  # Run HGT+RTE experiment
```

Then open `notebooks/03_model_comparison.ipynb` to see the results.
