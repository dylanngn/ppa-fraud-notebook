# Model Experiments Plan

This document details the specific experiments to run to validate the hybrid fraud detection architecture.

## 1. Experiment A: The Tabular Baseline (XGBoost)
**Goal**: Establish a strong baseline using traditional methods. If the GNN cannot beat this, it adds no value.

*   **Model**: XGBoost (Gradient Boosted Decision Trees).
*   **Input Data**: `nodes_listing.parquet` joined with `nodes_user.parquet`.
*   **Feature Engineering**:
    *   **Raw Features**: Price, Living Space, Rooms, Zip Code, Submission Hour, Time from Creation to Submission.
    *   **Text**: Use the *raw* text length, or simple TF-IDF (optional, but maybe keep simple for baseline).
    *   **Manual Aggregates (Crucial)**:
        *   Since this model doesn't see the graph, we *must* manually engineer "graph-like" features.
        *   `user_listing_count`: How many listings has this user posted?
        *   `ip_listing_count`: How many listings from this IP?
        *   `user_fraud_history`: Has this user posted fraud before? (Be careful of data leakage!).
*   **Pros**: Fast, interpretable, industry standard.
*   **Cons**: Cannot easily capture complex "guilt-by-association" (e.g., User A shares IP with User B who is fraud).

## 2. Experiment B: The Hybrid Architecture (HGT + XGBoost)
**Goal**: Combine the best of both worlds. Use HGT for representation learning and XGBoost for tabular classification.

*   **Stage 1: Representation Learning (HGT)**
    *   **Model**: Heterogeneous Graph Transformer (HGT).
    *   **Input**: The full HeteroData graph (`User`, `Listing`, `IP` nodes + edges).
    *   **Training Objective**: Supervised Node Classification on `Listing` nodes (`fraud_flag`).
    *   **Action**: Train the model, then **discard the classification head** and extract the node embeddings from the final hidden layer.
    *   **Output**: A dense vector embedding (e.g., 64-dim) for each listing.

*   **Stage 2: Classification (XGBoost)**
    *   **Input**: [Original Tabular Features] + [HGT Embeddings].
    *   **Model**: XGBoost.
    *   **Hypothesis**: The HGT embeddings will act as powerful "relational features" that summarize the graph neighborhood (e.g., "connected to a known fraudster"), allowing XGBoost to make better decisions than with tabular data alone.

## 4. Evaluation Strategy

*   **Split**: Time-based split (Train on past, Test on future) to simulate production.
    *   *Train*: Listings created before Date X.
    *   *Test*: Listings created after Date X.
*   **Metrics**:
    *   **AUC-PR (Average Precision)**: Primary metric. Focuses on the minority class (Fraud).
    *   **Recall @ Precision k**: "What % of fraud do we catch if we want 95% precision?" (Important for reducing false positives).
    *   **AUC-ROC**: Secondary metric.

## 5. Handling Concept Drift (Monthly Split & Inductive Learning)

To avoid "Time Travel" and simulate real-world drift, we will split the dataset by **Month**:

1.  **Temporal Split**:
    *   Data is partitioned into $M_1, M_2, ..., M_N$ (e.g., Nov '23, Dec '23...).
    *   **Training**: Train on $M_1...M_k$.
    *   **Testing**: Test on $M_{k+1}$.
    *   **Retrain**: Slide window forward.

2.  **Handling New Users (Inductive GNN)**:
    *   **Challenge**: Users are created at the time of their first listing. A user in $M_{k+1}$ might be completely new (unseen in training).
    *   **Solution**: We must use an **Inductive GNN** (HGT).
        *   **No ID Embeddings**: We cannot use `torch.nn.Embedding(num_users)` because the vocab size changes.
        *   **Feature-Based Initialization**: We initialize User nodes using a linear projection of their *features* (`email_domain`, `account_age`).
        *   This allows the model to generate embeddings for *any* user, new or old, based on their attributes and graph connections.

## 6. Implementation Roadmap

1.  **`train_baseline.py`**: Implement Experiment A.
2.  **`train_gnn.py`**: Implement Experiment B using PyTorch Geometric.
3.  **`train_hybrid.py`**: Implement Experiment C (Load GNN, extract embeddings, train XGB).
