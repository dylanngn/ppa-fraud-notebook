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

## 5. Handling Concept Drift (Sliding Window Backtesting)

To **simulate** the production retraining loop and measure robustness to drift, we will not just do a single split. Instead, we will perform **Sliding Window Backtesting**:

1.  **Methodology**:
    *   Define a **Training Window** (e.g., 3 months).
    *   Define a **Test Window** (e.g., 1 week immediately following training).
    *   Define a **Step Size** (e.g., move forward by 1 week).
2.  **Process**:
    *   **Fold 1**: Train on `[Jan-Mar]`, Test on `[Apr Week 1]`.
    *   **Fold 2**: Train on `[Jan Week 2 - Apr Week 1]`, Test on `[Apr Week 2]`.
    *   ...and so on.
3.  **Goal**: This measures how the model performs *if we were to retrain it weekly*. If performance degrades over time despite retraining, it indicates fundamental shifts in fraud patterns that the model architecture cannot capture.

## 6. Implementation Roadmap

1.  **`train_baseline.py`**: Implement Experiment A.
2.  **`train_gnn.py`**: Implement Experiment B using PyTorch Geometric.
3.  **`train_hybrid.py`**: Implement Experiment C (Load GNN, extract embeddings, train XGB).
