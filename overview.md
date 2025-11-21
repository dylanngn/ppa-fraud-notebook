# Research Overview: Hybrid GNN-XGBoost Fraud Detection

This document outlines the technical options for implementing the hybrid GNN-XGBoost fraud detection system, based on the provided research proposal and database schema.

## 1. ETL Options (Polars)

**Goal**: Efficiently transform raw data from the Aurora PostgreSQL database into a structured heterogeneous graph dataset suitable for PyTorch Geometric (PyG).

**Tools**:
- **Polars**: For high-performance data manipulation (Rust-based).
- **ConnectorX**: For fastest data extraction from Postgres to Polars DataFrames.
- **Sentence-Transformers**: For generating text embeddings from listing descriptions.

### Pipeline Steps

```mermaid
graph TD
    subgraph "Aurora PostgreSQL (Source)"
        Users[hginsertionapiprod.users]
        Insertions[hginsertionapiprod.insertions]
        Status[hginsertionapiprod.status_history]
    end

    subgraph "Extraction (ConnectorX)"
        ExtUsers[Extract Users]
        ExtIns[Extract Insertions]
        ExtStatus[Extract Status History]
    end

    subgraph "Processing (Polars)"
        JoinStatus[Join Status History<br/>(Filter: DRAFT -> PENDING)]
        FilterDate[Filter Date<br/>(2023-11-01 to 2025-11-01)]
        ParseJSON[Parse Listing JSONB]
        FeatEng[Feature Engineering<br/>(Fraud Logic, Lister Info)]
        Embed[Generate Text Embeddings<br/>(Sentence-Transformers)]
    end

    subgraph "Graph Construction"
        NodesUser[Create User Nodes]
        NodesListing[Create Listing Nodes]
        NodesIP[Create IP Nodes]
        EdgesPost[Create User-Post-Listing Edges]
        EdgesUse[Create User-Uses-IP Edges]
    end

    subgraph "Output (Processed Parquet)"
        OutUser[nodes_user.parquet]
        OutListing[nodes_listing.parquet]
        OutIP[nodes_ip.parquet]
        OutEdge1[edges_user_posts.parquet]
        OutEdge2[edges_user_uses_ip.parquet]
    end

    subgraph "Graph Construction (graph_builder.py)"
        PyG[Construct HeteroData]
        GraphPT[Save graph.pt]
    end

    Users --> ExtUsers
    Insertions --> ExtIns
    Status --> ExtStatus

    ExtIns --> JoinStatus
    ExtStatus --> JoinStatus
    JoinStatus --> FilterDate
    FilterDate --> ParseJSON
    ExtUsers --> NodesUser

    ParseJSON --> FeatEng
    FeatEng --> Embed
    Embed --> NodesListing
    FeatEng --> NodesIP
    
    FeatEng --> EdgesPost
    FeatEng --> EdgesUse

    NodesUser --> OutUser
    NodesListing --> OutListing
    NodesIP --> OutIP
    EdgesPost --> OutEdge1
    EdgesUse --> OutEdge2

    OutUser --> PyG
    OutListing --> PyG
    OutIP --> PyG
    OutEdge1 --> PyG
    OutEdge2 --> PyG
    PyG --> GraphPT
```

1.  **Data Extraction & Raw Checkpoint**:
    -   Extract data from DB.
    -   **Save Raw**: `artifacts/raw_users.parquet`, `artifacts/raw_insertions.parquet` (Preserves raw state).

2.  **Data Processing & Feature Engineering (`ETL.py`)**:
    -   Load Raw Parquet (or DB).
    -   **JSON Parsing**: Extract fields from `listing` JSONB.
    -   **Node/Edge Creation**: Create Polars DataFrames for all nodes and edges.
    -   **Save Processed**: `artifacts/nodes_*.parquet`, `artifacts/edges_*.parquet`.

3.  **Graph Construction (`graph_builder.py`)**:
    -   **Input**: Processed Parquet files.
    -   **Action**:
        -   Map string IDs (e.g., "listing_123") to integer indices (0, 1, 2...).
        -   Convert Polars Columns to PyTorch Tensors.
        -   Create `torch_geometric.data.HeteroData` object.
    -   **Output**: `artifacts/graph.pt` (The ready-to-train graph object).

## 2. The Hybrid Architecture (HGT + XGBoost)

We have selected a **Hybrid Architecture** that combines the structural learning capabilities of a Graph Neural Network (GNN) with the tabular performance of XGBoost.

### Stage 1: Representation Learning (HGT)
**Goal**: Learn a dense vector embedding for each listing that captures its "graph context" (e.g., "connected to a suspicious IP").

*   **Model**: **Heterogeneous Graph Transformer (HGT)**.
    *   *Why*: State-of-the-art for heterogeneous graphs. Uses attention to learn which relations (e.g., `User->Posts->Listing` vs `User->Uses->IP`) are most important.
*   **Input**: The full HeteroData graph.
*   **Training Objective**:
    *   **Supervised**: Train on `fraud_flag` (Binary Classification).
    *   **Action**: After training, discard the final classification head and extract the **node embeddings** (e.g., 64-dim vector) from the last hidden layer.
*   **Output**: A feature vector $Z_L$ for every Listing $L$.

### Stage 2: Classification (XGBoost)
**Goal**: Make the final fraud prediction using all available information.

*   **Model**: **XGBoost**.
*   **Input**: Concatenation of:
    1.  **Tabular Features**: Price, Size, Rooms, Zip, Description Length.
    2.  **Graph Embeddings ($Z_L$)**: The 64-dim vector from Stage 1.
*   **Why**: XGBoost is superior at handling mixed tabular data and simple rules (e.g., "Price > 1M"), while the embedding provides the complex relational signal.

## 3. Experiment Plan

1.  **Baseline (XGBoost)**: Train on tabular data only (with manual aggregates).
2.  **Hybrid (HGT + XGBoost)**:
    *   Train HGT.
    *   Extract Embeddings.
    *   Train XGBoost on Tabular + Embeddings.
3.  **Evaluation**: Compare AUC-PR on a sliding time window (to handle concept drift).
