# A Technical Blueprint for Implementing a Hybrid Graph-Based Fraud Detection System for Online Real Estate Marketplaces

## Section 1: Architecting the Data Foundation: From Tabular Records to a Heterogeneous Marketplace Graph

The foundational step in developing a sophisticated, graph-based fraud detection system is the transformation of raw, tabular marketplace data into a structured, high-signal heterogeneous graph. The success of the proposed hybrid model is contingent upon the expressiveness and accuracy of this graph representation. This section provides a formal specification for the data schema and the graph model, outlining the necessary entities, relationships, and features required to capture the subtle, networked patterns of fraudulent activity.

### 1.1 Defining the Canonical Data Schema and Input Tables

Before constructing the graph, it is essential to define the expected structure of the raw input data. The system ingests data from the core **Aurora PostgreSQL** database, specifically focusing on the insertion funnel.

**Required Input Tables:**

*   **`hginsertionapiprod.insertions`**: This table serves as the central repository for all information specific to an individual real estate listing (insertion). It contains the target variable for the classification task and critical metadata.
    *   **Schema**: `id` (Primary Key), `user_id` (Foreign Key), `object_reference` (Business ID), `listing` (JSONB content), `user_ip_address`, `fraud_flag` (Timestamp), `created_at`, `first_published_date`.
*   **`hginsertionapiprod.users`**: This table contains all information related to user accounts.
    *   **Schema**: `id` (Primary Key), `created_at`, `contact_emails`, `user_type`, `platform`.
*   **`hginsertionapiprod.status_history`**: This table tracks the lifecycle of an insertion. It is used to identify the precise moment a listing was submitted for approval (`DRAFT` -> `PENDING_APPROVAL`), which defines the temporal scope of the dataset.

**Note on "Slip-Through" Fraud**:
A critical target for this research is "slip-through" fraud—listings that initially passed automated checks (like Seon) but were later flagged as fraud. This is identified where `fraud_flag > first_published_date`.

### 1.2 The Heterogeneous Graph Model: A Formal Specification (RQ1)

The core innovation is the transition from a flat, tabular view to a rich, multi-relational graph structure. We define a heterogeneous graph $G = (V, E)$ with multiple types of nodes and edges.

**Node Types and Their Attributes:**

1.  **Listing (Target Node)**: The primary entity to classify.
    *   **Features**: `price` (buy/rent), `living_space`, `rooms`, `zip_code`, `city`, `description_embedding` (Dense vector from Sentence-Transformer).
2.  **User**: The actor creating listings.
    *   **Features**: `account_age`, `user_type`, `email_domain`.
3.  **IP Address**: The network address used to submit the listing.
    *   **Features**: (Structural only initially, e.g., degree centrality).
4.  **Email**: Extracted from user profiles, listing contacts, and billing info.
    *   **Features**: (Structural).
5.  **Phone**: Extracted from listing contacts and billing info.
    *   **Features**: (Structural).
6.  **Location**: A composite node representing a specific `Zip_City` pair.
    *   **Features**: (Structural).

**Edge Types (Relational Schema):**

The edges represent shared identifiers and interactions, creating a "guilt by association" network.

*   **(User) - [posts] -> (Listing)**: User created the listing.
*   **(User) - [uses] -> (IP)**: User submitted a listing from this IP.
*   **(User) - [has] -> (Email)**: User account is registered with this email.
*   **(Listing) - [has] -> (Email)**: Listing contact/billing uses this email.
*   **(Listing) - [has] -> (Phone)**: Listing contact/billing uses this phone number.
*   **(Listing) - [located_at] -> (Location)**: Listing is physically located in this Zip/City.
*   **(User) - [located_at] -> (Location)**: User is associated with this location (via billing/lister address).

**Formal Schema Summary:**

| Node Type | ID Example | Key Features |
| :--- | :--- | :--- |
| **Listing** | `insertion_123` | `price`, `size`, `rooms`, `text_embedding` |
| **User** | `user_ABC` | `account_age`, `user_type`, `email_domain` |
| **IP** | `192.168.1.1` | *Structural* |
| **Email** | `foo@bar.com` | *Structural* |
| **Phone** | `+4179...` | *Structural* |
| **Location** | `8001_Zurich` | *Structural* |

| Edge Type | Source -> Target | Description |
| :--- | :--- | :--- |
| `posts` | User -> Listing | Authorship |
| `uses` | User -> IP | Network Trace |
| `has_email` | User/Listing -> Email | Shared Contact Info |
| `has_phone` | Listing -> Phone | Shared Contact Info |
| `located_at` | User/Listing -> Location | Geographic Clustering |

### 1.3 Feature Engineering for a High-Signal Graph

*   **Text Processing**: The unstructured text in a listing's description is converted into a fixed-size vector embedding using a pre-trained language model (e.g., `all-MiniLM-L6-v2`). This captures semantic content.
*   **Graph Construction**: By explicitly modeling shared entities like **Emails**, **Phones**, and **IPs** as nodes, we allow the GNN to automatically learn "shared resource" patterns (e.g., one phone number used across 50 different listings by different users) without manual feature engineering.

---

## Section 2: Advanced Modeling and Rigorous Benchmarking

### 2.1 The Hybrid GNN-XGBoost Architecture (RQ2)

The proposed architecture is a two-stage hybrid model:

**Stage 1: GNN for Relational Feature Engineering**
*   **Model**: **Heterogeneous Graph Transformer (HGT)**.
*   **Role**: The GNN takes the heterogeneous graph as input and learns a low-dimensional vector representation (embedding) for each **Listing** node. This embedding aggregates information from the listing's neighborhood (e.g., "connected to a high-risk IP" or "shares a phone number with a known fraudster"). We select HGT over alternatives like R-GCN because its transformer-based attention mechanism is better suited to capture complex, non-linear dependencies across different node types and relations in a fraud network.

**Stage 2: XGBoost for High-Performance Classification**
*   **Model**: **XGBoost**.
*   **Input**: A concatenated feature vector:
    1.  Original Tabular Features (Price, Size, etc.)
    2.  GNN-generated Relational Embedding (from Stage 1)
*   **Role**: XGBoost performs the final classification, leveraging its superior handling of tabular data and decision boundaries while benefiting from the rich relational signals provided by the GNN.

### 2.2 Evaluation Strategy

*   **Primary Metric**: **AUC-PR (Area Under the Precision-Recall Curve)**. Given the extreme class imbalance (<1% fraud), this is the only reliable metric.
*   **Validation Scheme**: **Sliding Window Backtesting**. To simulate production and account for concept drift, models will be trained on a past window (e.g., 3 months) and evaluated on a future window (e.g., the following week), rolling forward over time.

---

## Section 3: The End-to-End System: Architecture and Technology Stack

### 3.1 System Architecture Overview

1.  **Data Processing (ETL)**:
    *   **Extract**: `ConnectorX` pulls data from Aurora Postgres.
    *   **Transform**: `Polars` performs high-performance data manipulation and JSON parsing.
    *   **Load**: Data is saved as Parquet artifacts (`nodes_*.parquet`, `edges_*.parquet`).
2.  **Graph Construction**:
    *   `PyTorch Geometric` (`HeteroData`) constructs the in-memory graph object from Parquet files.
3.  **Training Pipeline**:
    *   **Stage 1**: Train GNN (HGT) on the graph. Extract embeddings.
    *   **Stage 2**: Train XGBoost on features + embeddings.
4.  **Deployment**:
    *   **API**: `FastAPI` serves the model.
    *   **Container**: `Docker` packages the environment.

### 3.2 Technology Stack Selection

| Component | Recommended Technology | Justification |
| :--- | :--- | :--- |
| **Data Processing** | **Polars** | Rust-based, multi-threaded, significantly faster than Pandas for large datasets. |
| **Graph Library** | **PyTorch Geometric (PyG)** | Industry standard, pythonic, strong support for heterogeneous graphs (`HeteroData`). |
| **Classifier** | **XGBoost** | State-of-the-art for tabular data, robust, interpretable. |
| **Embeddings** | **Sentence-Transformers** | Efficient, high-quality text embeddings (`all-MiniLM-L6-v2`). |
| **API** | **FastAPI** | High performance, async support, auto-documentation. |

---

## Section 4: Accelerating Development with Open-Source Frameworks

To accelerate implementation, we leverage:
*   **PyTorch Geometric**: For all graph neural network components.
*   **XGBoost**: For the gradient boosting classifier.
*   **SHAP**: For explaining the XGBoost model predictions.
*   **GNNExplainer**: For explaining the GNN embeddings (future work).

## Section 5: Feasibility Analysis

**Performance Strategy: Polars (Rust-backed Python)**
We have opted for a pure Python development environment powered by **Polars** for data processing. Polars is built on Rust and Arrow, providing performance comparable to low-level languages while maintaining the ease of use and ecosystem integration of Python. This strategic choice eliminates the need for a separate Rust codebase or complex FFI integrations, allowing the project to focus on its core machine learning objectives without sacrificing data pipeline throughput. This ensures we can efficiently process millions of listings and interactions within a unified Python environment.

---

## Works Cited

1.  **Fraud Detection Dataset** - Kaggle, accessed October 25, 2025, [Link](https://www.kaggle.com/datasets/goyaladi/fraud-detection-dataset)
2.  **Research on Financial Fraud Detection Models Integrating Multiple Relational Graphs**, accessed October 25, 2025, [Link](https://www.mdpi.com/2079-8954/11/11/539)
3.  **Fraud Detection Graph Database** - TigerGraph, accessed October 25, 2025, [Link](https://www.tigergraph.com/glossary/fraud-detection-with-graph/)
4.  **NguyenHoangMinh_ResearchProposal**
5.  **Building a Fraud Detection Model using Graph Neural Networks**, accessed October 25, 2025, [Link](https://natashagluons.medium.com/building-a-fraud-detection-model-using-graph-neural-networks-gnns-d3c62b7c38e9)
6.  **Research on Fraud Detection Method Based on Heterogeneous Graph Representation Learning** - MDPI, accessed October 25, 2025, [Link](https://www.mdpi.com/2079-9292/12/14/3070)
7.  **Detect financial transaction fraud using a Graph Neural Network with Amazon SageMaker**, accessed October 25, 2025, [Link](https://aws.amazon.com/blogs/machine-learning/detect-financial-transaction-fraud-using-a-graph-neural-network-with-amazon-sagemaker/)
8.  **Real-Time Transaction Fraud Detection via Heterogeneous Temporal Graph Neural Network** - SciTePress, accessed October 25, 2025, [Link](https://www.scitepress.org/Papers/2025/131613/131613.pdf)
9.  **Graph-Based Deep Learning for E-Commerce Fraud Detection** - ResearchGate, accessed October 25, 2025, [Link](https://www.researchgate.net/publication/390130910_Graph-Based_Deep_Learning_for_E-Commerce_Fraud_Detection)
10. **Optimizing Fraud Detection in Financial Services with Graph Neural Networks and NVIDIA GPUs**, accessed October 25, 2025, [Link](https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/)
11. **intel/credit-card-fraud-detection** - GitHub, accessed October 25, 2025, [Link](https://github.com/intel/credit-card-fraud-detection)
12. **Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters**, accessed October 25, 2025, [Link](https://penghao-bdsc.github.io/papers/cikm20.pdf)
13. **Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum** - AAAI Publications, accessed October 25, 2025, [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28773/29483)
14. **Towards Causal Classification: A Comprehensive Study on Graph Neural Networks** - arXiv, accessed October 25, 2025, [Link](https://arxiv.org/html/2401.15444v1)
15. **Comprehensive Guide to GNN, GAT, and GCN** | by Joyce Birkins | Medium, accessed October 25, 2025, [Link](https://medium.com/@joycebirkins/comprehensive-guide-to-gnn-gat-and-gcn-a-beginners-introduction-to-graph-neural-networks-after-51d09ac043b5)
16. **Fraud Detection Using Graph Neural Networks** - GitHub, accessed October 25, 2025, [Link](https://github.com/bimbomuri/Fraud_Detection_GNN)
17. **Detecting Credit Card Fraud via Heterogeneous Graph Neural ...**, accessed October 25, 2025, [Link](https://arxiv.org/abs/2504.08183)
18. **Guidance for Near Real-Time Fraud Detection with Graph Neural Network on AWS**, accessed October 25, 2025, [Link](https://aws.amazon.com/solutions/guidance/near-real-time-fraud-detection-with-graph-neural-network-on-aws/)
19. **Build a GNN-based real-time fraud detection solution using Amazon SageMaker...**, accessed October 25, 2025, [Link](https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-amazon-sagemaker-amazon-neptune-and-the-deep-graph-library/)
20. **awslabs/realtime-fraud-detection-with-gnn-on-dgl** - GitHub, accessed October 25, 2025, [Link](https://github.com/awslabs/realtime-fraud-detection-with-gnn-on-dgl)
21. **Serving ML Models with FastAPI** - Grigor Khachatryan, accessed October 25, 2025, [Link](https://grigorkh.medium.com/serving-ml-models-with-fastapi-a-production-ready-api-in-minutes-b5f4839a33a9)
22. **Create a PyTorch Docker image ready for production** - Riccardo Padovani, accessed October 25, 2025, [Link](https://rpadovani.com/pytorch-docker-image)
23. **shap/shap** - GitHub, accessed October 25, 2025, [Link](https://github.com/shap/shap)
24. **GNNShap: Scalable and Accurate GNN Explanation using Shapley Values** - NSF, accessed October 25, 2025, [Link](https://par.nsf.gov/biblio/10510625-gnnshap-scalable-accurate-gnn-explanation-using-shapley-values)
25. **Go vs Python vs Rust: Which One Should You Learn in 2025?** - PullFlow, accessed October 25, 2025, [Link](https://pullflow.com/blog/go-vs-python-vs-rust-complete-performance-comparison)

