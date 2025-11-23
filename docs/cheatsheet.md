# Research Cheat Sheet: Hybrid HGT + XGBoost for Real Estate Fraud

## 1. The Graph Structure (The "Why")
* **Heterogeneity:** The graph must model different entities to capture semantics.
    * **Nodes:** `User`, `Listing`, `IP`, `Device`, `Phone`.
* **The Trap (Bridge Nodes):** Use technical nodes (IP, Device, Description Hash) to connect seemingly unrelated users. This exposes "Fraud Rings."
* **Goal:** Turn hidden attributes into traversable structural paths.

## 2. The Engine: HGT (The "How")
* **Relative Temporal Encoding (RTE):**
    * Embeds the time gap $\Delta T$ directly into the attention mechanism.
    * **Benefit:** Allows the model to distinguish between normal behavior (years of history) and suspicious behavior (microseconds between creation and posting).

## 3. Data Engineering (The Safety)
* **Inductive Splitting:** Train on past data, validate on future data.
* **Snapshotting Strategy:**
    * Construct the graph at time $T$ using **only** edges that existed at $t < T$.
    * **Crucial:** Prevents "Time Travel" (Data Leakage), ensuring the model doesn't see future "Reported" flags during training.

## 4. The Cold Start Solution (The Architecture)
* **Method:** Cascading / Sequential Integration.
    1.  **HGT:** Acts as a "Semantic Feature Extractor."
    2.  **Extraction:** Pull **node embeddings** from the *penultimate layer* (not the final score).
    3.  **XGBoost:** Takes `[Raw Features + Structural Embeddings]` as input.
* **Logic:**
    * **Isolated Node:** HGT embedding is generic; XGBoost relies on Raw Features (Price, Text).
    * **Connected Node:** HGT embedding is rich; XGBoost leverages the deep structural context.

## 5. Evaluation (The Proof)
* **Metrics:**
    * **AUC-PR:** Global performance.
    * **Lift (Precision @ K):** Operational efficiency (how many fraudsters in the top 100 reviewed?).
* **Loss Function:** **Focal Loss**.
    * Penalizes the model heavily for missing the minority class (fraud), preventing the "99% accuracy on empty data" trap.