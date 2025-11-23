# Thesis Ablation Study: Validating Model Components

**Objective:** To scientifically demonstrate that every component of the proposed Hybrid HGT+XGBoost architecture is necessary for optimal performance in fraud detection. This moves the research from "it works" to "here is *why* it works."

## Experiment A: The Necessity of Heterogeneity
* **Setup:** Replace the HGT (Heterogeneous Graph Transformer) with a standard GAT (Graph Attention Network) or GCN.
* **Modification:** Treat all node types (User, IP, Listing, Device) as a single generic "Node" type.
* **Hypothesis:** Performance will drop significantly because the model loses the semantic distinction between a "User-Listing" connection (behavior) and a "Listing-IP" connection (technical infrastructure).
* **Thesis Argument:** Validates that modeling the **schema** of the real estate marketplace is just as important as the topology.

## Experiment B: The Necessity of Relative Temporal Encoding (RTE)
* **Setup:** Remove the $\Delta T$ (Time Difference) term from the HGT attention mechanism.
* **Modification:** Use a static graph where edges exist, but the model is blind to *when* they were created relative to the source node.
* **Hypothesis:** The detection rate for "high-velocity" attacks (e.g., bot attacks, rapid account creation + posting) will decrease.
* **Thesis Argument:** Validates that fraud is a **dynamic** phenomenon and that the *velocity* of edges is a predictive signal.

## Experiment C: The Necessity of the Hybrid Architecture (Cold Start)
* **Setup:** Remove the XGBoost component and rely solely on the HGT's final classification head.
* **Modification:** Train and test using only the Graph Neural Network.
* **Hypothesis:** Overall accuracy might remain high (due to established nodes), but performance on **new listings (Degree = 0)** will crash to near-random levels.
* **Thesis Argument:** Validates the **Cascading Hybrid** approach. Pure GNNs fail on isolated nodes; the Hybrid model successfully bridges the gap by falling back on raw XGBoost features.

## Critical Visualization: Performance vs. Node Degree
* **Plot:** X-Axis = Node Degree (0 to High); Y-Axis = F1-Score or AUC.
* **Expected Result:**
    * **Pure HGT Line:** Starts low/near zero at Degree 0, rises with degree.
    * **Hybrid Line:** Starts high (at XGBoost baseline) at Degree 0, remains high.
* **Conclusion:** This chart visually proves the solution to the Cold Start problem.