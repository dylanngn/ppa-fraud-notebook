# Conclusion and Future Work

## 1. Conclusion
This thesis set out to develop a robust, explainable, and improved fraud detection framework for online real estate marketplaces. By constructing a heterogeneous graph of over 30 million interactions and implementing a hybrid architecture that combines Graph Neural Networks (GNNs) with XGBoost, we have successfully met the research objectives.

The key contributions of this work are:
1.  **Successful Hybrid Architecture:** We demonstrated that a hybrid model, where a supervised GNN extracts relational embeddings to enrich a Gradient Boosting classifier, achieves a **5.3% improvement in AUC-PR** over a strong industry-standard baseline.
2.  **Graph Signal Validation:** We refuted the simplistic assumption that "more connections = more fraud" in this domain. Instead, we found that fraudsters create distinct *types* of network structures (e.g., high-density, short-lived clusters) compared to the broad, long-term connectivity of legitimate power users. The supervised GNN was essential in learning these nuanced distinctions.
3.  **Operational Viability:** The framework is not just accurate but operationally efficient, reducing false positives by 18.5% and offering 100% precision for the top-100 riskiest listings. This directly translates to reduced workload for fraud analysts.
4.  **Transparency:** Through the integration of SHAP, the complex "black box" of neural/ensemble learning was opened, providing interpretable rationale for every decision.

## 2. Limitations
*   **Computational Cost:** Graph construction and GNN inference add latency to the pipeline compared to a simple tabular model. While feasible for batch processing, real-time inference would require optimized graph serving infrastructure.
*   **Geographic Bias:** The model relies heavily on geographic signals (e.g., high-risk countries). While effective, this raises ethical considerations regarding fairness and potential bias against legitimate users from those regions.
*   **Temporal Edges:** The current graph uses a static snapshot window. A fully dynamic graph (Discrete-Time Dynamic Graph) model could potentially capture "rapid-fire" attacks more effectively.

## 3. Future Work
To further advance this research, several avenues are recommended:
1.  **Dynamic Graph Networks:** Implementing Temporal Graph Networks (TGNs) to update node embeddings in real-time as interaction events occur, rather than rebuilding the graph daily.
2.  **Text & Image Analysis:** The current model ignores the unstructured content of listings. Integrating NLP (for descriptions) and Computer Vision (for listing photos) into the multi-modal GNN could uncover "content-based" fraud patterns (e.g., stolen or watermarked images).
3.  **Active Learning:** Implementing a feedback loop where uncertain predictions are routed to analysts, and their labels are immediately used to update the model, creating a more responsive adaptive system.
4.  **Expanding Window GNN Evaluation:** While we validated the concept drift robustness of the *Vanilla* model, the **longitudinal stability of the Hybrid GNN model** remains to be fully backtested. Running the `expanding-gnn` experiment is a critical next step to ensure the graph signals remain robust over time.
