# Conclusion and Future Work

## 1. Conclusion
This thesis set out to develop a robust, explainable, and improved fraud detection framework for online real estate marketplaces. By constructing a heterogeneous graph of over 31 million interactions and implementing a hybrid architecture that combines Graph Neural Networks (GNNs) with XGBoost, we have successfully met the research objectives while uncovering critical insights about when and how graph-based methods deliver value.

The key contributions of this work are:

1.  **Conditionally Successful Hybrid Architecture:** We demonstrated that a hybrid model, where a supervised GNN extracts relational embeddings to enrich a Gradient Boosting classifier, achieves a **5.3% improvement in AUC-PR** (0.6990 vs 0.6639) over vanilla XGBoost when trained on 380K+ samples. Additionally, our model outperforms the commercial SEON fraud detection service by **+17.0% AUC-PR**.

2.  **Data Requirements Quantified:** Through expanding window experiments, we discovered that **GNNs require ≥300K training samples** to outperform tabular baselines. With smaller datasets (100K-200K), vanilla XGBoost significantly outperforms the hybrid model (0.6286 vs 0.2187 mean AUC-PR). This is a critical finding for production system design.

3.  **Graph Architecture Insights:** We validated that supervised GNN training with 44-dimensional node features (36 base + 8 temporal encodings) and temporal edge construction (31M edges) enables effective fraud ring detection. GraphSAGE outperforms HGT by +7.4% AUC-PR due to its simpler aggregation mechanism being better suited for large, homogeneous listing-to-listing graphs.

4.  **Operational Viability with Caveats:** In data-rich scenarios, the framework reduces false positives by 18.5% and improves precision from 67.51% to 72.41%. However, for production systems requiring frequent retraining with limited data, vanilla XGBoost remains the superior choice.

5.  **Transparency:** Through the integration of SHAP, we provided interpretable explanations showing that GNN embeddings contribute 17.3% of total feature importance, with browser fingerprints, listing prices, and categories as the strongest individual predictors.

## 2. Limitations

### 2.1 Data Requirements
*   **Sample Size Sensitivity:** The hybrid model requires ≥300K training samples to outperform vanilla XGBoost. This limits applicability for early-stage platforms or niche marketplaces with limited data.
*   **Graph Density Dependency:** Effective GNN learning requires rich connectivity (20M+ edges). Sparse graphs or isolated nodes receive minimal benefit from graph-based features.

### 2.2 Operational Constraints
*   **Computational Cost:** Graph construction and GNN inference add ~5x latency compared to vanilla XGBoost. While acceptable for daily batch processing, real-time inference requires optimized graph serving infrastructure.
*   **Retraining Complexity:** The expanding window results show that frequent retraining (monthly) with limited data favors vanilla XGBoost. GNN benefits diminish when training sets are refreshed before reaching critical mass.

### 2.3 Ethical and Practical Considerations
*   **Geographic Bias:** The model relies heavily on geographic signals (e.g., IP country, ISP). High-risk regions (Benin, Nepal) have 60-90% fraud rates, but this raises fairness concerns for legitimate users from those areas.
*   **Temporal Snapshots:** The current graph uses static temporal windows. A fully dynamic graph (Temporal Graph Networks) could better capture "rapid-fire" coordinated attacks.

## 3. Future Work
To further advance this research, several avenues are recommended:

### 3.1 Addressing Data Requirements
1.  **Few-Shot Graph Learning:** Investigate meta-learning approaches (e.g., MAML) to enable GNNs to generalize from smaller training sets, potentially reducing the 300K sample threshold.
2.  **Transfer Learning:** Pre-train GNN on large external fraud datasets, then fine-tune on real estate data to accelerate learning with limited samples.
3.  **Data Augmentation:** Explore graph augmentation techniques (edge dropping, feature masking) to artificially expand training diversity.

### 3.2 Architectural Improvements
4.  **Dynamic Graph Networks:** Implement Temporal Graph Networks (TGNs) to update node embeddings in real-time as interaction events occur, rather than rebuilding the graph daily. This could improve performance in expanding window scenarios.
5.  **Heterophily-Aware Models:** Investigate CARE-GNN or similar architectures designed for graphs where fraudsters deliberately connect to legitimate users ("camouflage" strategy).

### 3.3 Multi-Modal Integration
6.  **Text & Image Analysis:** The current model ignores unstructured listing content. Integrating NLP (for descriptions) and Computer Vision (for photos) could uncover "content-based" fraud patterns:
    - Stolen or watermarked images
    - Suspiciously templated descriptions
    - Unrealistic amenities or pricing narratives

### 3.4 Operational Enhancements
7.  **Active Learning:** Implement a feedback loop where uncertain predictions (0.4 < prob < 0.6) are routed to fraud analysts, and their labels immediately retrain the model.
8.  **Optimized Graph Serving:** Develop a graph database (e.g., Neo4j, DGL serving) to reduce inference latency and enable real-time predictions.
9.  **Hybrid Deployment Strategy:** Use vanilla XGBoost for weekly retraining and GNN+XGBoost for quarterly "deep dives" on accumulated data to catch sophisticated fraud rings.

### 3.5 Fairness and Explainability
10. **Bias Mitigation:** Implement fairness constraints to reduce geographic bias while maintaining fraud detection accuracy.
11. **Enhanced SHAP Analysis:** Develop graph-specific explanation methods (e.g., GNNExplainer) to visualize why specific connections triggered fraud alerts.
