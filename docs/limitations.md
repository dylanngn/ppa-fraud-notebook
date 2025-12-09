# Limitations & Threats to Validity

This document describes the limitations of our fraud detection study and potential threats to validity.

---

## Model Limitations

### 1. Cold Start Problem

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **New Users** | Users with 0 prior listings have fewer graph features | Account age is top SHAP feature; new accounts get enhanced scrutiny |
| **New Listings** | First listing has no historical patterns | Rely on tabular features (price, location, payment type) |
| **Graph Isolation** | Isolated nodes have no connectivity signals | `listing_pagerank=0` and `listing_component_size=1` serve as isolation indicators |

### 2. Novel Fraud Patterns

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Unseen Patterns** | Sophisticated fraudsters may develop new tactics | Weekly retraining incorporates new patterns |
| **Concept Drift** | Fraud patterns evolve over time | Accumulating window training (validated in Experiment 4) |
| **Adversarial Adaptation** | Fraudsters may learn to evade detection | Multi-feature redundancy; no single feature dominates |

### 3. Label Quality

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Label Delay** | Fraud labels may arrive weeks after submission | Accumulating windows use confirmed labels only |
| **Undetected Fraud** | Some fraud may never be labeled | Conservative fraud rate estimates; focus on precision |
| **Label Noise** | Some legitimate listings may be mislabeled | High P@100 (87%) reduces false positive impact |

---

## Data Limitations

### 1. Single Platform

| Limitation | Description |
|------------|-------------|
| **Dataset Source** | Data from one Swiss real estate marketplace |
| **Generalizability** | Results may not transfer to other platforms/countries |
| **Recommendation** | Validate on additional platforms before broader deployment |

### 2. Temporal Scope

| Limitation | Description |
|------------|-------------|
| **Time Range** | ~2 years of data (2023-2025) |
| **Seasonal Effects** | May not capture long-term seasonal patterns |
| **Economic Conditions** | Results specific to current market conditions |

### 3. Feature Coverage

| Limitation | Description |
|------------|-------------|
| **292 Raw Fields** | Only ~50 fields used as features |
| **Missing Signals** | Image content, external data not included |
| **Text Features** | Limited text analysis (caps ratio, word length only) |

---

## Methodological Limitations

### 1. Evaluation Approach

| Aspect | Limitation | Justification |
|--------|------------|---------------|
| **Temporal Split** | No k-fold cross-validation | Respects temporal ordering; prevents data leakage |
| **Single Dataset** | Cannot compare across datasets | Real-world production data; not synthetic |
| **Baseline** | Seon is proprietary; limited documentation | Best available production baseline |

### 2. Model Selection

| Aspect | Limitation | Justification |
|--------|------------|---------------|
| **XGBoost Only** | Did not test neural networks, LightGBM, CatBoost | XGBoost standard for tabular fraud detection |
| **GNN Comparison** | Only SAGE and HGT tested | Most common GNN architectures for heterogeneous graphs |
| **Hyperopt Scope** | 100 trials may not be exhaustive | Feature engineering provided more lift than hyperopt |

### 3. Graph Construction

| Aspect | Limitation | Justification |
|--------|------------|---------------|
| **8 Edge Types** | May miss other relationships | Coverage-based selection; <10% coverage edges excluded |
| **Static Graph** | Graph updated weekly, not real-time | Balances accuracy with computational cost |

---

## Threats to Validity

### Internal Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Data Leakage** | Features computed from future data | Strict temporal cutoffs; accumulating windows |
| **Selection Bias** | Only labeled fraud used for training | Conservative estimates; focus on precision |
| **Implementation Bugs** | Code errors affecting results | MLflow tracking; reproducible experiments |

### External Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Platform Specificity** | Results may not generalize | Document Swiss real estate context |
| **Temporal Specificity** | Fraud patterns may change | Weekly retraining; drift monitoring (Experiment 8) |
| **User Population** | Swiss market may differ from others | Acknowledge in conclusions |

### Construct Validity

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Metric Choice** | AUC-PR may not capture all aspects | Also report P@100, AUC-ROC, precision/recall |
| **Fraud Definition** | "Fraud" may be inconsistently labeled | Use platform's official fraud labels |
| **Baseline Fairness** | Seon comparison may be unfair | Document Seon's operating point (high recall, low precision) |

---

## What the Model Cannot Do

1. **Detect completely novel fraud types** not present in training data
2. **Predict fraud in real-time** (batch processing with weekly retraining)
3. **Explain fraud to end users** (SHAP explanations are for analysts)
4. **Guarantee zero false positives** (87% precision means 13% false positives)
5. **Operate without retraining** (concept drift causes severe degradation)

---

## Recommendations for Production

| Recommendation | Rationale |
|----------------|-----------|
| **Weekly Retraining** | Mitigates concept drift (validated -1.56%/month degradation without) |
| **Human Review** | High-confidence predictions still need analyst verification |
| **Monitoring** | Deploy Evidently AI for drift detection |
| **Fallback Plan** | Revert to Seon if model fails |
| **Quarterly Audits** | Use Feature Discovery Pipeline for new patterns |

---

## References

- Experiment 4: Concept Drift Evaluation (RQ3)
- Experiment 7: Production Readiness Validation
- Experiment 8: Feature Evolution & Monitoring

