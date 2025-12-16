# **Research Proposal**

## Fraud Detection in Online Real Estate Marketplaces

## Utilizing a Hybrid Graph and Gradient Boosting Model

**Author:** 

Nguyen Hoang Minh

**Student ID:** 

24MSE23152

**Degree Program:** 

Master of Software Engineering

**Supervising Professor:** 

Professor Doan Xuan Huy Minh, Ph.D.

**Institution:** 

FPT School of Business and Technology

**Academic Year:** 

2024 \- 2025

# **Abstract**

This project details a predictive framework for the automated detection of fraudulent listings in online real estate marketplaces. The application of advanced machine learning for fraud detection on real estate platforms remains a comparatively under-researched domain, especially when contrasted with the extensive body of work on financial and transactional fraud. This project leverages a real-world dataset from the Swiss Marketplace Group (SMG) to develop and validate a solution for this industry-wide challenge. It addresses the limitations of current heuristic-based systems, which are ill-equipped to handle the evolving, networked tactics of fraudsters—a phenomenon known as concept drift. This work proposes, develops, and validates a novel hybrid architecture that combines Graph Neural Networks (GNNs) for relational feature engineering with an XGBoost classifier for high-performance prediction. The framework is designed to be automated, adaptive through a robust retraining strategy, and transparent through the integration of Explainable AI (XAI) techniques.

## **1\. Introduction**

### **1.1 Background and Motivation**

The foundation of any online marketplace is trust. For real estate platforms, this trust is paramount but is continually threatened by fraudulent listings that can lead to significant financial loss and erode user confidence. This project is motivated by the need for intelligent, automated systems to protect users and preserve platform integrity. While fraud detection is a mature field, most research has focused on other domains. The specific characteristics of real estate data—such as unstructured text, image data, and complex, often subtle, relationships between users and listings—present a unique set of challenges \[1\].

This project utilizes a dataset provided by the Swiss Marketplace Group (SMG) and is driven by the operational need to move beyond static, rule-based fraud detection systems. Such systems struggle to adapt to the dynamic and adversarial nature of fraud, where malicious actors continuously evolve their strategies to evade capture (a process known as concept drift) \[2\]. The goal is to develop a data-driven framework that not only identifies individual fraudulent listings with high accuracy but also uncovers the underlying relational patterns indicative of coordinated fraud rings.

### **1.2 Research Problem**

While commercial black-box solutions for fraud detection exist, there is a notable lack of publicly documented and academically validated frameworks specifically tailored to the unique relational data of online real estate marketplaces. This project's academic contribution lies not in competing with commercial products, but in developing and evaluating a transparent, custom-built framework. The significance comes from applying a novel Graph Neural Network (GNN) architecture to this specific domain and integrating Explainable AI (XAI) to create a "glass-box" model, providing insights that proprietary systems often do not. There is a clear need for an open, validated framework that can:

1. Automatically learn complex, non-linear patterns from diverse data sources.  
2. Effectively model the relational "guilt by association" data inherent in collusive fraud.  
3. Maintain high performance in the face of evolving fraud tactics (concept drift).  
4. Provide transparent, actionable insights for operational teams.

## **2\. Research Objectives and Questions**

The primary objective of this project  is to design, implement, and evaluate an end-to-end predictive framework for fraud detection in the SMG real estate marketplace.

#### **2.1. Key Objectives**

1. **Develop a Hybrid Predictive Model:** To design and implement a hybrid machine learning model that leverages a Graph Neural Network (GNN) to learn relational features from user-listing interaction data and an XGBoost model for the final classification task.  
2. **Mitigate Concept Drift:** To implement and evaluate a passive retraining strategy (periodic model updates) to ensure the framework remains effective over time as fraud patterns evolve.  
3. **Generate Actionable Insights:** To integrate Explainable AI (XAI) techniques (e.g., SHAP, LIME) to make the model's predictions transparent and generate human-understandable insights into the key drivers of fraud.  
4. **Validate the Framework:** To rigorously evaluate the framework's performance using historical data from SMG, focusing on both predictive accuracy and its robustness to temporal data shifts.

#### **2.2. Research Questions**

* **RQ1:** What novel, relational indicators of fraud can be identified in a real-estate marketplace dataset using graph-based analysis?  
* **RQ2:** How can a hybrid Graph Neural Network and XGBoost architecture be designed to effectively model these indicators for automated fraud detection?  
* **RQ3:** To what extent does a periodic retraining strategy, measured by AUC-PR and F1-Score, maintain the model's predictive performance against concept drift over sequential time windows?  
* **RQ4:** How can Explainable AI (XAI) techniques translate the hybrid model's predictions into actionable, human-understandable insights for fraud analysts?

## **3\. Project Scope**

#### **3.1. In Scope**

This project will deliver the following key components:

* Development of a data pipeline for processing and transforming SMG's raw data.  
* Construction of a marketplace graph and implementation of a GNN for feature engineering.  
* Training and hyperparameter tuning of an XGBoost classification model.  
* Implementation and evaluation of a passive, periodic retraining pipeline.  
* Application of SHAP for global model explanation and LIME for local, instance-level explanations.  
* Deployment of the final model as a RESTful API service for proof-of-concept validation.  
* Benchmarking the final model against multiple baseline models (e.g., Logistic Regression, Random Forest, vanilla GNN).  
* Application of SHAP and LIME to generate global and local explanations for model predictions.

#### **3.2. Out of Scope**

* Analysis of image or video data within the listings..  
* The implementation of active (real-time) concept drift detection mechanisms.  
* A large-scale user acceptance testing (UAT) or A/B testing in a live production environment.

## **4\. Methodology and Proposed Framework**

The proposed methodology is a multi-stage process designed to create a robust and explainable fraud detection system.

### **4.1 Stage 1: Graph-Based Feature Engineering**

The core innovation of this framework is its treatment of the marketplace not as a simple tabular dataset, but as a heterogeneous graph. Entities such as Users, Listings, and Devices will be represented as nodes. Relationships, such as a USER-posts-LISTING or a USER-uses-DEVICE, will be represented as edges. This graph structure is essential for uncovering fraud rings and collusive behaviors that are invisible to traditional models \[3\]. A Graph Neural Network (GNN) will be trained on this graph to learn a low-dimensional vector representation (embedding) for each listing. This embedding serves as an automatically generated, highly predictive feature that distills complex relational information.

### **4.2 Stage 2: Hybrid Predictive Classification**

The GNN-generated embeddings will be appended to the original set of tabular features for each listing (e.g., price, account age, description length). This enriched feature set will then be used as input to train a high-performance XGBoost model \[4\]. This hybrid approach uses the GNN for what it does best—automated relational feature engineering—and XGBoost for what it is renowned for—achieving state-of-the-art performance and high efficiency on tabular classification tasks.

### **4.3 Stage 3: Mitigating Concept Drift via Periodic Retraining**

To address the dynamic nature of fraud, the framework will incorporate a passive concept drift adaptation strategy based on periodic retraining. This involves automatically retraining the entire GNN-XGBoost pipeline at regular intervals (e.g., weekly or monthly) using the most recent data. This MLOps practice ensures the model continuously adapts to new fraud patterns and user behaviors \[2\], \[5\]. The effectiveness of this strategy will be evaluated through a rigorous longitudinal backtest.

### **4.4 Stage 4: Explainable AI (XAI) for Insight Generation**

To ensure the framework is not a "black box," two XAI techniques will be integrated:

* SHAP (SHapley Additive exPlanations): To provide a *global* understanding of the model's behavior by quantifying the overall importance of each feature (including the GNN embeddings) in driving fraud predictions \[6\].  
* LIME (Local Interpretable Model-agnostic Explanations): To generate *local*, case-specific explanations that show why an individual listing was flagged as fraudulent. This is critical for building trust with SMG's fraud analysts and enabling them to validate the model's alerts \[7\].

## **5\. Evaluation Plan**

The framework's performance will be assessed through a multi-faceted evaluation plan.

1. **Predictive Performance:** The primary evaluation metric will be the Area Under the Precision-Recall Curve (AUC-PR), which is well-suited for imbalanced datasets common in fraud detection. This provides a holistic view of the model's performance across different decision thresholds. We will also report on Precision, Recall, and F1-Score at an operationally relevant threshold determined in consultation with SMG.  
2. **Baseline Comparison:** The performance of the hybrid GNN-XGBoost model will be compared against a strong set of baseline models:  
* **Traditional Models:** Logistic Regression and Random Forest.  
* **Ablation Baselines:** An XGBoost model trained only on tabular features (without GNN embeddings) and a GNN-only model (using embeddings for direct classification).  
3. **Concept Drift Robustness (Longitudinal Backtest):** To validate the retraining strategy, we will simulate the model's operational lifecycle. The model will be trained on an initial time window of data (e.g., Month 1), tested on the next window (Month 2), then retrained with data from Months 1-2 and tested on Month 3, and so on. This simulation allows for a rigorous evaluation of the model's adaptability to potential concept drift—such as gradual changes or sudden 'fraud waves'—using historical data, without needing to observe these events in real-time.   
4. **Qualitative Evaluation of XAI:** To assess the practical utility of the XAI component (addressing RQ4), a qualitative evaluation will be conducted. A sample of SHAP (global) and LIME (local) explanations will be presented to an expert audience (e.g., the supervising professor). Feedback will be collected on the clarity, actionability, and trustworthiness of the generated insights.  
5. **System Performance:** The deployed Prediction Service API will be evaluated on its technical efficiency using key system metrics, including:  
* **Latency:** The average response time per prediction request.  
* **Throughput:** The number of requests the service can handle per second.

## 

## **6\. Expected Outcomes and Contributions**

* **Primary Contribution (Final Project Report):** The final written report will contribute a novel, hybrid GNN-XGBoost framework for fraud detection, providing a detailed case study of its application and evaluation in the under-researched domain of online real estate marketplaces.  
* **Secondary Contribution (Software Artifact):** A functional, validated, and containerized software framework (including the data pipeline, trained model, and prediction API) that serves as a robust proof-of-concept for SMG.  
* **Tertiary Contribution (Actionable Intelligence):** A set of data-driven, explainable insights into the primary characteristics and relational patterns of fraudulent listings on the SMG platform, delivered through XAI-generated reports.

## **7\. Resources Required**

* Software: Python, PyTorch Geometric (for GNNs), Polars, Scikit-learn, XGBoost, FastAPI, Docker.  
* Data: The primary dataset for this case study will be provided by Swiss Marketplace Group (SMG) under a data usage and confidentiality agreement.  
* Hardware: A modern development computer. Cloud resources (e.g., for GPU-based GNN training) may be required depending on data scale.

## 

# **Timeline**

| Weeks | Key Activities | Deliverables |
| :---: | ----- | ----- |
| 1-2 | Literature Review & Finalize Methodology | Completed Literature Review, Finalized Proposal |
| 3-4 | Data Exploration, Preprocessing, and Pipeline Implementation | Cleaned Dataset, Data Pipeline Code |
| 5-6 | Graph Construction & GNN Model Implementation | GNN Model Code & Learned Embeddings |
| 7-8 | Hybrid XGBoost Model Training, Tuning, and Baseline Comparison | Trained Hybrid Model, Baseline Model |
| 9-10 | Implementation and Evaluation of Retraining Strategy (Longitudinal Backtest) | Backtest Results & Performance Analysis |
| 11 | XAI Implementation and Insight Generation (SHAP/LIME) | Explanations, Feature Importance Plotsspacing |
| 12 | API Deployment and Final Report Write-up | Deployed API, Final Project Report |

## 

# **References**

\[1\] J. Chen et al., “Data-driven Anomaly Detection for Responsible and Trusted Home Valuation,” in *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2020, pp. 3131–3139.

\[2\] B. Krawczyk, L. L. Minku, J. Gama, J. Stefanowski, and M. Woźniak, "Ensemble learning for data stream analysis: A survey," *Information Fusion*, vol. 37, pp. 132-156, 2017\.

\[3\] P. R. Madduru and N. Janvekar, “A heterogeneous graph-based framework for scalable fraud detection,” in *Proceedings of the 2nd International Conference on AI-ML Systems*, 2023, pp. 1–4.

\[4\] Y. Chen, W. Y. Lin, and Z. W. Hong, “A fraud detection system for e-commerce merchant based on Xgboost,” in *2020 IEEE International Conference on Consumer Electronics-Taiwan (ICCE-TW)*, 2020, pp. 1–2.

\[5\] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A survey on concept drift adaptation," *ACM Computing Surveys*, vol. 46, no. 4, pp. 1-37, 2014\.

\[6\] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *Advances in Neural Information Processing Systems 30 (NIPS 2017\)*, 2017, pp. 4765–4774.

\[7\] M. T. Ribeiro, S. Singh, and C. Guestrin, “"Why Should I Trust You?": Explaining the Predictions of Any Classifier,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 1135–1144.