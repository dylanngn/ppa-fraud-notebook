# Fraud Detection in Online Real Estate Marketplaces: Utilizing a Hybrid Graph and Gradient Boosting Model

**Author:** Nguyen Hoang Minh

This repository contains the source code, data preprocessing pipelines, and experimental framework for my master's thesis research on fraud detection in real estate marketplaces. The research investigates graph-based fraud pattern discovery using Swiss Marketplace Group (SMG) real estate data (804,717 listing events).

## Overview

Online marketplace fraud causes significant financial losses and erodes user trust. This project builds a **heterogeneous graph** to encode entity-sharing relationships (device fingerprints, IP addresses, emails, phones) and applies **Graph Neural Network (GNN)** extraction combined with **XGBoost classification**. 

While GNN architectures reveal unique fraud ring structures invisible to standard tabular analysis, we found that a purely tabular Vanilla XGBoost model trained on raw SEON integration signals natively presents superior operational trade-offs for production deployment due to the temporal distribution shift in real-world environments.

## Documentation

The project documentation is organized to align with the research evaluation and the system's operational architecture:

- 📊 **[Data Overview & Discovery](docs/data_overview.md)**: Details the dataset characteristics, including temporal analysis, geographic distributions (e.g., high-risk IP origins), device connectivity, and the distribution of fraud flags.
- 🏗️ **[System Architecture](docs/architecture.md)**: Describes the ETL pipeline (built on Polars), temporal graph construction, and the models implemented to train both Vanilla XGBoost and the Hybrid GNN+XGBoost classifiers.
- 📓 **[Experiment Journal](docs/experiment_journal.md)**: A comprehensive record of model evaluations, benchmark comparisons, concept drift analysis across expanding temporal windows, and SHAP explainability.
- 💾 **[Data Dictionary & Schema](docs/data_dictionary.md)** (*Pending*): Defines the expected schemas for inserting new event and SEON transaction data to retrain the pipeline.
- 🖥️ **[Demo Plan Proposal](docs/demo_plan.md)** (*Pending*): An outline demonstrating how the model detects fraud, handles real-time submissions, detects drift, and performs SHAP context evaluation.

## Research Contributions

1. **Heterogeneous Graph Pattern Discovery:** Constructed a dynamic, temporal graph capturing over 31 million edges connecting fraudulent entities across shared device networks, emails, and IPs.
2. **Concept Drift Mitigation:** Evaluated model robustness over expanding temporal windows, proving that frequent retraining is crucial over raw complex topological features in production pipelines.
3. **Interpretability Integration:** Incorporated SHAP globally and locally, attributing 82.7% of critical predictive signals natively to targeted tabular signals extracted through SEON.

## Citing This Work

If you find this code or research helpful, please cite the associated thesis:

```bibtex
@mastersthesis{minh2024frauddetection,
  author       = {Nguyen Hoang Minh},
  title        = {Fraud Detection in Online Real Estate Marketplaces: Utilizing a Hybrid Graph and Gradient Boosting Model},
  school       = {FPT School of Business and Technology},
  year         = {2024},
  month        = {December}
}
```

## Setup & Local Usage

Instructions for environment setup, data preparation, and training execution:

```bash
# Install dependencies
pip install -r requirements.txt

# Execute model training (Vanilla XGBoost by default, configured via Hydra)
python -m src.training.trainer
```
