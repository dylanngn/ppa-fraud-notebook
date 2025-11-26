# Continuous Learning Lifecycle Plan

This document outlines the high-level architecture for the continuous fraud detection pipeline. Instead of a custom Python orchestrator, we recommend using **GitHub Actions** (or a similar CI/CD scheduler) to trigger individual pipeline steps.

## Architecture Overview

The pipeline consists of three main periodic jobs:
1.  **Daily Drift Check**: Monitors data quality and concept drift.
2.  **Weekly Retraining**: Retrains the model on the latest data and promotes it if performance improves.
3.  **Monthly Optimization**: Tunes hyperparameters to adapt to long-term trend changes.

## 1. Daily Drift Check (06:00 UTC)

**Goal**: Detect if the production data distribution has shifted significantly from the training data.

**Workflow**:
1.  **Trigger**: Scheduled cron job (`0 6 * * *`).
2.  **Action**: Run a Python script (e.g., `scripts/check_drift.py`) that:
    *   Loads the last 24 hours of production data.
    *   Compares it with the training baseline (stored in `artifacts/training_stats.json`).
    *   Calculates drift metrics (PSI or Wasserstein distance).
3.  **Alerting**:
    *   If **Critical Drift** (> 3 features drifted): Send High Priority Alert (Slack/PagerDuty).
    *   If **Moderate Drift**: Send Warning (Slack).
    *   If **No Drift**: Log success.

**Implementation Note**:
You will need to implement `scripts/check_drift.py` which uses `src.data` and `src.features` to load and process data, then uses `scipy` or `alibi-detect` to calculate drift.

## 2. Weekly Retraining (Sunday 02:00 UTC)

**Goal**: Keep the model fresh by training on the most recent window of data.

**Workflow**:
1.  **Trigger**: Scheduled cron job (`0 2 * * 0`).
2.  **Action**: Run `make train-mlflow-register` (or `src/cli.py train-mlflow --register-model`).
    *   Trains a new model on the latest `window_days` (e.g., 90 days).
    *   Logs metrics to MLflow.
    *   Registers the model in the MLflow Model Registry.
3.  **Evaluation & Promotion**:
    *   Compare the new model's `AUC-PR` with the current `Production` model.
    *   **Rule**: If `New AUC-PR > Production AUC-PR + 1%`:
        *   Promote New Model to `Staging`.
        *   Run automated integration tests.
        *   (Optional) Auto-promote to `Production` or request manual approval.
    *   **Rule**: If `New AUC-PR <= Production AUC-PR`:
        *   Archive the new model (do not deploy).
        *   Log a warning that retraining did not improve performance.

## 3. Monthly Hyperparameter Optimization (1st Sunday 00:00 UTC)

**Goal**: Re-tune the model architecture (e.g., tree depth, learning rate) to fit the evolving fraud patterns.

**Workflow**:
1.  **Trigger**: Scheduled cron job (`0 0 1-7 * 0` - First Sunday).
2.  **Action**: Run `src/cli.py staged-hyperopt`.
    *   Runs a distributed search (e.g., Optuna) for optimal hyperparameters.
    *   Updates the default configuration file (e.g., `config/model_config.yaml`).
3.  **Follow-up**: Trigger the **Weekly Retraining** job immediately after optimization to train a production model with the new parameters.

## GitHub Actions Example

```yaml
name: Continuous Learning Pipeline

on:
  schedule:
    - cron: '0 6 * * *'  # Daily Drift Check
    - cron: '0 2 * * 0'  # Weekly Retrain

jobs:
  drift-check:
    if: github.event.schedule == '0 6 * * *'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run Drift Check
        run: python scripts/check_drift.py
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  weekly-retrain:
    if: github.event.schedule == '0 2 * * 0'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train and Evaluate
        run: python src/cli.py train-mlflow --register-model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```
