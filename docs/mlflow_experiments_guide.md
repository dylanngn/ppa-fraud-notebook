# MLflow Experiments: When to Create New vs Reuse

## What is an MLflow Experiment?

An **experiment** is a container that groups related runs together. Think of it as a folder for organizing your ML work:

```
MLflow
├── Experiment: "ppa-fraud-detection"  ← Container
│   ├── Run: baseline_20241127_1430    ← Individual training run
│   ├── Run: hybrid_sage_20241127_1500
│   ├── Run: hyperopt_xgboost_20241127
│   └── Run: window_optimization_90d
│
├── Experiment: "new-feature-testing"  ← Different experiment
│   ├── Run: test_new_graph_features
│   └── Run: test_time_weighted_features
```

### Hierarchy

```
Experiment (ppa-fraud-detection)
  └── Run (baseline_accumulating_20241127_1430)
      └── Nested Run (window_0)
      └── Nested Run (window_1)
      └── Nested Run (window_2)
```

---

## Current Setup: Single Experiment

Your current codebase uses **one experiment** for everything:

```python
mlflow.set_experiment("ppa-fraud-detection")
```

This groups:
- Baseline model training
- Hybrid SAGE model training
- Hybrid HGT model training
- Hyperparameter optimization
- Window size optimization
- Seon baseline evaluation

**This is fine for now**, but let's understand when you might want to split this.

---

## When to Create a NEW Experiment

### ✅ Create New Experiment For:

#### 1. **Major Research Direction Change**
When you're exploring a fundamentally different approach:

```python
# Current: Graph-based fraud detection
mlflow.set_experiment("ppa-fraud-detection")

# New: Testing deep learning approach
mlflow.set_experiment("ppa-fraud-detection-deep-learning")

# New: Testing ensemble methods
mlflow.set_experiment("ppa-fraud-detection-ensembles")
```

**Example scenarios:**
- Switching from XGBoost to neural networks
- Testing completely different feature engineering approaches
- Exploring different problem formulations (e.g., multi-class vs binary)

#### 2. **Different Data Sources or Datasets**
When working with different data:

```python
# Current: Production data
mlflow.set_experiment("ppa-fraud-detection")

# New: Synthetic data testing
mlflow.set_experiment("ppa-fraud-detection-synthetic")

# New: Different time period
mlflow.set_experiment("ppa-fraud-detection-2024-Q1")
```

#### 3. **Different Problem Formulations**
When the target or objective changes:

```python
# Current: Binary fraud classification
mlflow.set_experiment("ppa-fraud-detection")

# New: Fraud severity prediction (regression)
mlflow.set_experiment("ppa-fraud-severity-prediction")

# New: Multi-class (fraud type classification)
mlflow.set_experiment("ppa-fraud-type-classification")
```

#### 4. **Major Feature Engineering Changes**
When testing fundamentally different feature sets:

```python
# Current: Graph + tabular features
mlflow.set_experiment("ppa-fraud-detection")

# New: Testing embeddings-only approach
mlflow.set_experiment("ppa-fraud-detection-embeddings-only")

# New: Testing without graph features
mlflow.set_experiment("ppa-fraud-detection-no-graph")
```

#### 5. **Different Business Contexts**
When the use case or requirements change:

```python
# Current: General fraud detection
mlflow.set_experiment("ppa-fraud-detection")

# New: High-priority fraud only
mlflow.set_experiment("ppa-high-priority-fraud")

# New: Real-time vs batch processing
mlflow.set_experiment("ppa-fraud-detection-realtime")
```

---

## When to REUSE Existing Experiment

### ✅ Use Same Experiment For:

#### 1. **Regular Training Cycles** (Your Current Use Case)
When retraining models with the same approach:

```python
# Week 1: Baseline training
mlflow.set_experiment("ppa-fraud-detection")
# Creates run: baseline_accumulating_20241127_1430

# Week 2: Retraining with new data
mlflow.set_experiment("ppa-fraud-detection")  # Same experiment!
# Creates run: baseline_accumulating_20241204_1430
```

**Why?** All runs are comparable - same model type, same features, just different data/time.

#### 2. **Hyperparameter Tuning**
When optimizing parameters for the same model:

```python
mlflow.set_experiment("ppa-fraud-detection")
# All hyperopt runs go here - easy to compare
```

#### 3. **Model Variants** (Your Current Use Case)
When testing different models on the same problem:

```python
mlflow.set_experiment("ppa-fraud-detection")
# baseline, hybrid_sage, hybrid_hgt all in same experiment
# Easy to compare which model performs best
```

#### 4. **Feature Ablation Studies**
When testing which features matter:

```python
mlflow.set_experiment("ppa-fraud-detection")
# Run 1: All features
# Run 2: Without graph features
# Run 3: Without time-weighted features
# Easy to compare impact of each feature set
```

#### 5. **Window Size Optimization**
When testing different time windows:

```python
mlflow.set_experiment("ppa-fraud-detection")
# All window optimization runs together
```

---

## Best Practices

### 1. **Use Tags for Organization Within Experiment**

Instead of creating new experiments, use tags to organize:

```python
mlflow.set_experiment("ppa-fraud-detection")

with mlflow.start_run(tags={
    "model_type": "baseline",
    "training_mode": "accumulating_window",
    "experiment": "exp13",  # Your experiment journal number
    "feature_set": "graph+tabular"
}):
    # Training code
```

Then filter in MLflow UI by tags!

### 2. **Use Run Names for Clarity**

```python
mlflow.set_experiment("ppa-fraud-detection")

# Clear, descriptive run names
mlflow.start_run(run_name="baseline_accumulating_20241127_1430")
mlflow.start_run(run_name="hybrid_sage_with_new_features_20241127")
mlflow.start_run(run_name="hyperopt_baseline_100trials")
```

### 3. **Nested Runs for Related Work**

```python
mlflow.set_experiment("ppa-fraud-detection")

# Parent run for the entire training session
with mlflow.start_run(run_name="baseline_training_session"):
    # Nested runs for each window
    for window_idx in range(num_windows):
        with mlflow.start_run(run_name=f"window_{window_idx}", nested=True):
            # Train and evaluate this window
```

---

## Recommended Structure for Your Project

### Option 1: Keep Single Experiment (Current - Good for Now)

```
ppa-fraud-detection
├── baseline_accumulating_20241127
├── hybrid_sage_20241127
├── hybrid_hgt_20241127
├── hyperopt_xgboost_20241127
└── window_optimization_90d
```

**Pros:**
- Simple, easy to compare all models
- Good for single research project
- Easy to find best model

**Cons:**
- Can get cluttered with many runs
- Hard to separate different research directions

### Option 2: Split by Model Type (When You Have Many Runs)

```
ppa-fraud-detection-baseline
├── baseline_20241127
├── baseline_20241204
└── baseline_20241211

ppa-fraud-detection-hybrid
├── hybrid_sage_20241127
├── hybrid_hgt_20241127
└── hybrid_sage_v2_20241204

ppa-fraud-detection-optimization
├── hyperopt_xgboost
└── window_optimization
```

**Use when:** You have 50+ runs and need better organization.

### Option 3: Split by Research Phase (Recommended for Long-Term)

```
ppa-fraud-detection-v1-baseline
├── Initial baseline models
└── Hyperparameter optimization

ppa-fraud-detection-v2-graph-features
├── Models with graph features
└── Graph feature ablation studies

ppa-fraud-detection-v3-hybrid
├── Hybrid GNN+XGBoost models
└── Embedding experiments

ppa-fraud-detection-production
├── Production model retraining runs
└── Model registry candidates
```

**Use when:** You have distinct research phases or versions.

---

## Practical Decision Tree

```
Are you testing a fundamentally different approach?
├─ YES → Create new experiment
│   └─ Examples: Different model type, different data, different problem
│
└─ NO → Use existing experiment
    ├─ Regular retraining? → Same experiment
    ├─ Hyperparameter tuning? → Same experiment
    ├─ Testing model variants? → Same experiment (use tags)
    └─ Feature ablation? → Same experiment (use tags)
```

---

## Your Current Situation

**Current approach: Single experiment `"ppa-fraud-detection"`**

This is **perfectly fine** because:
1. ✅ All models solve the same problem (fraud detection)
2. ✅ All use similar features (graph + tabular)
3. ✅ Easy to compare baseline vs hybrid models
4. ✅ All runs are related to the same research project

**When to split:**
- When you start testing completely different approaches (e.g., deep learning)
- When you have 100+ runs and need better organization
- When you want to separate research from production retraining

**For now:** Keep using `"ppa-fraud-detection"` and use **tags** and **run names** to organize!

---

## Example: When You'd Create New Experiment

```python
# Current work: Graph-based fraud detection
mlflow.set_experiment("ppa-fraud-detection")
# All your current runs here

# New research: Testing transformer-based approach
mlflow.set_experiment("ppa-fraud-detection-transformers")
# Completely different approach - new experiment makes sense

# New research: Testing on different dataset
mlflow.set_experiment("ppa-fraud-detection-synthetic-data")
# Different data source - new experiment
```

---

## Summary

1. **Experiment = Container** for related runs
2. **Create new experiment** when:
   - Major research direction change
   - Different data/problem/approach
3. **Reuse experiment** when:
   - Regular retraining
   - Hyperparameter tuning
   - Model variants (use tags!)
   - Feature ablation (use tags!)
4. **Your current setup is good** - single experiment works well for your use case
5. **Use tags and run names** to organize within an experiment before splitting

