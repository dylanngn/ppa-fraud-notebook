"""
Generic XGBoost trainer.

Trains a single model on provided train/test data.
Does NOT handle data splitting - that's the orchestrator's job.
"""
import logging
import warnings
import pandas as pd
import xgboost as xgb
import mlflow

# Suppress MLflow's integer column warning - XGBoost natively handles missing values
# and we convert int columns to float64 anyway
warnings.filterwarnings(
    "ignore",
    message=".*Inferred schema contains integer column.*",
    category=UserWarning
)
from mlflow.models import infer_signature
from typing import Dict, Any, List

from src.models.xgb_trainer.utils import (
    get_optimal_tree_method,
    get_categorical_features,
    build_monotonic_constraints,
    build_feature_interaction_constraints,
)
from src.models.utils.mlflow_helpers import get_model_dependencies
from src.utils.metrics import calculate_metrics, measure_inference_latency

logger = logging.getLogger(__name__)


def train_single_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    xgb_params: Dict[str, Any],
    window_idx: int = 0,
    log_model: bool = True,
    nested: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost model on a single train/test split.
    
    This is a GENERIC trainer - it doesn't know about temporal splits,
    accumulating windows, or data loading. It just trains on what it receives.
    
    Args:
        train_df: Training DataFrame (pandas)
        test_df: Test DataFrame (pandas)
        feature_cols: List of feature column names
        target_col: Target column name
        xgb_params: XGBoost parameters
        window_idx: Window index for logging
        log_model: Whether to log model to MLflow
        nested: Whether to create a nested MLflow run
        
    Returns:
        Dictionary with metrics and model info
    """
    if not feature_cols:
        raise ValueError("No feature columns provided!")
    
    # Extract features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].values
    
    if sum(y_test) == 0:
        logger.warning(f"Window {window_idx}: No fraud in test set, skipping")
        return {"skipped": True, "window_idx": window_idx}
    
    # Get categorical features
    categorical_features = get_categorical_features(feature_cols)
    
    # Convert categorical columns to pandas categorical dtype for XGBoost
    # Make explicit copies to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()
    for cat_feat in categorical_features:
        unique_vals = train_df[cat_feat].dropna().unique()
        X_train[cat_feat] = pd.Categorical(X_train[cat_feat], categories=unique_vals)
        X_test[cat_feat] = pd.Categorical(X_test[cat_feat], categories=unique_vals)
    
    # Create nested run if requested
    from contextlib import nullcontext
    run_context = (
        mlflow.start_run(run_name=f"window_{window_idx}", nested=True)
        if nested
        else nullcontext()
    )
    
    with run_context as window_run:
        # Log window info
        mlflow.log_params({
            "window_index": window_idx,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_test": float(y_test.mean()),
            "num_features": len(feature_cols),
        })
        
        # Train model
        scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
        
        # Build constraints
        monotonic_constraints = build_monotonic_constraints(feature_cols)
        interaction_constraints = build_feature_interaction_constraints(feature_cols)
        
        # Resolve tree method
        xgb_params_copy = xgb_params.copy()
        tree_method = xgb_params_copy.get("tree_method", "auto")
        if tree_method == "auto":
            tree_method = get_optimal_tree_method()
            xgb_params_copy["tree_method"] = tree_method
        
        # Enable categorical support if needed
        if categorical_features:
            xgb_params_copy["enable_categorical"] = True
        
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            monotone_constraints=monotonic_constraints if monotonic_constraints else None,
            interaction_constraints=interaction_constraints if interaction_constraints else None,
            **xgb_params_copy
        )
        
        model.fit(X_train, y_train)
        
        # Log model if requested
        model_info = None
        if log_model:
            signature = infer_signature(X_train.head(100), model.predict(X_train[:100]))
            deps = get_model_dependencies()
            model_info = mlflow.xgboost.log_model(
                xgb_model=model,
                name="model",
                signature=signature,
                **deps
            )
        
        # Predict and evaluate
        proba = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, proba)
        
        # Measure inference latency (sample of 100 rows)
        latency_sample = X_test.head(min(100, len(X_test)))
        latency_metrics = measure_inference_latency(model, latency_sample, n_iterations=50)
        
        mlflow.log_metrics({
            "auc_pr": metrics["auc_pr"],
            "auc_roc": metrics["auc_roc"],
            "p_at_100": metrics["p@100"],
            "latency_mean_ms": latency_metrics.get("latency_mean_ms", 0),
            "latency_p95_ms": latency_metrics.get("latency_p95_ms", 0),
            "latency_per_sample_ms": latency_metrics.get("latency_per_sample_ms", 0),
        })
        
        logger.info(
            f"Window {window_idx}: AUC-PR={metrics['auc_pr']:.4f}, "
            f"AUC-ROC={metrics['auc_roc']:.4f}, P@100={metrics['p@100']:.4f}"
        )
        
        result = {
            "skipped": False,
            "window_idx": window_idx,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "model": model,
            **metrics
        }
        
        if window_run and log_model and model_info:
            result["run_id"] = window_run.info.run_id
            result["model_uri"] = model_info.model_uri
        
        return result
