"""
XGBoost trainer with accumulating window.
"""
import os
import time
import logging
import polars as pl
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature, evaluate
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from src.models.xgboost.utils import (
    get_optimal_tree_method,
    get_categorical_features,
    build_monotonic_constraints,
    build_feature_interaction_constraints,
    validate_features,
)
from src.models.utils.common import setup_mlflow
from src.models.registry import ModelRegistry
from src.models.utils.mlflow_helpers import get_model_dependencies
from src.utils.metrics import calculate_metrics, measure_inference_latency
from src.features.processor import FeatureProcessor

logger = logging.getLogger(__name__)

from typing import Callable

# Type alias for embedding generator callback
EmbeddingGenerator = Callable[[pl.DataFrame, pl.DataFrame, datetime], tuple]


def train_accumulating_window(
    df: pl.DataFrame,
    config: Dict[str, Any],
    max_windows: Optional[int] = None,
    embedding_generator: Optional[EmbeddingGenerator] = None
) -> Dict[str, Any]:
    """
    Train XGBoost with accumulating window and MLflow tracking.
    
    Supports hybrid GNN+XGBoost training via embedding_generator callback.
    
    Args:
        df: DataFrame with base features and target
        config: Hydra configuration dictionary
        max_windows: Optional maximum number of windows (for debugging)
        embedding_generator: Optional callback for per-window embedding generation.
            Signature: (train_data, test_data, train_end) -> (train_embed_df, test_embed_df, embed_cols)
            The embedding DataFrames should have 'insertion_id' as join key.
        
    Returns:
        Dictionary with run info and metrics
    """
    # Extract config
    model_name = config.model.name
    initial_window_days = config.model.training.initial_window_days
    step_days = config.model.training.step_days
    xgb_params = dict(config.model.params)
    feature_categories = config.features.categories
    experiment_name = config.experiment_name
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Define window parameters
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    target = "is_fraud"
    
    # Setup MLflow
    setup_mlflow(experiment_name)
    
    # Start parent run
    training_start_time = time.time()
    with mlflow.start_run(
        run_name=f"{model_name}_accumulating_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"model_type": model_name, "training_mode": "accumulating_window"}
    ) as parent_run:
        
        # Log configuration
        mlflow.log_params({
            "model_name": model_name,
            "initial_window_days": initial_window_days,
            "step_days": step_days,
            "total_samples": len(df),
            "fraud_rate": float(df[target].mean()),
            "data_start_date": str(start_date.date()),
            "data_end_date": str(end_date.date()),
            "feature_categories": ",".join(feature_categories),
            "uses_gnn_embeddings": embedding_generator is not None,
        })
        
        # Training loop
        current_date = start_date + timedelta(days=initial_window_days)
        test_size = timedelta(days=14)
        step_size = timedelta(days=step_days)
        
        results = []
        window_idx = 0
        best_auc_pr = 0
        best_run_id = None
        
        while current_date + test_size <= end_date:
            train_end = current_date
            test_end = current_date + test_size
            
            # ACCUMULATING WINDOW: Use ALL data from start to train_end
            train_data = df.filter(pl.col("submission_at") < train_end)
            test_data = df.filter(
                (pl.col("submission_at") >= train_end) & 
                (pl.col("submission_at") < test_end)
            )
            
            if len(test_data) < 50 or len(train_data) < 1000:
                current_date += step_size
                if max_windows is not None and window_idx >= max_windows:
                    break
                continue
            
            # Generate features with temporal filtering
            feature_processor = FeatureProcessor.from_config(config.features)
            train_data = feature_processor.process(train_data, cutoff_date=train_end)
            test_data = feature_processor.process(test_data, cutoff_date=train_end)
            
            # Generate and merge GNN embeddings if embedding_generator is provided
            embed_cols = []
            if embedding_generator is not None:
                try:
                    logger.info(f"Generating embeddings for window {window_idx}...")
                    train_embed_df, test_embed_df, embed_cols = embedding_generator(
                        train_data, test_data, train_end
                    )
                    
                    # Join embeddings with feature data
                    train_data = train_data.join(
                        train_embed_df, on="insertion_id", how="left"
                    )
                    test_data = test_data.join(
                        test_embed_df, on="insertion_id", how="left"
                    )
                    
                    # Fill any missing embeddings with zeros
                    for col in embed_cols:
                        train_data = train_data.with_columns(
                            pl.col(col).fill_null(0.0)
                        )
                        test_data = test_data.with_columns(
                            pl.col(col).fill_null(0.0)
                        )
                    
                    logger.info(f"Added {len(embed_cols)} embedding features")
                    
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}. Continuing without embeddings.")
                    embed_cols = []
            
            # Convert to pandas
            train_df = train_data.to_pandas()
            test_df = test_data.to_pandas()
            
            # Validate features against manifest - this ensures only declared features are used
            # and catches type mismatches early rather than silently filtering
            valid_features, invalid_cols = validate_features(train_df, strict=False)
            
            # Add embedding columns to valid features (they're not in the manifest)
            if embed_cols:
                valid_features = valid_features + embed_cols
                logger.info(f"Added {len(embed_cols)} embedding columns to feature set")
            
            if not valid_features:
                raise ValueError(
                    f"No valid features found! Check that feature generators are producing expected columns. "
                    f"Invalid columns: {invalid_cols[:10]}..."  # Show first 10
                )
            
            logger.info(f"Using {len(valid_features)} features ({len(valid_features) - len(embed_cols)} manifest + {len(embed_cols)} embeddings)")
            
            # Get categorical features from the valid set
            categorical_features = get_categorical_features(valid_features)
            
            # Convert categorical columns to pandas categorical dtype for XGBoost
            for cat_feat in categorical_features:
                unique_vals = train_df[cat_feat].dropna().unique()
                train_df[cat_feat] = pd.Categorical(train_df[cat_feat], categories=unique_vals)
                test_df[cat_feat] = pd.Categorical(test_df[cat_feat], categories=unique_vals)
            
            # Select only manifest-validated features
            X_train = train_df[valid_features]
            y_train = train_data.select(target).to_numpy().flatten()
            X_test = test_df[valid_features]
            y_test = test_data.select(target).to_numpy().flatten()
            
            if sum(y_test) == 0:
                current_date += step_size
                if max_windows is not None and window_idx >= max_windows:
                    break
                continue
            
            # Nested run for this window
            with mlflow.start_run(
                run_name=f"window_{window_idx}",
                nested=True
            ) as window_run:
                
                mlflow.xgboost.autolog(log_input_examples=False, log_model_signatures=False, log_models=False)
                
                # Log window info
                mlflow.log_params({
                    "window_index": window_idx,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "fraud_rate_train": float(y_train.mean()),
                    "fraud_rate_test": float(y_test.mean()),
                })
                
                # Train model
                scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
                
                # Build constraints
                monotonic_constraints = build_monotonic_constraints(valid_features)
                interaction_constraints = build_feature_interaction_constraints(valid_features)
                
                # Resolve tree method
                tree_method = xgb_params.get("tree_method", "auto")
                if tree_method == "auto":
                    tree_method = get_optimal_tree_method()
                    xgb_params["tree_method"] = tree_method
                
                # Enable categorical support if needed
                if categorical_features:
                    xgb_params["enable_categorical"] = True
                
                model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    monotone_constraints=monotonic_constraints if monotonic_constraints else None,
                    interaction_constraints=interaction_constraints if interaction_constraints else None,
                    **xgb_params
                )
                
                model.fit(X_train, y_train)
                
                # Log model
                signature = infer_signature(X_train.head(100), model.predict(X_train[:100]))
                input_example = X_train.head(5).copy()
                
                deps = get_model_dependencies()
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
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
                
                # Track best model
                if metrics["auc_pr"] > best_auc_pr:
                    best_auc_pr = metrics["auc_pr"]
                    best_run_id = window_run.info.run_id
                
                results.append({
                    "window_idx": window_idx,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    **metrics
                })
            
            window_idx += 1
            current_date += step_size
            
            if max_windows is not None and window_idx >= max_windows:
                break
        
        # Log aggregate metrics
        training_time_minutes = (time.time() - training_start_time) / 60.0
        mlflow.set_tag("training_time_minutes", f"{training_time_minutes:.2f}")
        
        if results:
            results_df = pl.DataFrame(results)
            mean_auc_pr = float(results_df["auc_pr"].mean())
            mean_auc_roc = float(results_df["auc_roc"].mean())
            
            mlflow.log_metrics({
                "mean_auc_pr": mean_auc_pr,
                "mean_auc_roc": mean_auc_roc,
                "best_auc_pr": best_auc_pr,
                "num_windows": float(len(results)),
            })
            
            # Register best model
            if best_run_id is not None:
                ModelRegistry.register_model(
                    run_id=best_run_id,
                    model_name=f"fraud-detection-{model_name}",
                    description=f"Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}"
                )
        
        return {
            "run_id": parent_run.info.run_id,
            "best_run_id": best_run_id,
            "results": results,
            "mean_auc_pr": mean_auc_pr if results else 0,
            "best_auc_pr": best_auc_pr,
            "num_windows": len(results),
        }
