"""
Accumulating window training logic shared across all models.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature, evaluate

from src.models.mlflow_helpers import get_model_dependencies
from src.utils.metrics import calculate_metrics
from src.utils.mlflow_feature_store import (
    log_feature_store_metadata,
    log_feature_store_statistics,
    log_feature_lineage
)
from src.features.store import FeatureStore
from src.models.feature_engineering import (
    feature_engineering,
    build_feature_sources,
)
from src.models.constants import (
    BASE_FEATURES,
    ALL_GRAPH_FEATURE_COLUMNS,
)
from src.models.common import setup_mlflow
from src.models.experiment_config import ExperimentConfig


def get_optimal_tree_method() -> str:
    """
    Detect optimal tree method based on available hardware.
    
    Note: XGBoost GPU support (gpu_hist) only works with NVIDIA CUDA GPUs.
    Apple Silicon (M1/M2/M3/M4) GPUs are NOT supported by XGBoost.
    
    Returns:
        Tree method string: "gpu_hist" if NVIDIA GPU available, "hist" otherwise
    """
    # Try to detect NVIDIA GPU availability (XGBoost only supports CUDA)
    # Method 1: Check for CUDA via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            # NVIDIA GPU detected via PyTorch, try gpu_hist
            # XGBoost will raise an error if GPU support isn't compiled
            return "gpu_hist"
    except ImportError:
        pass
    
    # Method 2: Check nvidia-smi (if available) - only detects NVIDIA GPUs
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            # NVIDIA GPU detected, try gpu_hist
            # Note: XGBoost will fall back to hist if GPU support isn't available
            return "gpu_hist"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Check for Apple Silicon (M1/M2/M3/M4) - for informational purposes
    # Note: XGBoost does NOT support Apple Silicon GPU acceleration
    try:
        import platform
        import subprocess
        # Check if running on macOS with Apple Silicon
        if platform.system() == "Darwin":
            # Check for Apple Silicon architecture
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.strip().lower()
                if "apple" in cpu_brand or any(arch in cpu_brand for arch in ["m1", "m2", "m3", "m4"]):
                    # Apple Silicon detected, but XGBoost doesn't support it
                    # Will use CPU (hist method) - this is correct behavior
                    pass
    except Exception:
        pass
    
    # Default to CPU (hist method)
    # This is correct for:
    # - Systems without GPU
    # - Apple Silicon (M1/M2/M3/M4) - XGBoost doesn't support Metal GPU
    # - Systems where GPU detection failed
    return "hist"


def get_categorical_features(features: list) -> list:
    """
    Identify categorical features that should use XGBoost's native categorical support.
    
    Returns:
        List of categorical feature names
    """
    categorical_features = []
    
    # Categorical features that should use native XGBoost support
    potential_categoricals = ["payment_type", "bundle_tier", "offer_type"]
    
    for feat in potential_categoricals:
        if feat in features:
            categorical_features.append(feat)
    
    return categorical_features


def build_feature_interaction_constraints(features: list) -> list:
    """
    Build feature interaction constraints based on domain knowledge.
    
    Interaction constraints control which features can interact in the model.
    Features in the same group can interact, but features in different groups cannot.
    
    Args:
        features: List of feature names
        
    Returns:
        List of feature name groups (each group is a list of feature names as strings)
        Empty list means no constraints (all features can interact)
        
    Note: For XGBoost 3.1.0+ with pandas DataFrames, constraints must use feature names, not indices.
    
    DOMAIN KNOWLEDGE TO FILL IN:
    Based on your fraud detection domain knowledge, define which features should
    be allowed to interact. Common patterns:
    
    1. Account age + Graph features: New accounts with high graph connectivity are suspicious
    2. Payment + Bundle: Payment type and bundle tier may interact (e.g., invoice + premium = suspicious)
    3. Graph features: Shared emails/phones/components may interact with each other
    4. Time-weighted + Graph: Recent activity combined with graph structure
    
    Example structure (TO BE FILLED IN):
    [
        [account_age_idx, shared_email_idx, shared_phone_idx, component_idx],  # Group 1
        [payment_type_idx, bundle_tier_idx],  # Group 2
        [email_velocity_idx, phone_velocity_idx, shared_email_idx],  # Group 3
    ]
    """
    constraints = []
    
    # Helper function to check if feature exists
    def has_feat(feat_name: str) -> bool:
        return feat_name in features
    
    # ============================================
    # FILL IN YOUR DOMAIN KNOWLEDGE BELOW
    # ============================================
    
    # GROUP 1: Account Age + Graph Features
    # Hypothesis: New accounts with high graph connectivity (shared emails/phones, large components) are more suspicious
    group1 = []
    if has_feat("account_age_days"):
        group1.append("account_age_days")
    
    # Add graph features that should interact with account age
    graph_features_to_interact = [
        "shared_contact_email_count",
        "shared_contact_phone_count",
        "listing_component_size",
        "max_shared_contact_email",
        "max_shared_contact_phone",
    ]
    for feat in graph_features_to_interact:
        if has_feat(feat):
            group1.append(feat)
    
    if len(group1) > 1:  # Only add if we have at least 2 features
        constraints.append(group1)
    
    # GROUP 2: Payment + Bundle Features
    # Hypothesis: Payment type and bundle tier may interact (e.g., invoice payment + premium bundle = suspicious)
    group2 = []
    payment_bundle_features = ["payment_type", "bundle_tier"]
    for feat in payment_bundle_features:
        if has_feat(feat):
            group2.append(feat)
    
    if len(group2) > 1:
        constraints.append(group2)
    
    # GROUP 3: Time-Weighted + Graph Features
    # Hypothesis: Recent activity (velocity, bursts) combined with graph structure is highly predictive
    group3 = []
    time_weighted_features = [
        "email_velocity_7d",
        "phone_velocity_7d",
        "email_is_burst",
        "phone_is_burst",
        "combined_velocity_7d",
    ]
    for feat in time_weighted_features:
        if has_feat(feat):
            group3.append(feat)
    
    # Add related graph features
    graph_velocity_features = ["shared_contact_email_count", "shared_contact_phone_count"]
    for feat in graph_velocity_features:
        if has_feat(feat) and feat not in group3:
            group3.append(feat)
    
    if len(group3) > 1:
        constraints.append(group3)
    
    # GROUP 4: Interaction Features (they're already interactions, but may interact with base features)
    # Hypothesis: Interaction features like "new_account_high_email_reuse" may interact with base account_age
    group4 = []
    interaction_features = [
        "new_account_high_email_reuse",
        "new_account_high_phone_reuse",
        "new_account_high_reuse_any",
        "new_account_invoice_payment",
        "account_age_risk_score",
    ]
    for feat in interaction_features:
        if has_feat(feat):
            group4.append(feat)
    
    # Add account_age if not already in another group
    if has_feat("account_age_days") and "account_age_days" not in [feat for group in constraints for feat in group]:
        group4.append("account_age_days")
    
    if len(group4) > 1:
        constraints.append(group4)
    
    # ============================================
    # END OF DOMAIN KNOWLEDGE SECTION
    # ============================================
    
    # Return empty list if no constraints (allows all interactions)
    # Or return constraints to restrict interactions
    return constraints if constraints else []


def build_monotonic_constraints(features: list) -> dict:
    """
    Build monotonic constraints dictionary based on domain knowledge.
    
    Monotonic constraints enforce known relationships:
    - account_age_days: -1 (fraud decreases as account age increases)
    - account_age_risk_score: 1 (fraud increases as risk score increases, but this is derived from age)
    
    Args:
        features: List of feature names
        
    Returns:
        Dictionary mapping feature name (string) to constraint (-1, 0, or 1)
        -1: Decreasing (fraud decreases as feature increases)
         0: No constraint
         1: Increasing (fraud increases as feature increases)
        
    Note: For XGBoost 3.1.0+ with pandas DataFrames, constraints must use feature names, not indices.
    """
    constraints = {}
    
    # Account age: fraud decreases as account age increases
    if "account_age_days" in features:
        constraints["account_age_days"] = -1
    
    # Note: account_age_risk_score is inversely related to account_age_days
    # (higher risk for newer accounts), so we don't constrain it separately
    # to avoid conflicts. The model will learn the relationship.
    
    return constraints


def train_accumulating_window(
    df: pl.DataFrame,
    initial_window_days: int = 180,
    step_days: int = 7,
    extra_features: list = None,
    embedding_generator=None,
    model_name: str = "baseline_graph",
    config: Optional[ExperimentConfig] = None,
    max_windows: Optional[int] = None
):
    """
    Train with accumulating window (all historical data) and MLflow tracking.
    
    Always enabled:
    - MLflow experiment tracking
    - Model saving to mlruns/models/
    - Automatic model registration to Model Registry
    - Graph features with temporal filtering (computed per window)
    
    Args:
        df: DataFrame with features and target
        initial_window_days: Initial training window size (default: 180 days)
        step_days: Step size for evaluation (default: 7 days)
        extra_features: Additional feature columns (e.g., pre-computed embeddings).
                        If embedding_generator is provided, this is ignored for embeddings.
        embedding_generator: Optional callback function(train_data, test_data, train_end) -> 
                           (train_embeddings_df, test_embeddings_df, embed_cols)
                           Generates embeddings per window with temporal filtering.
                           Returns DataFrames with embeddings joined to insertion_id, and list of embedding column names.
        model_name: Model name for MLflow registry (default: "baseline_graph")
        config: Optional experiment configuration (overrides other parameters if provided)
        max_windows: Optional maximum number of windows to evaluate (for faster optimization)
    
    Returns:
        Dictionary with run info and metrics
    """
    # Use config if provided, otherwise use function parameters
    if config is not None:
        initial_window_days = config.initial_window_days
        step_days = config.step_days
        experiment_name = config.experiment_name
        xgb_params = config.xgb_params or {}
        feature_categories = config.get_feature_columns()
    else:
        experiment_name = "ppa-fraud-detection"
        xgb_params = {}
        feature_categories = BASE_FEATURES
    
    if extra_features is None:
        extra_features = []
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Build initial feature list (graph features will be added per window)
    # Graph features are computed per window with temporal filtering
    # For hyperopt, we use the feature categories from config
    # Otherwise, we build from BASE_FEATURES + extra_features
    if config is not None:
        # feature_categories is already a list of column names from config.get_feature_columns()
        # Filter to only base features (others will be added per window)
        base_features = [f for f in feature_categories if f in BASE_FEATURES]
        features = base_features + extra_features
    else:
        features = BASE_FEATURES + extra_features
    
    target = "is_fraud"
    
    # Define window parameters  
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    # MLflow experiment setup
    setup_mlflow(experiment_name)
    
    # Initialize feature store for metadata logging
    try:
        feature_store = FeatureStore(artifacts_dir=Path("artifacts"))
    except Exception as e:
        feature_store = None
    
    # Start parent run
    with mlflow.start_run(
        run_name=f"{model_name}_accumulating_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"model_type": model_name, "training_mode": "accumulating_window"}
    ) as parent_run:
        
        # Log configuration
        mlflow.log_params({
            "model_name": model_name,
            "initial_window_days": initial_window_days,
            "step_days": step_days,
            "feature_count": len(features),
            "total_samples": len(df),
            "fraud_rate": float(df[target].mean()),
            "data_start_date": str(start_date.date()),
            "data_end_date": str(end_date.date()),
        })
        
        # Log feature store metadata
        if feature_store is not None:
            try:
                log_feature_store_metadata(
                    feature_store=feature_store,
                    feature_names=features,
                    artifacts_dir=Path("artifacts"),
                    context="training"
                )
                
                # Log feature lineage
                feature_sources = build_feature_sources(features)
                log_feature_lineage(features, feature_sources)
                
                # Log feature store statistics (sample from training data)
                sample_listing_ids = df.head(1000)["insertion_id"].to_list() if "insertion_id" in df.columns else None
                if sample_listing_ids:
                    sample_listing_ids = [x for x in sample_listing_ids if x is not None]
                
                log_feature_store_statistics(
                    feature_store=feature_store,
                    sample_listing_ids=sample_listing_ids,
                    as_of_time=end_date  # Use end of data range as reference
                )
            except Exception as e:
                pass  # Errors already logged to MLflow
        
        # Log dataset (using Polars directly - MLflow supports it)
        try:
            dataset = mlflow.data.from_polars(
                df, 
                name=f"{model_name}_fraud_detection",
                targets=target
            )
            mlflow.log_input(dataset, context="training")
        except Exception as e:
            pass  # Errors already logged to MLflow
        
        # Training loop
        current_date = start_date + timedelta(days=initial_window_days)
        test_size = timedelta(days=14)
        step_size = timedelta(days=step_days)
        
        results = []
        window_idx = 0
        best_auc_pr = 0
        best_model = None
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
                continue
            
            # Compute graph features with temporal filtering for this window
            # Use train_end as cutoff to prevent leakage (both train and test use same cutoff)
            if config is None:
                from src.models.experiment_config import ExperimentConfig
                config = ExperimentConfig()
            train_data = feature_engineering(train_data, cutoff_date=train_end, config=config)
            test_data = feature_engineering(test_data, cutoff_date=train_end, config=config)
            
            # Generate embeddings per window if callback provided (for hybrid models)
            window_embed_cols = []
            if embedding_generator is not None:
                train_embeddings_df, test_embeddings_df, embed_cols = embedding_generator(
                    train_data, test_data, train_end
                )
                # Join embeddings to train and test data
                train_data = train_data.join(train_embeddings_df, on="insertion_id", how="left")
                test_data = test_data.join(test_embeddings_df, on="insertion_id", how="left")
                window_embed_cols = embed_cols
                # Fill null embeddings with zeros
                for col in embed_cols:
                    if col in train_data.columns:
                        train_data = train_data.with_columns(pl.col(col).fill_null(0.0))
                    if col in test_data.columns:
                        test_data = test_data.with_columns(pl.col(col).fill_null(0.0))
            
            # Rebuild feature list (graph features have been added, embeddings if generated)
            if config is not None:
                # Use feature categories from config, but filter to what's actually available
                available_graph_cols = [col for col in feature_categories if col in train_data.columns and col not in BASE_FEATURES]
                base_cols = [col for col in feature_categories if col in BASE_FEATURES and col in train_data.columns]
            else:
                available_graph_cols = [col for col in ALL_GRAPH_FEATURE_COLUMNS if col in train_data.columns]
                base_cols = [col for col in BASE_FEATURES if col in train_data.columns]
            
            # Use window embeddings if generated, otherwise use pre-computed extra_features
            embed_features = window_embed_cols if window_embed_cols else extra_features
            features = base_cols + available_graph_cols + embed_features
            
            # Identify categorical features for XGBoost native support
            categorical_features = get_categorical_features(features)
            
            # Convert to pandas DataFrame to preserve categorical dtypes for XGBoost
            # XGBoost 3.1.0+ supports categorical features when passed as pandas DataFrame with categorical dtype
            train_df = train_data.select(features).to_pandas()
            test_df = test_data.select(features).to_pandas()
            
            # Convert categorical columns to pandas categorical dtype
            for cat_feat in categorical_features:
                if cat_feat in train_df.columns:
                    # Get unique values from training data to ensure consistent categories
                    unique_vals = train_df[cat_feat].dropna().unique()
                    train_df[cat_feat] = pd.Categorical(train_df[cat_feat], categories=unique_vals)
                    test_df[cat_feat] = pd.Categorical(test_df[cat_feat], categories=unique_vals)
            
            # For XGBoost 3.1.0+ with enable_categorical=True, pass pandas DataFrames directly
            # This preserves categorical dtypes which XGBoost can handle natively
            X_train = train_df
            y_train = train_data.select(target).to_numpy().flatten()
            X_test = test_df
            y_test = test_data.select(target).to_numpy().flatten()
            
            if sum(y_test) == 0:
                current_date += step_size
                continue
            
            # Nested run for this window
            with mlflow.start_run(
                run_name=f"window_{window_idx}",
                nested=True
            ) as window_run:
                
                # Enable autologging (disable model logging to handle it manually with signature)
                mlflow.xgboost.autolog(log_input_examples=False, log_model_signatures=False, log_models=False)
                
                # Log window info
                mlflow.log_params({
                    "window_index": window_idx,
                    "window_start": str(start_date.date()),
                    "window_end": str(train_end.date()),
                    "test_start": str(train_end.date()),
                    "test_end": str(test_end.date()),
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "fraud_rate_train": float(y_train.mean()),
                    "fraud_rate_test": float(y_test.mean()),
                })
                
                # Train model
                scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
                
                # Split training data temporally for early stopping (preserves accumulating window philosophy)
                # Use last 20% of training period as validation (time-based split, not random)
                if len(train_data) > 1000:  # Only split if we have enough data
                    # Calculate validation cutoff: last 20% of training period
                    train_period_days = (train_end - start_date).days
                    validation_period_days = int(train_period_days * 0.2)
                    val_cutoff = train_end - timedelta(days=validation_period_days)
                    
                    # Split train_data temporally
                    train_data_split = train_data.filter(pl.col("submission_at") < val_cutoff)
                    val_data = train_data.filter(
                        (pl.col("submission_at") >= val_cutoff) & 
                        (pl.col("submission_at") < train_end)
                    )
                    
                    # Check if validation set has enough samples and fraud cases
                    if len(val_data) >= 50 and len(val_data.filter(pl.col(target) == 1)) > 0:
                        # Convert to pandas DataFrame to preserve categorical dtypes
                        train_split_df = train_data_split.select(features).to_pandas()
                        val_df = val_data.select(features).to_pandas()
                        
                        # Convert categorical columns to pandas categorical dtype (use same categories as full training set)
                        for cat_feat in categorical_features:
                            if cat_feat in train_split_df.columns:
                                unique_vals = train_df[cat_feat].dropna().unique()  # Use categories from full training set
                                train_split_df[cat_feat] = pd.Categorical(train_split_df[cat_feat], categories=unique_vals)
                                val_df[cat_feat] = pd.Categorical(val_df[cat_feat], categories=unique_vals)
                        
                        # For XGBoost 3.1.0+ with enable_categorical=True, pass pandas DataFrames directly
                        X_train_split = train_split_df
                        y_train_split = train_data_split.select(target).to_numpy().flatten()
                        X_val = val_df
                        y_val = val_data.select(target).to_numpy().flatten()
                        
                        eval_set = [(X_val, y_val)]
                        early_stopping_rounds = 10
                        
                        # Log validation split info
                        mlflow.log_params({
                            "val_cutoff_date": str(val_cutoff.date()),
                            "train_split_size": len(train_data_split),
                            "val_split_size": len(val_data),
                            "fraud_rate_val": float(y_val.mean()),
                        })
                    else:
                        # Validation set too small or no fraud cases, skip early stopping
                        X_train_split, y_train_split = X_train, y_train
                        eval_set = None
                        early_stopping_rounds = None
                        mlflow.log_param("early_stopping_skipped", "validation_set_too_small")
                else:
                    # For small datasets, use all training data and no early stopping
                    X_train_split, y_train_split = X_train, y_train
                    eval_set = None
                    early_stopping_rounds = None
                    mlflow.log_param("early_stopping_skipped", "dataset_too_small")
                
                # Use hyperparameters from config if provided, otherwise defaults
                # Detect optimal tree method (GPU if available, otherwise CPU)
                # If config specifies tree_method, use it (overrides auto-detection)
                if xgb_params and "tree_method" in xgb_params:
                    optimal_tree_method = xgb_params["tree_method"]
                else:
                    optimal_tree_method = get_optimal_tree_method()
                
                default_params = {
                    "objective": "binary:logistic",
                    "eval_metric": "aucpr",
                    "n_estimators": 500,  # Increased since early stopping will stop early
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_jobs": -1,
                    "random_state": 42,
                    "tree_method": optimal_tree_method,  # Auto-detect GPU or use CPU
                }
                default_params.update(xgb_params)
                
                # Build monotonic constraints based on domain knowledge
                monotonic_constraints = build_monotonic_constraints(features)
                
                # Build feature interaction constraints based on domain knowledge
                interaction_constraints = build_feature_interaction_constraints(features)
                
                # Log categorical features info
                if categorical_features:
                    mlflow.log_param("categorical_features", ",".join(categorical_features))
                    mlflow.log_param("num_categorical_features", len(categorical_features))
                    # Enable categorical support in XGBoost
                    default_params["enable_categorical"] = True
                else:
                    mlflow.log_param("num_categorical_features", 0)
                
                # Log constraints info
                if monotonic_constraints:
                    # monotonic_constraints now uses feature names as keys, not indices
                    constraint_info = {
                        f"monotonic_constraint_{feat_name}": constraint
                        for feat_name, constraint in monotonic_constraints.items()
                    }
                    mlflow.log_params(constraint_info)
                    mlflow.log_param("num_monotonic_constraints", len(monotonic_constraints))
                else:
                    mlflow.log_param("num_monotonic_constraints", 0)
                
                # Log interaction constraints info
                if interaction_constraints:
                    mlflow.log_param("num_interaction_constraint_groups", len(interaction_constraints))
                    for i, group in enumerate(interaction_constraints):
                        # group already contains feature names (strings), not indices
                        group_features = group
                        mlflow.log_param(f"interaction_group_{i}", ",".join(group_features))
                else:
                    mlflow.log_param("num_interaction_constraint_groups", 0)
                    mlflow.log_param("interaction_constraints", "none")
                
                # Log tree method
                mlflow.log_param("tree_method_used", optimal_tree_method)
                
                # Set early_stopping_rounds in constructor if we have eval_set
                constructor_params = default_params.copy()
                if eval_set is not None and early_stopping_rounds is not None:
                    constructor_params["early_stopping_rounds"] = early_stopping_rounds
                
                model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    monotone_constraints=monotonic_constraints if monotonic_constraints else None,
                    interaction_constraints=interaction_constraints if interaction_constraints else None,
                    **constructor_params
                )
                
                # Fit with early stopping if validation set is available
                if eval_set is not None:
                    model.fit(
                        X_train_split, y_train_split,
                        eval_set=eval_set,
                        verbose=False
                    )
                    # Log early stopping info
                    best_iteration = model.get_booster().best_iteration
                    mlflow.log_param("best_iteration", best_iteration if best_iteration is not None else default_params["n_estimators"])
                else:
                    model.fit(X_train_split, y_train_split)
                    mlflow.log_param("best_iteration", default_params["n_estimators"])
                
                # Manually log model with signature and input example
                # Infer signature from pandas DataFrame (before numpy conversion) to preserve categorical types
                # This ensures signature correctly reflects categorical features
                train_df_for_signature = train_data.select(features).to_pandas()
                # Convert categorical columns for signature inference and input example
                # Ensure they're properly typed as category (not object) for MLflow validation
                for cat_feat in categorical_features:
                    if cat_feat in train_df_for_signature.columns:
                        unique_vals = train_df_for_signature[cat_feat].dropna().unique()
                        # Sort to ensure consistent category ordering
                        unique_vals = sorted(unique_vals)
                        train_df_for_signature[cat_feat] = pd.Categorical(
                            train_df_for_signature[cat_feat], 
                            categories=unique_vals,
                            ordered=False
                        )
                
                # Infer signature from pandas DataFrame (preserves categorical types)
                signature = infer_signature(
                    train_df_for_signature.head(100),  # Use sample for signature
                    model.predict(X_train[:100])  # Match predictions to sample
                )
                # Use pandas DataFrame for input example (preserves categorical types)
                # MLflow validation accepts category dtype, but columns must be category, not object
                input_example = train_df_for_signature.head(5).copy()
                # Ensure categorical dtypes are preserved (they should be from the conversion above)
                for cat_feat in categorical_features:
                    if cat_feat in input_example.columns and input_example[cat_feat].dtype.name != 'category':
                        # Re-convert if somehow lost
                        unique_vals = sorted(input_example[cat_feat].dropna().unique())
                        input_example[cat_feat] = pd.Categorical(
                            input_example[cat_feat],
                            categories=unique_vals,
                            ordered=False
                        )
                
                # Get explicit dependencies
                deps = get_model_dependencies()
                
                # Build params dict, excluding None values
                model_params = {
                    "objective": "binary:logistic",
                    "eval_metric": "aucpr",
                    "n_estimators": default_params["n_estimators"],
                    "max_depth": default_params["max_depth"],
                    "learning_rate": default_params["learning_rate"],
                    "scale_pos_weight": float(scale_pos_weight),
                    "random_state": 42,
                    "tree_method": default_params.get("tree_method", "hist"),
                }
                if eval_set is not None:
                    model_params["early_stopping_rounds"] = early_stopping_rounds
                
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    **deps,  # Add explicit dependencies
                    metadata={
                        "model_type": "XGBoost Classifier",
                        "task": "fraud_detection",
                        "window_index": window_idx,
                        "training_mode": "accumulating_window",
                        "feature_count": len(features),
                        "train_size": len(train_data),
                        "test_size": len(test_data),
                    },
                    params=model_params
                )
                
                # Extract and log feature importance
                try:
                    feature_importance = model.get_booster().get_score(importance_type='gain')
                    # Create DataFrame with feature importance
                    importance_data = []
                    for i, feat_name in enumerate(features):
                        importance_value = feature_importance.get(f'f{i}', 0.0)
                        importance_data.append({
                            'feature': feat_name,
                            'importance': importance_value
                        })
                    
                    importance_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
                    
                    # Log top 50 features as table artifact
                    mlflow.log_table(
                        data=importance_df.head(50),
                        artifact_file="feature_importance.json"
                    )
                    
                    # Log top 10 feature importances as metrics for quick comparison
                    top_features = importance_df.head(10)
                    for idx, row in top_features.iterrows():
                        # Sanitize feature name for MLflow metric (replace special chars with underscore)
                        safe_feature_name = str(row['feature']).replace('/', '_').replace(' ', '_').replace('-', '_')
                        mlflow.log_metric(f"feature_importance_{safe_feature_name}", row['importance'])
                except Exception as e:
                    # Feature importance extraction failed, continue without it
                    pass
                
                # Predict
                proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                metrics = calculate_metrics(y_test, proba)
                
                # Log custom metrics
                mlflow.log_metrics({
                    "auc_pr": metrics["auc_pr"],
                    "auc_roc": metrics["auc_roc"],
                    "p_at_100": metrics["p@100"],
                    "lift_at_100": metrics["lift@100"],
                    "fraud_count": metrics["fraud_count"],
                })
                
                # Use mlflow.models.evaluate for SHAP (replaces deprecated mlflow.evaluate)
                # Note: mlflow.models.evaluate() requires pandas DataFrame, not Polars
                # This is a small test set conversion, so acceptable performance impact
                try:
                    # Convert to pandas and cast to float to avoid integer schema warnings
                    eval_data = test_data.select(features).to_pandas().astype(float)
                    eval_data["target"] = y_test
                    
                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    
                    # Use mlflow.models.evaluate() instead of deprecated mlflow.evaluate()
                    # evaluator_config must be a dict mapping evaluator name to config dict
                    evaluate(
                        model=model_uri,
                        data=eval_data,
                        targets="target",
                        model_type="classifier",
                        evaluators=["default"],
                        evaluator_config={"default": {"log_explainer": False}}
                    )
                except Exception as e:
                    pass  # Errors already logged to MLflow
                
                # Track best model
                if metrics["auc_pr"] > best_auc_pr:
                    best_auc_pr = metrics["auc_pr"]
                    best_model = model
                    best_run_id = window_run.info.run_id
                
                results.append({
                    "window_idx": window_idx,
                    "window_start": train_end,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    **metrics
                })
            
            window_idx += 1
            current_date += step_size
            
            # Stop early if max_windows is set (for hyperparameter optimization)
            if max_windows is not None and window_idx >= max_windows:
                break
        
        # Log aggregate metrics to parent run
        if results:
            results_df = pl.DataFrame(results)
            mean_auc_pr = float(results_df["auc_pr"].mean())
            mean_p100 = float(results_df["p@100"].mean())
            
            mlflow.log_metrics({
                "mean_auc_pr": mean_auc_pr,
                "mean_p_at_100": mean_p100,
                "best_auc_pr": best_auc_pr,
                "num_windows": float(len(results)),
            })
            
            # Save results CSV as artifact
            results_path = f"artifacts/results/{model_name}_accumulating_results.csv"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            results_df.write_csv(results_path)
            mlflow.log_artifact(results_path, artifact_path="results")
            
            # Register best model to Model Registry
            if best_model is not None and best_run_id is not None:
                try:
                    model_uri = f"runs:/{best_run_id}/model"
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name=f"fraud-detection-{model_name}"
                    )
                    
                    # Add description to model version
                    client = mlflow.tracking.MlflowClient()
                    client.update_model_version(
                        name=registered_model.name,
                        version=registered_model.version,
                        description=f"Accumulating window training. Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}"
                    )
                    
                    mlflow.log_param("registered_model_version", registered_model.version)
                    
                except Exception as e:
                    pass  # Errors already logged to MLflow
        
        return {
            "run_id": parent_run.info.run_id,
            "best_run_id": best_run_id,  # Add best run ID for hybrid model logging
            "results": results,
            "mean_auc_pr": mean_auc_pr if results else 0,
            "best_auc_pr": best_auc_pr,
            "num_windows": len(results),
        }

