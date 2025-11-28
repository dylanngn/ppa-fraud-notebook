import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature, evaluate

from src.utils.metrics import calculate_metrics
from src.utils.mlflow_feature_store import (
    log_feature_store_metadata,
    log_feature_store_statistics,
    log_feature_lineage
)
from src.features.store import FeatureStore

GRAPH_FEATURES_PATH = Path("artifacts/listing_graph_features.parquet")
GRAPH_FEATURE_COLUMNS = [
    "contact_email_count",
    "shared_contact_email_count",
    "max_shared_contact_email",
    "contact_phone_count",
    "shared_contact_phone_count",
    "max_shared_contact_phone",
    "user_listing_count",
    "user_unique_ip_count",
    "shared_ip_user_count",
    "max_shared_ip_users",
    "listing_component_size",
    "listing_pagerank",
]

ADVANCED_GRAPH_FEATURES_PATH = Path("artifacts/listing_advanced_features.parquet")
ADVANCED_GRAPH_FEATURE_COLUMNS = [
    "degree_total",
    "is_isolated",
    "unique_identifier_count",
    "neighbor_overlap_score",
    "avg_neighbor_degree",
]

TIME_WEIGHTED_FEATURES_PATH = Path("artifacts/listing_time_weighted_features.parquet")
TIME_WEIGHTED_FEATURE_COLUMNS = [
    # Email time-weighted features
    "email_total_historical",
    "email_count_7d",
    "email_count_30d",
    "email_recent_weighted",
    "email_velocity_7d",
    "email_acceleration",
    "email_is_burst",
    "email_is_dormant_reactivation",
    "email_recency_weighted",
    "email_time_spread",
    # Phone time-weighted features
    "phone_total_historical",
    "phone_count_7d",
    "phone_count_30d",
    "phone_recent_weighted",
    "phone_velocity_7d",
    "phone_acceleration",
    "phone_is_burst",
    "phone_is_dormant_reactivation",
    "phone_recency_weighted",
    "phone_time_spread",
    # Combined features
    "combined_velocity_7d",
    "combined_acceleration",
    "any_burst",
    "any_dormant_reactivation",
    "combined_recency_weighted",
]

INTERACTION_FEATURES_PATH = Path("artifacts/listing_interaction_features.parquet")
INTERACTION_FEATURE_COLUMNS = [
    # Binary interactions
    "new_account_high_email_reuse",
    "new_account_high_phone_reuse",
    "new_account_high_reuse_any",
    "new_account_invoice_payment",
    "very_new_account_invoice",
    "is_small_listing",
    "new_account_small_listing",
    "in_large_component",
    "new_account_large_component",
    "new_account_isolated",
    # Continuous scores
    "account_age_risk_score",
    "email_reuse_intensity",
    "component_risk_score",
    "suspicious_combo_score",
]

TEXT_FEATURES_PATH = Path("artifacts/listing_text_features.parquet")
TEXT_FEATURE_COLUMNS = [
    "description_length",
    "description_word_count",
    "description_has_url",
    "description_has_email",
    "description_has_phone",
    "description_caps_ratio",
    "description_exclamation_count",
    "description_question_count",
    "description_all_caps_words",
    "description_avg_word_length",
]


def compute_graph_features_for_window(cutoff_date) -> pl.DataFrame:
    """
    Compute graph features for a specific time window.
    
    This function computes all graph features using only data before cutoff_date,
    preventing temporal leakage.
    
    Args:
        cutoff_date: Only use edges from listings before this date
        
    Returns:
        DataFrame with graph features for all listings (features computed with cutoff_date)
    """
    from src.features import (
        graph_features,
        advanced_graph_features,
        time_weighted_features,
        interaction_features,
        text_features
    )
    from pathlib import Path
    import tempfile
    import os
    
    # Use temporary files to avoid overwriting the main feature files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Compute each feature type with temporal filtering
        df = None
        
        # Graph features
        graph_path = tmp_path / "graph_features.parquet"
        graph_features.generate_graph_features(
            output_path=graph_path,
            cutoff_date=cutoff_date
        )
        if graph_path.exists():
            graph_df = pl.read_parquet(graph_path)
            df = graph_df if df is None else df.join(graph_df, on="insertion_id", how="outer")
        
        # Advanced graph features
        advanced_path = tmp_path / "advanced_features.parquet"
        advanced_graph_features.generate_advanced_features(
            output_path=advanced_path,
            cutoff_date=cutoff_date
        )
        if advanced_path.exists():
            advanced_df = pl.read_parquet(advanced_path)
            df = advanced_df if df is None else df.join(advanced_df, on="insertion_id", how="outer")
        
        # Time-weighted features
        time_weighted_path = tmp_path / "time_weighted_features.parquet"
        time_weighted_features.generate_time_weighted_features(
            output_path=time_weighted_path,
            cutoff_date=cutoff_date
        )
        if time_weighted_path.exists():
            time_df = pl.read_parquet(time_weighted_path)
            df = time_df if df is None else df.join(time_df, on="insertion_id", how="outer")
        
        # Interaction features (depends on graph features)
        # Pass the computed graph_features and advanced_features DataFrames directly
        interaction_path = tmp_path / "interaction_features.parquet"
        
        # Extract graph_features and advanced_features from the combined df if they exist
        graph_features_for_interaction = None
        advanced_features_for_interaction = None
        
        if graph_path.exists():
            graph_features_for_interaction = pl.read_parquet(graph_path)
        
        if advanced_path.exists():
            advanced_features_for_interaction = pl.read_parquet(advanced_path)
        
        # Compute interaction features using the temporally-filtered graph features
        interaction_features.generate_interaction_features(
            output_path=interaction_path,
            cutoff_date=cutoff_date,
            graph_features_df=graph_features_for_interaction,
            advanced_features_df=advanced_features_for_interaction
        )
        if interaction_path.exists():
            interaction_df = pl.read_parquet(interaction_path)
            df = interaction_df if df is None else df.join(interaction_df, on="insertion_id", how="outer")
        
        # Text features (no temporal filtering needed - they're per-listing)
        if TEXT_FEATURES_PATH.exists():
            text_df = pl.read_parquet(TEXT_FEATURES_PATH)
            df = text_df if df is None else df.join(text_df, on="insertion_id", how="outer")
        
        if df is None:
            raise ValueError("No graph features were computed")
        
        return df


def load_graph_features() -> pl.DataFrame:
    df = None
    if GRAPH_FEATURES_PATH.exists():
        print("Loading graph-derived features...")
        df = pl.read_parquet(GRAPH_FEATURES_PATH)
    
    if ADVANCED_GRAPH_FEATURES_PATH.exists():
        print("Loading advanced graph features...")
        advanced = pl.read_parquet(ADVANCED_GRAPH_FEATURES_PATH)
        if df is not None:
            df = df.join(advanced, on="insertion_id", how="left")
        else:
            df = advanced
    
    if TIME_WEIGHTED_FEATURES_PATH.exists():
        print("Loading time-weighted graph features...")
        time_weighted = pl.read_parquet(TIME_WEIGHTED_FEATURES_PATH)
        if df is not None:
            df = df.join(time_weighted, on="insertion_id", how="left")
        else:
            df = time_weighted
    
    if INTERACTION_FEATURES_PATH.exists():
        print("Loading interaction features...")
        interactions = pl.read_parquet(INTERACTION_FEATURES_PATH)
        if df is not None:
            df = df.join(interactions, on="insertion_id", how="left")
        else:
            df = interactions
    
    if TEXT_FEATURES_PATH.exists():
        print("Loading text features...")
        text_features = pl.read_parquet(TEXT_FEATURES_PATH)
        if df is not None:
            df = df.join(text_features, on="insertion_id", how="left")
        else:
            df = text_features
            
    if df is None:
        raise FileNotFoundError(
            f"Graph features not found. "
            "Run `make graph-features` and `python src/cli.py advanced-graph-features` to generate them."
        )
    return df


def load_data():
    """
    Loads and joins listing and user data.
    """
    print("Loading data...")
    df_listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    
    df = df_listings.join(df_users, on="user_id", how="left")
    
    return df

def _add_base_tabular_features(df):
    """
    Add base tabular features that don't depend on graph structure.
    
    These are computed once and reused across all windows because they are
    intrinsic properties of each listing (fixed at submission time):
    - account_age_days: Age of account when listing was submitted (fixed)
    - log_price: Price of the listing (fixed)
    - living_space, rooms, etc.: Properties of the property (fixed)
    
    Unlike graph features (which are relational and change as graph grows),
    tabular features are point-in-time snapshots that don't depend on other
    listings or future data.
    
    Graph features are computed separately per window with temporal filtering.
    """
    # 1. Account Age (The critical feature)
    df = df.with_columns(
        (pl.col("submission_at") - pl.col("account_created_at")).dt.total_days().alias("account_age_days")
    )
    
    # 2. Price Normalization (Simple log)
    df = df.with_columns(
        pl.col("price_rent_gross").log1p().alias("log_price")
    )
    
    # 3. Boolean features (Cast to Int)
    bool_cols = ["is_new", "has_balcony", "has_elevator", "has_parking"]
    bool_exprs = [
        pl.col(col).fill_null(False).cast(pl.Int8) if col in df.columns else pl.lit(0).alias(col)
        for col in bool_cols
    ]
    df = df.with_columns(bool_exprs)
            
    # 4. Bundle Info
    df = df.with_columns([
        pl.col("bundle_period").fill_null(7) if "bundle_period" in df.columns else pl.lit(7).alias("bundle_period"),
        (
            pl.col("bundle_tier").fill_null("basic").str.to_lowercase()
            .replace_strict({"basic": 0, "premium": 1, "top": 2}, default=0)
            .cast(pl.Int64).alias("bundle_tier_score")
        ) if "bundle_tier" in df.columns else pl.lit(0).alias("bundle_tier_score")
    ])

    # 5. Payment Type (Binary: DIRECT vs INVOICE)
    df = df.with_columns(
        (pl.col("payment_type") == "DIRECT").cast(pl.Int8).alias("is_direct_payment")
        if "payment_type" in df.columns else pl.lit(0).alias("is_direct_payment")
    )
        
    # 6. Offer Type (Binary: BUY vs RENT)
    df = df.with_columns(
        (pl.col("offer_type") == "BUY").cast(pl.Int8).alias("is_buy")
        if "offer_type" in df.columns else pl.lit(0).alias("is_buy")
    )
        
    # 7. Location features
    location_exprs = [
        pl.col(col).fill_null(0.0) if col in df.columns else pl.lit(0.0).alias(col)
        for col in ["latitude", "longitude"]
    ]
    df = df.with_columns(location_exprs)

    return df


def feature_engineering(df, cutoff_date=None):
    """
    Creates tabular features for XGBoost.
    
    Always includes graph features with temporal filtering to prevent data leakage.
    
    Args:
        df: DataFrame with listings
        cutoff_date: If provided, compute graph features with temporal filtering.
                     If None, load pre-computed features (for backward compatibility).
    """
    print("Engineering features...")
    
    if cutoff_date is not None:
        # Compute features per window with temporal filtering
        print(f"Computing graph features with cutoff_date: {cutoff_date}")
        graph_features = compute_graph_features_for_window(cutoff_date)
    else:
        # Load pre-computed features (backward compatibility)
        print("Loading pre-computed graph features...")
        graph_features = load_graph_features()
    
    # Cast UInt32 columns to Int64 to avoid MLflow warnings
    graph_features = graph_features.select([
        pl.col(c).cast(pl.Int64) if graph_features[c].dtype == pl.UInt32 else pl.col(c)
        for c in graph_features.columns
    ])
    
    df = df.join(graph_features, on="insertion_id", how="left")
    all_graph_cols = (
        GRAPH_FEATURE_COLUMNS + 
        ADVANCED_GRAPH_FEATURE_COLUMNS + 
        TIME_WEIGHTED_FEATURE_COLUMNS + 
        INTERACTION_FEATURE_COLUMNS +
        TEXT_FEATURE_COLUMNS
    )
    df = df.with_columns([
        pl.col(col).fill_null(0) for col in all_graph_cols if col in df.columns
    ])
    
    return df

def get_base_features():
    """Returns the list of base tabular features."""
    return [
        "account_age_days", "log_price", "living_space", "rooms",
        "is_new", "has_balcony", "has_elevator", "has_parking",
        "bundle_period", "bundle_tier_score",
        "is_direct_payment", "is_buy",
        "latitude", "longitude"
    ]


def _build_feature_sources(feature_names: list) -> dict:
    """Build a mapping of feature names to their sources."""
    sources = {}
    
    # Base tabular features
    base_features = get_base_features()
    for feat in base_features:
        if feat in feature_names:
            sources[feat] = "tabular"
    
    # Graph features
    graph_features = GRAPH_FEATURE_COLUMNS
    for feat in graph_features:
        if feat in feature_names:
            sources[feat] = "graph"
    
    # Advanced graph features
    advanced_features = ADVANCED_GRAPH_FEATURE_COLUMNS
    for feat in advanced_features:
        if feat in feature_names:
            sources[feat] = "graph_advanced"
    
    # Time-weighted features
    time_weighted_features = TIME_WEIGHTED_FEATURE_COLUMNS
    for feat in time_weighted_features:
        if feat in feature_names:
            sources[feat] = "time_weighted"
    
    # Interaction features
    interaction_features = INTERACTION_FEATURE_COLUMNS
    for feat in interaction_features:
        if feat in feature_names:
            sources[feat] = "interaction"
    
    # Text features
    text_features = TEXT_FEATURE_COLUMNS
    for feat in text_features:
        if feat in feature_names:
            sources[feat] = "text"
    
    # Default for any remaining
    for feat in feature_names:
        if feat not in sources:
            sources[feat] = "unknown"
    
    return sources


def train_accumulating_window(df, initial_window_days=180, step_days=7, extra_features=None, 
                              embedding_generator=None, model_name="baseline_graph"):
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
        embedding_generator: Optional callback function(train_data, test_data, train_end) -> (train_embeddings_df, test_embeddings_df, embed_cols)
                           Generates embeddings per window with temporal filtering.
                           Returns DataFrames with embeddings joined to insertion_id, and list of embedding column names.
        model_name: Model name for MLflow registry (default: "baseline_graph")
    
    Returns:
        Dictionary with run info and metrics
    """
    if extra_features is None:
        extra_features = []
    
    print("="*70)
    print(f"ACCUMULATING WINDOW TRAINING - {model_name.upper()}")
    print("="*70)
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Build initial feature list (graph features will be added per window)
    # Graph features are computed per window with temporal filtering
    features = get_base_features() + extra_features
    
    target = "is_fraud"
    
    # Define window parameters  
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    print(f"Data range: {start_date.date()} to {end_date.date()}")
    print(f"Total listings: {len(df):,}")
    print(f"Features: {len(features)}")
    print(f"Initial window: {initial_window_days} days, Step: {step_days} days\n")
    
    # MLflow experiment setup
    # Initialize MLflow with database backend (idempotent - safe to call multiple times)
    # Force SQLite backend to avoid filesystem deprecation warnings
    
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    experiment_name = "ppa-fraud-detection"
    mlflow.set_experiment(experiment_name)
    
    # Initialize feature store for metadata logging
    try:
        feature_store = FeatureStore(artifacts_dir=Path("artifacts"))
    except Exception as e:
        print(f"Warning: Could not initialize FeatureStore: {e}")
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
                feature_sources = _build_feature_sources(features)
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
                print(f"Warning: Feature store metadata logging failed: {e}")
            except Exception as e:
                print(f"Warning: Feature store metadata logging failed: {e}")
        
        # Log dataset (using Polars directly - MLflow supports it)
        try:
            dataset = mlflow.data.from_polars(
                df, 
                name=f"{model_name}_fraud_detection",
                targets=target
            )
            mlflow.log_input(dataset, context="training")
        except Exception as e:
            print(f"Warning: Could not log dataset: {e}")
        
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
            print(f"\n[Window {window_idx}] Computing graph features with cutoff: {train_end.date()}")
            # Re-engineer features with temporal filtering
            train_data = feature_engineering(train_data, cutoff_date=train_end)
            test_data = feature_engineering(test_data, cutoff_date=train_end)
            
            # Generate embeddings per window if callback provided (for hybrid models)
            window_embed_cols = []
            if embedding_generator is not None:
                print(f"[Window {window_idx}] Generating embeddings with temporal cutoff: {train_end.date()}")
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
            all_graph_cols = (
                GRAPH_FEATURE_COLUMNS + 
                ADVANCED_GRAPH_FEATURE_COLUMNS + 
                TIME_WEIGHTED_FEATURE_COLUMNS + 
                INTERACTION_FEATURE_COLUMNS +
                TEXT_FEATURE_COLUMNS
            )
            graph_columns = [col for col in all_graph_cols if col in train_data.columns]
            # Use window embeddings if generated, otherwise use pre-computed extra_features
            embed_features = window_embed_cols if window_embed_cols else extra_features
            features = get_base_features() + graph_columns + embed_features
            
            X_train = train_data.select(features).to_numpy()
            y_train = train_data.select(target).to_numpy().flatten()
            X_test = test_data.select(features).to_numpy()
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
                
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Manually log model with signature and input example
                signature = infer_signature(X_train, model.predict(X_train))
                input_example = X_train[:5]
                
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model",
                    signature=signature,
                    input_example=input_example
                )
                
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
                    print(f"Warning: MLflow evaluate failed: {e}")
                
                print(f"Window {window_idx}: Train={len(train_data):,}, "
                      f"Test={len(test_data):,}, AUC-PR={metrics['auc_pr']:.4f}")
                
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
            
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)
            print(f"Windows evaluated: {len(results)}")
            print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
            print(f"Best AUC-PR: {best_auc_pr:.4f}")
            
            # Register best model to Model Registry
            if best_model is not None and best_run_id is not None:
                print(f"\nRegistering best model (window with AUC-PR={best_auc_pr:.4f})...")
                
                try:
                    model_uri = f"runs:/{best_run_id}/model"
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name=f"fraud-detection-{model_name}"
                    )
                    
                    print(f"✓ Registered as: {registered_model.name} (version {registered_model.version})")
                    
                    # Add description to model version
                    client = mlflow.tracking.MlflowClient()
                    client.update_model_version(
                        name=registered_model.name,
                        version=registered_model.version,
                        description=f"Accumulating window training. Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}"
                    )
                    
                    mlflow.log_param("registered_model_version", registered_model.version)
                    
                except Exception as e:
                    print(f"Warning: Model registration failed: {e}")
        
        return {
            "run_id": parent_run.info.run_id,
            "results": results,
            "mean_auc_pr": mean_auc_pr if results else 0,
            "best_auc_pr": best_auc_pr,
            "num_windows": len(results),
        }

def run_baseline():
    """
    Train baseline XGBoost with accumulating window and MLflow tracking.
    
    Always enabled:
    - MLflow tracking
    - Model registration
    - Graph features with temporal filtering (computed on-the-fly per window)
    """
    print("Loading data...")
    df = load_data()
    df = _add_base_tabular_features(df)
    
    model_name = "baseline_graph"
    result = train_accumulating_window(df, model_name=model_name)
    
    return result


def main():
    """CLI entry point for baseline training."""
    run_baseline()

if __name__ == "__main__":
    main()
