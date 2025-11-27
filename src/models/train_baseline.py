import os
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature

from src.utils.metrics import calculate_metrics

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

def feature_engineering(df, include_graph_features: bool = False):
    """
    Creates tabular features for XGBoost.
    """
    print("Engineering features...")
    
    if include_graph_features:
        graph_features = load_graph_features()
        df = df.join(graph_features, on="insertion_id", how="left")
        all_graph_cols = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS + TIME_WEIGHTED_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS
        df = df.with_columns([
            pl.col(col).fill_null(0) for col in all_graph_cols if col in df.columns
        ])

    # 1. Account Age (The critical feature)
    # submission_at - account_created_at
    df = df.with_columns(
        (pl.col("submission_at") - pl.col("account_created_at")).dt.total_days().alias("account_age_days")
    )
    
    # 2. Price Normalization (Simple log)
    # XGBoost handles NaNs, so we don't need to fill_null(0) before log.
    # However, log(NaN) is NaN, which is fine. But log(0) is -inf.
    # If price is missing, we leave it as null.
    df = df.with_columns(
        pl.col("price_rent_gross").log1p().alias("log_price")
    )
    
    # 3. Text Length
    # We have description_embedding, but for baseline let's use length if available, else skip.
    
    # --- NEW FEATURES (Sync with GNN) ---
    
    # 4. Boolean features (Cast to Int)
    bool_cols = ["is_new", "has_balcony", "has_elevator", "has_parking"]
    bool_exprs = [
        pl.col(col).fill_null(False).cast(pl.Int8) if col in df.columns else pl.lit(0).alias(col)
        for col in bool_cols
    ]
    df = df.with_columns(bool_exprs)
            
    # 5. Bundle Info
    df = df.with_columns([
        pl.col("bundle_period").fill_null(7) if "bundle_period" in df.columns else pl.lit(7).alias("bundle_period"),
        (
            pl.col("bundle_tier").fill_null("basic").str.to_lowercase()
            .replace({"basic": 0, "premium": 1, "top": 2}, default=0)
            .cast(pl.Int64).alias("bundle_tier_score")
        ) if "bundle_tier" in df.columns else pl.lit(0).alias("bundle_tier_score")
    ])

    # 6. Payment Type (Binary: DIRECT vs INVOICE)
    df = df.with_columns(
        (pl.col("payment_type") == "DIRECT").cast(pl.Int8).alias("is_direct_payment")
        if "payment_type" in df.columns else pl.lit(0).alias("is_direct_payment")
    )
        
    # 7. Offer Type (Binary: BUY vs RENT)
    df = df.with_columns(
        (pl.col("offer_type") == "BUY").cast(pl.Int8).alias("is_buy")
        if "offer_type" in df.columns else pl.lit(0).alias("is_buy")
    )
        
    # 8. Location features
    location_exprs = [
        pl.col(col).fill_null(0.0) if col in df.columns else pl.lit(0.0).alias(col)
        for col in ["latitude", "longitude"]
    ]
    df = df.with_columns(location_exprs)

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


def train_expanding_window(df, initial_window_days=180, step_days=7, extra_features=None, model_name="baseline"):
    """
    Train with EXPANDING window (accumulating data) and MLflow tracking.
    
    Always enabled:
    - MLflow experiment tracking
    - Model saving to mlruns/models/
    - Automatic model registration to Model Registry
    
    Args:
        df: DataFrame with features and target
        initial_window_days: Initial training window size (default: 180 days)
        step_days: Step size for evaluation (default: 7 days)
        extra_features: Additional feature columns (e.g., embeddings)
        model_name: Model name for MLflow registry (default: "baseline")
    
    Returns:
        Dictionary with run info and metrics
    """
    if extra_features is None:
        extra_features = []
    
    print("="*70)
    print(f"EXPANDING WINDOW TRAINING - {model_name.upper()}")
    print("="*70)
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Build feature list
    all_graph_cols = (
        GRAPH_FEATURE_COLUMNS + 
        ADVANCED_GRAPH_FEATURE_COLUMNS + 
        TIME_WEIGHTED_FEATURE_COLUMNS + 
        INTERACTION_FEATURE_COLUMNS
    )
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    features = get_base_features() + graph_columns + extra_features
    target = "is_fraud"
    
    # Define window parameters  
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    print(f"Data range: {start_date.date()} to {end_date.date()}")
    print(f"Total listings: {len(df):,}")
    print(f"Features: {len(features)}")
    print(f"Initial window: {initial_window_days} days, Step: {step_days} days\n")
    
    # MLflow experiment setup
    experiment_name = "ppa-fraud-detection"
    mlflow.set_experiment(experiment_name)
    
    # Start parent run
    with mlflow.start_run(
        run_name=f"{model_name}_expanding_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"model_type": model_name, "training_mode": "expanding_window"}
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
        
        # Log dataset
        try:
            dataset = mlflow.data.from_pandas(
                df.to_pandas(), 
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
            
            # EXPANDING WINDOW: Use ALL data from start to train_end
            train_data = df.filter(pl.col("submission_at") < train_end)
            test_data = df.filter(
                (pl.col("submission_at") >= train_end) & 
                (pl.col("submission_at") < test_end)
            )
            
            if len(test_data) < 50 or len(train_data) < 1000:
                current_date += step_size
                continue
            
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
                
                # Enable autologging
                mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)
                
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
                
                # Use mlflow.evaluate for SHAP
                try:
                    eval_data = test_data.select(features).to_pandas()
                    eval_data["target"] = y_test
                    
                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    
                    mlflow.evaluate(
                        model=model_uri,
                        data=eval_data,
                        targets="target",
                        model_type="classifier",
                        evaluators=["default"],
                        evaluator_config={"log_explainer": True}
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
            results_path = f"mlruns/results/{model_name}_expanding_results.csv"
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
                        description=f"Expanding window training. Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}"
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

def run_baseline(include_graph_features: bool = True):
    """
    Train baseline XGBoost with expanding window and MLflow tracking.
    
    Always enabled:
    - MLflow tracking
    - Model registration
    - Graph features (optional)
    
    Args:
        include_graph_features: Whether to include graph features (default: True)
    """
    print("Loading data...")
    df = load_data()
    df = feature_engineering(df, include_graph_features=include_graph_features)
    
    model_name = "baseline" if not include_graph_features else "baseline_graph"
    result = train_expanding_window(df, model_name=model_name)
    
    return result


def main():
    """CLI entry point for baseline training."""
    run_baseline(include_graph_features=True)

if __name__ == "__main__":
    main()
