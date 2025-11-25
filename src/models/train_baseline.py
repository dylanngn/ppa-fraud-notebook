import os
import pickle
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

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
        all_graph_cols = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS
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


def train_sliding_window(df, window_days=90, step_days=14, extra_features=None, save_models=True, models_dir="artifacts/models"):
    """
    Performs sliding window backtesting.
    
    Args:
        df: DataFrame with features and target
        window_days: Size of training window in days
        step_days: Step size for sliding window in days
        extra_features: Additional feature column names to include (e.g., embeddings, graph features)
        save_models: Whether to save trained models for SHAP analysis
        models_dir: Directory to save models
    """
    if extra_features is None:
        extra_features = []
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Define Window
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=14)
    
    current_date = start_date + window_size
    
    results = []
    saved_models = []
    
    # Build feature list
    all_graph_cols = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    features = get_base_features() + graph_columns + extra_features
    target = "is_fraud"
    
    if save_models:
        os.makedirs(models_dir, exist_ok=True)
    
    window_idx = 0
    while current_date + test_size <= end_date:
        train_end = current_date
        test_end = current_date + test_size
        
        # Split
        train_data = df.filter((pl.col("submission_at") < train_end) & (pl.col("submission_at") >= train_end - window_size))
        test_data = df.filter((pl.col("submission_at") >= train_end) & (pl.col("submission_at") < test_end))
        
        if len(test_data) == 0 or len(train_data) == 0:
            current_date += step_size
            continue
            
        X_train = train_data.select(features).to_numpy()
        y_train = train_data.select(target).to_numpy().flatten()
        X_test = test_data.select(features).to_numpy()
        y_test = test_data.select(target).to_numpy().flatten()
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate using unified metrics
        metrics = calculate_metrics(y_test, proba)
        
        print(f"Window {train_end.date()} - {test_end.date()}: "
              f"AUC-PR = {metrics['auc_pr']:.4f}, "
              f"P@100 = {metrics['p@100']:.4f}, "
              f"Lift@100 = {metrics['lift@100']:.2f}, "
              f"Fraud Count = {metrics['fraud_count']}")
        
        results.append({
            "window_start": train_end,
            **metrics
        })
        
        # Save model if requested
        if save_models:
            model_bundle = {
                'model': model,
                'features': features,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': proba,
                'window_info': {
                    'window_idx': window_idx,
                    'window_start': train_end,
                    'window_end': test_end,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                },
                'metrics': metrics
            }
            
            model_path = os.path.join(models_dir, f"model_window_{window_idx}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_bundle, f)
            
            saved_models.append(model_path)
        
        window_idx += 1
        current_date += step_size
    
    if save_models:
        return results, saved_models
    return results

def run_baseline(
    window_days: int = 90,
    step_days: int = 14,
    include_graph_features: bool = False,
    results_filename: str = "artifacts/results/baseline_results.csv",
    save_models: bool = True,
    models_dir: str = "artifacts/models/baseline"
):
    # Check if artifacts exist
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
    else:
        df = load_data()
        df = feature_engineering(df, include_graph_features=include_graph_features)
        
        result = train_sliding_window(df, window_days, step_days, save_models=save_models, models_dir=models_dir)
        
        if save_models:
            results, saved_model_paths = result
            print(f"\nSaved {len(saved_model_paths)} models to {models_dir}")
        else:
            results = result
        
        # Save results to CSV
        os.makedirs("artifacts/results", exist_ok=True)
        results_df = pl.DataFrame(results)
        results_df.write_csv(results_filename)
        print(f"\nSaved results to {results_filename}")
        print(f"Mean AUC-PR: {float(results_df['auc_pr'].mean()):.4f}")
        print(f"Mean AUC-ROC: {float(results_df['auc_roc'].mean()):.4f}")
        print(f"Mean P@100: {float(results_df['p@100'].mean()):.4f}")
        print(f"Mean Lift@100: {float(results_df['lift@100'].mean()):.2f}")
        
        return results


def main(window_days: int = 90, step_days: int = 14):
    run_baseline(window_days=window_days, step_days=step_days, include_graph_features=False)

if __name__ == "__main__":
    main()
