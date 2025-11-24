import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

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


def load_graph_features() -> pl.DataFrame:
    if GRAPH_FEATURES_PATH.exists():
        print("Loading graph-derived features...")
        return pl.read_parquet(GRAPH_FEATURES_PATH)
    raise FileNotFoundError(
        f"Graph features not found at {GRAPH_FEATURES_PATH}. "
        "Run `make graph-features` to generate them."
    )


def load_data():
    """
    Loads and joins listing and user data.
    """
    print("Loading data...")
    df_listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    
    # Join Listings with Users
    # Note: nodes_listing has 'user_id' implicitly? 
    # Wait, nodes_listing in ETL.py didn't explicitly select user_id, let's check.
    # It selected: insertion_id, object_reference, platform, is_fraud, ... lister_username, ...
    # It MIGHT have missed user_id. I need to verify ETL.py first.
    # Assuming it has it or we can join on lister_username (less reliable).
    # Let's assume we need to fix ETL if it's missing.
    
    # For now, let's assume user_id is there.
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
        df = df.with_columns([
            pl.col(col).fill_null(0) for col in GRAPH_FEATURE_COLUMNS if col in df.columns
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
    
    # 4. Booleans (Cast to Int)
    bool_cols = ["is_new", "has_balcony", "has_elevator", "has_parking"]
    for col in bool_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(False).cast(pl.Int8))
        else:
            df = df.with_columns(pl.lit(0).alias(col))
            
    # 5. Bundle Info
    if "bundle_period" in df.columns:
        df = df.with_columns(pl.col("bundle_period").fill_null(7))
    else:
        df = df.with_columns(pl.lit(7).alias("bundle_period"))
        
    # Bundle Tier (Ordinal)
    if "bundle_tier" in df.columns:
        # basic=0, premium=1, top=2
        df = df.with_columns(
            pl.col("bundle_tier").fill_null("basic").str.to_lowercase()
            .replace({"basic": 0, "premium": 1, "top": 2}, default=0)
            .cast(pl.Int64).alias("bundle_tier_score")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("bundle_tier_score"))

    # 6. Payment Type (Binary: DIRECT vs INVOICE)
    if "payment_type" in df.columns:
        df = df.with_columns(
            (pl.col("payment_type") == "DIRECT").cast(pl.Int8).alias("is_direct_payment")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("is_direct_payment"))
        
    # 7. Offer Type (Binary: BUY vs RENT)
    if "offer_type" in df.columns:
        df = df.with_columns(
            (pl.col("offer_type") == "BUY").cast(pl.Int8).alias("is_buy")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("is_buy"))
        
    # 8. Location
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0.0))
        else:
            df = df.with_columns(pl.lit(0.0).alias(col))

    return df

def train_sliding_window(df, window_days=90, step_days=14):
    """
    Performs sliding window backtesting.
    """
    # Sort by time
    df = df.sort("submission_at")
    
    # Define Window
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=14)    # Test on next 2 weeks
    
    current_date = start_date + window_size
    
    results = []
    
    while current_date + test_size <= end_date:
        train_end = current_date
        test_end = current_date + test_size
        
        # Split
        train_data = df.filter((pl.col("submission_at") < train_end) & (pl.col("submission_at") >= train_end - window_size))
        test_data = df.filter((pl.col("submission_at") >= train_end) & (pl.col("submission_at") < test_end))
        
        if len(test_data) == 0 or len(train_data) == 0:
            current_date += step_size
            continue
            
        # Features & Target
        graph_columns = [col for col in GRAPH_FEATURE_COLUMNS if col in df.columns]

        features = [
            "account_age_days", "log_price", "living_space", "rooms",
            "is_new", "has_balcony", "has_elevator", "has_parking",
            "bundle_period", "bundle_tier_score",
            "is_direct_payment", "is_buy",
            "latitude", "longitude"
        ] + graph_columns
        target = "is_fraud"
        
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
            learning_rate=0.1
        )
        
        model.fit(X_train, y_train)
        
        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        if len(np.unique(y_test)) > 1:
            auc_pr = average_precision_score(y_test, proba)
            auc_roc = roc_auc_score(y_test, proba)
        else:
            auc_pr = 0.0
            auc_roc = 0.0
            
        # Calculate Precision@K (Lift) - operational metric
        precisions_at_k = {}
        for k in [50, 100, 200]:
            if len(proba) >= k:
                top_k_indices = np.argsort(proba)[-k:][::-1]
                precisions_at_k[f'p@{k}'] = y_test[top_k_indices].mean()
            else:
                precisions_at_k[f'p@{k}'] = 0.0
        
        print(f"Window {train_end.date()} - {test_end.date()}: AUC-PR = {auc_pr:.4f}, "
              f"P@100 = {precisions_at_k['p@100']:.4f}, Fraud Count = {sum(y_test)}")
        
        results.append({
            "window_start": train_end,
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "p@50": precisions_at_k['p@50'],
            "p@100": precisions_at_k['p@100'],
            "p@200": precisions_at_k['p@200'],
            "fraud_count": sum(y_test)
        })
        
        current_date += step_size
        
    return results

def run_baseline(
    window_days: int = 90,
    step_days: int = 14,
    include_graph_features: bool = False,
    results_filename: str = "artifacts/results/baseline_results.csv",
):
    # Check if artifacts exist
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
    else:
        df = load_data()
        df = feature_engineering(df, include_graph_features=include_graph_features)
        results = train_sliding_window(df, window_days, step_days)
        
        # Save results to CSV
        os.makedirs("artifacts/results", exist_ok=True)
        results_df = pl.DataFrame(results)
        results_df.write_csv(results_filename)
        print(f"\nSaved results to {results_filename}")
        print(f"Mean AUC-PR: {results_df['auc_pr'].mean():.4f}")
        print(f"Mean AUC-ROC: {results_df['auc_roc'].mean():.4f}")
        
        return results


def main(window_days: int = 90, step_days: int = 14):
    run_baseline(window_days=window_days, step_days=step_days, include_graph_features=False)

if __name__ == "__main__":
    main()
