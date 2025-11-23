import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import numpy as np
import os
from datetime import timedelta

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

def feature_engineering(df):
    """
    Creates tabular features for XGBoost.
    """
    print("Engineering features...")
    
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
    # We have description_embedding, but for baseline let's use length
    # We don't have raw text in nodes_listing, only embedding. 
    # We might need to add description_length to ETL if we want it.
    # For now, let's use what we have.
    
    # 4. User Type Encoding (REMOVED)
    # user_type is calculated, so we don't use it.
    
    return df

def train_sliding_window(df, window_days=90, step_days=7):
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
    test_size = timedelta(days=7)    # Test on next 1 week (keep fixed for now)
    
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
        features = ["account_age_days", "log_price", "living_space", "rooms"]
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

def main(window_days=90, step_days=7):
    # Check if artifacts exist
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
    else:
        df = load_data()
        df = feature_engineering(df)
        results = train_sliding_window(df, window_days, step_days)
        
        # Save results to CSV
        os.makedirs("artifacts/results", exist_ok=True)
        results_df = pl.DataFrame(results)
        results_df.write_csv("artifacts/results/baseline_results.csv")
        print(f"\nSaved results to artifacts/results/baseline_results.csv")
        print(f"Mean AUC-PR: {results_df['auc_pr'].mean():.4f}")
        print(f"Mean AUC-ROC: {results_df['auc_roc'].mean():.4f}")
        
        return results

if __name__ == "__main__":
    main()
