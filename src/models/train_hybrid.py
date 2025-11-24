import polars as pl
import xgboost as xgb
import torch
import numpy as np
import os
from src.models.train_baseline import load_data, feature_engineering, train_sliding_window

def load_embeddings(df, model_name="hgt"):
    """
    Loads GNN embeddings and merges them with the DataFrame.
    """
    embedding_path = f"artifacts/embeddings_{model_name}.pt"
    print(f"Loading Embeddings from {embedding_path}...")
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embeddings not found at {embedding_path}. Run train_embeddings.py --model {model_name} first.")
        
    # Load Tensor
    embeddings = torch.load(embedding_path).numpy()
    
    # We need to ensure alignment.
    # train_embeddings.py loaded 'nodes_listing.parquet' and used it in that order.
    # So embeddings[i] corresponds to the i-th row in nodes_listing.parquet.
    # We must load nodes_listing.parquet again and attach embeddings by index.
    
    df_listing = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    # Create a DataFrame of embeddings
    # Column names: embed_0, embed_1, ...
    embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
    df_embed = pl.DataFrame(embeddings, schema=embed_cols)
    
    # Horizontally stack (polars hstack or just with_columns if lengths match)
    if len(df_listing) != len(df_embed):
        raise ValueError(f"Mismatch: Listings {len(df_listing)} vs Embeddings {len(df_embed)}")
        
    df_listing = df_listing.hstack(df_embed)
    
    # Now we join this enriched listing DF with users, just like in baseline
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    df = df_listing.join(df_users, on="user_id", how="left")
    
    return df, embed_cols

def main(model_name="hgt"):
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
    else:
        # 1. Load Data & Embeddings
        df, embed_cols = load_embeddings(None, model_name)
        
        # 2. Feature Engineering (Tabular)
        df = feature_engineering(df)
        
        # 3. Train (Sliding Window)
        # We need to modify train_sliding_window to include embed_cols in 'features'
        # Since train_sliding_window hardcodes features, we might need to override it
        # or just copy the function here to add the new features.
        # Let's copy-paste for safety and flexibility.
        
        from datetime import timedelta
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        # --- Modified Sliding Window ---
        df = df.sort("submission_at")
        start_date = df["submission_at"].min()
        end_date = df["submission_at"].max()
        
        window_size = timedelta(days=90)
        step_size = timedelta(days=14)
        test_size = timedelta(days=14)
        
        current_date = start_date + window_size
        
        results = []
        
        # Define Features: Original + Embeddings
        # Define Features: Original + Embeddings
        base_features = [
            "account_age_days", "log_price", "living_space", "rooms",
            "is_new", "has_balcony", "has_elevator", "has_parking",
            "bundle_period", "bundle_tier_score",
            "is_direct_payment", "is_buy",
            "latitude", "longitude"
        ]
        features = base_features + embed_cols
        target = "is_fraud"
        
        print(f"Training with {len(features)} features ({len(base_features)} tabular + {len(embed_cols)} embedding)...")
        
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
            y_pred = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            if len(np.unique(y_test)) > 1:
                auc_pr = average_precision_score(y_test, y_pred)
                auc_roc = roc_auc_score(y_test, y_pred)
            else:
                auc_pr = 0.0
                auc_roc = 0.0
                
            print(f"Window {train_end.date()} - {test_end.date()}: AUC-PR = {auc_pr:.4f}, Fraud Count = {sum(y_test)}")
            
            results.append({
                "window_start": train_end,
                "auc_pr": auc_pr,
                "auc_roc": auc_roc,
                "fraud_count": sum(y_test)
            })
            
            current_date += step_size
            
        # Save results to CSV
        os.makedirs("artifacts/results", exist_ok=True)
        results_df = pl.DataFrame(results)
        results_df.write_csv(f"artifacts/results/hybrid_{model_name}_results.csv")
        print(f"\nSaved results to artifacts/results/hybrid_{model_name}_results.csv")
        print(f"Mean AUC-PR: {results_df['auc_pr'].mean():.4f}")

if __name__ == "__main__":
    import typer
    typer.run(main)
