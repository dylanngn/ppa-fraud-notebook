"""
Graph Feature Window Optimization

Tests different time windows for graph feature computation to find optimal recency.

Hypothesis: Graph features computed on shorter/longer windows might perform better
than the default 90-day window.

Experiment 8: Window size optimization (30, 60, 90, 120, 180, 365 days)
"""
import os
from pathlib import Path
from datetime import timedelta

import polars as pl
import xgboost as xgb
import networkx as nx
import mlflow

from src.models.train_baseline import (
    load_data,
    get_base_features,
    GRAPH_FEATURE_COLUMNS,
    ADVANCED_GRAPH_FEATURE_COLUMNS
)
from src.utils.metrics import calculate_metrics


def compute_graph_features_with_window(window_days: int) -> pl.DataFrame:
    """
    Compute graph features using only connections within the specified window.
    
    Args:
        window_days: Number of days to look back for graph connections
    
    Returns:
        DataFrame with graph features computed on windowed graph
    """
    print(f"\nComputing graph features with {window_days}-day window...")
    
    # Load nodes
    listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    # Load edges
    email_edges = pl.read_parquet("artifacts/edges_listing_contact_email.parquet")
    phone_edges = pl.read_parquet("artifacts/edges_listing_contact_phone.parquet")
    
    # Filter edges based on time window
    # Only include edges where both listings are within window_days of each other
    cutoff_date = listings["submission_at"].max()
    start_date = cutoff_date - timedelta(days=window_days)
    
    windowed_listings = listings.filter(
        pl.col("submission_at") >= start_date
    )["insertion_id"]
    
    # Filter edges to only include listings in window
    email_edges_windowed = email_edges.filter(
        pl.col("source").is_in(windowed_listings)
    )
    phone_edges_windowed = phone_edges.filter(
        pl.col("source").is_in(windowed_listings)
    )
    
    print(f"  Listings in window: {len(windowed_listings):,}")
    print(f"  Email edges: {len(email_edges_windowed):,} (original: {len(email_edges):,})")
    print(f"  Phone edges: {len(phone_edges_windowed):,} (original: {len(phone_edges):,})")
    
    # Compute features on windowed graph
    features = []
    
    for listing_id in listings["insertion_id"]:
        # Email features
        email_count = email_edges_windowed.filter(pl.col("source") == listing_id).height
        
        # Phone features
        phone_count = phone_edges_windowed.filter(pl.col("source") == listing_id).height
        
        # Shared email count
        if email_count > 0:
            my_emails = email_edges_windowed.filter(
                pl.col("source") == listing_id
            )["target"].to_list()
            
            shared_email = email_edges_windowed.filter(
                pl.col("target").is_in(my_emails) &
                (pl.col("source") != listing_id)
            )["source"].n_unique()
        else:
            shared_email = 0
        
        # Shared phone count
        if phone_count > 0:
            my_phones = phone_edges_windowed.filter(
                pl.col("source") == listing_id
            )["target"].to_list()
            
            shared_phone = phone_edges_windowed.filter(
                pl.col("target").is_in(my_phones) &
                (pl.col("source") != listing_id)
            )["source"].n_unique()
        else:
            shared_phone = 0
        
        features.append({
            "insertion_id": listing_id,
            "contact_email_count": email_count,
            "shared_contact_email_count": shared_email,
            "contact_phone_count": phone_count,
            "shared_contact_phone_count": shared_phone,
        })
    
    return pl.DataFrame(features)


def test_window_size(window_days: int, n_folds: int = 5):
    """
    Test a specific window size with cross-validation.
    
    Args:
        window_days: Window size to test
        n_folds: Number of CV folds
    
    Returns:
        dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing {window_days}-day window")
    print(f"{'='*60}")
    
    # Start MLflow run for this window test
    mlflow.set_experiment("ppa-fraud-detection")
    with mlflow.start_run(
        run_name=f"window_optimization_{window_days}d",
        tags={"type": "window_optimization", "experiment": "exp8", "window_days": str(window_days)},
        nested=True
    ):
        # Load data
        df = load_data()
        
        # Compute windowed graph features
        graph_features = compute_graph_features_with_window(window_days)
        
        # Join with main data
        df = df.join(graph_features, on="insertion_id", how="left")
        
        # Fill nulls
        for col in ["contact_email_count", "shared_contact_email_count", 
                    "contact_phone_count", "shared_contact_phone_count"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(0))
        
        # Feature engineering (tabular only, we have custom graph features)
        from src.models.train_baseline import feature_engineering
        df = feature_engineering(df, include_graph_features=False)
        
        # Prepare features
        features = get_base_features() + [
            "contact_email_count", "shared_contact_email_count",
            "contact_phone_count", "shared_contact_phone_count"
        ]
        
        # Simple time-based split (last 20% as test)
        df = df.sort("submission_at")
        split_idx = int(len(df) * 0.8)
        
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        X_train = train_df.select(features).to_numpy()
        y_train = train_df.select("is_fraud").to_numpy().flatten()
        X_test = test_df.select(features).to_numpy()
        y_test = test_df.select("is_fraud").to_numpy().flatten()
        
        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Enable XGBoost autologging
        mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True, silent=True)
        
        # Log parameters
        mlflow.log_params({
            "window_days": window_days,
            "n_folds": n_folds,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(features),
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_test": float(y_test.mean()),
        })
        
        # Train model
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "auc_pr": metrics["auc_pr"],
            "auc_roc": metrics["auc_roc"],
            "p_at_100": metrics["p@100"],
            "lift_at_100": metrics["lift@100"],
        })
        
        print(f"\nResults:")
        print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  P@100:   {metrics['p@100']:.4f}")
        
        return {
            "window_days": window_days,
            **metrics
        }


def run_window_optimization():
    """
    Test multiple window sizes and find the optimal one.
    """
    print("="*60)
    print("GRAPH FEATURE WINDOW OPTIMIZATION")
    print("="*60)
    
    mlflow.set_experiment("ppa-fraud-detection")
    with mlflow.start_run(
        run_name="window_optimization_study",
        tags={"type": "window_optimization", "experiment": "exp8"}
    ):
        # Test different window sizes
        windows = [30, 60, 90, 120, 180, 365]
        
        mlflow.log_params({
            "experiment": "exp8",
            "window_sizes": str(windows),
            "n_windows_tested": len(windows),
        })
        
        results = []
        for window in windows:
            try:
                result = test_window_size(window)
                results.append(result)
            except Exception as e:
                print(f"Error with {window}-day window: {e}")
                continue
        
        # Save results
        os.makedirs("artifacts/results", exist_ok=True)
        results_df = pl.DataFrame(results)
        results_path = "artifacts/results/window_optimization_results.csv"
        results_df.write_csv(results_path)
        
        # Log results as artifact
        mlflow.log_artifact(results_path, artifact_path="results")
        
        # Find best
        if results:
            best = max(results, key=lambda x: x['auc_pr'])
            
            # Log best results to parent run
            mlflow.log_params({
                "best_window_days": best['window_days'],
            })
            mlflow.log_metrics({
                "best_auc_pr": best['auc_pr'],
                "best_auc_roc": best['auc_roc'],
                "best_p_at_100": best['p@100'],
            })
    
    # Summary
    print("\n" + "="*60)
    print("WINDOW OPTIMIZATION SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"{r['window_days']:3d} days: AUC-PR={r['auc_pr']:.4f}, P@100={r['p@100']:.4f}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x['auc_pr'])
        print(f"\n✅ Best window: {best['window_days']} days (AUC-PR={best['auc_pr']:.4f})")
    
    return results


if __name__ == "__main__":
    run_window_optimization()
