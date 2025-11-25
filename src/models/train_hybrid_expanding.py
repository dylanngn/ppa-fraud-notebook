"""
Expanding Window Training - Hybrid HGT

Trains HGT with ACCUMULATING data to enable complete graph structure.
"""
import os
from datetime import timedelta
from pathlib import Path

import torch
import polars as pl
import xgboost as xgb
import numpy as np

from src.models.train_baseline import get_base_features, GRAPH_FEATURE_COLUMNS, ADVANCED_GRAPH_FEATURE_COLUMNS
from src.utils.metrics import calculate_metrics


def train_hybrid_expanding(
    model_name: str = "hgt",
    window_days: int = 180,
    step_days: int = 7,
    save_models: bool = True
):
    """
    Train hybrid model with EXPANDING window.
    
    This gives HGT access to the COMPLETE graph at each step,
    allowing it to learn fraud rings that span across time.
    """
    print("="*60)
    print(f"EXPANDING WINDOW TRAINING - Hybrid {model_name.upper()}")
    print("="*60)
    
    # Load or build graph data
    graph_data_path = Path("artifacts/graph.pt")
    if not graph_data_path.exists():
        print("Graph data not found. Please run: python src/cli.py build-graph")
        raise FileNotFoundError(f"Missing {graph_data_path}. Run: make build-graph")
    
    print("Loading graph data...")
    graph_data = torch.load(graph_data_path, weights_only=False)
    
    # Load listing data
    df_listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    df = df_listings.join(df_users, on="user_id", how="left")
    
    # Feature engineering
    from src.models.train_baseline import feature_engineering
    df = feature_engineering(df, include_graph_features=True)
    df = df.sort("submission_at")
    
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    print(f"Data range: {start_date.date()} to {end_date.date()}")
    print(f"Total listings: {len(df):,}\n")
    
    # Initialize dates
    current_date = start_date + timedelta(days=window_days)
    test_size = timedelta(days=14)
    step_size = timedelta(days=step_days)
    
    # Build feature list
    all_graph_cols = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    tabular_features = get_base_features() + graph_columns
    
    results = []
    window_idx = 0
    
    # Load or train embeddings (using FULL graph)
    embeddings_path = Path(f"artifacts/embeddings_{model_name}.pt")
    if embeddings_path.exists():
        print(f"Loading pre-trained {model_name} embeddings...")
        embeddings = torch.load(embeddings_path, weights_only=False)
    else:
        print(f"{model_name} embeddings not found. Please run: python src/cli.py train-embeddings --model {model_name}")
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run: python src/cli.py train-embeddings --model {model_name}"
        )
    
    # Debug: Check embeddings structure
    print(f"Embeddings type: {type(embeddings)}")
    print(f"Embeddings keys: {embeddings.keys() if isinstance(embeddings, dict) else 'Not a dict'}")
    
    # Extract listing embeddings
    if isinstance(embeddings, dict) and 'listing' in embeddings:
        listing_embeddings = embeddings['listing'].cpu().numpy()
    elif isinstance(embeddings, torch.Tensor):
        # Embeddings is directly a tensor
        listing_embeddings = embeddings.cpu().numpy()
    else:
        raise ValueError(f"Unexpected embeddings structure: {type(embeddings)}")
    
    print(f"Loaded embeddings: {listing_embeddings.shape}")
    
    # Map insertion_id to embedding
    # Embeddings were created in the same order as listings in the graph
    # So we can use the df insertion_id order
    all_listing_ids = df["insertion_id"].to_list()
    
    # Create mapping (embeddings should be in same order as graph nodes)
    embedding_dict = {}
    for idx in range(min(len(all_listing_ids), len(listing_embeddings))):
        embedding_dict[all_listing_ids[idx]] = listing_embeddings[idx]
    
    print(f"Created embedding mapping for {len(embedding_dict)} listings")
    
    while current_date + test_size <= end_date:
        train_end = current_date
        test_end = current_date + test_size
        
        # EXPANDING WINDOW
        train_data = df.filter(pl.col("submission_at") < train_end)
        test_data = df.filter(
            (pl.col("submission_at") >= train_end) & 
            (pl.col("submission_at") < test_end)
        )
        
        if len(test_data) == 0 or len(train_data) < 1000:
            current_date += step_size
            continue
        
        print(f"\nWindow {window_idx}:")
        print(f"  Train: {start_date.date()} → {train_end.date()} ({len(train_data):,} rows)")
        print(f"  Test:  {train_end.date()} → {test_end.date()} ({len(test_data):,} rows)")
        
        # Get features + embeddings
        X_train_tabular = train_data.select(tabular_features).to_numpy()
        X_test_tabular = test_data.select(tabular_features).to_numpy()
        
        # Add embeddings
        train_ids = train_data["insertion_id"].to_list()
        test_ids = test_data["insertion_id"].to_list()
        
        X_train_emb = np.array([embedding_dict.get(iid, np.zeros(64)) for iid in train_ids])
        X_test_emb = np.array([embedding_dict.get(iid, np.zeros(64)) for iid in test_ids])
        
        X_train = np.hstack([X_train_tabular, X_train_emb])
        X_test = np.hstack([X_test_tabular, X_test_emb])
        
        y_train = train_data.select("is_fraud").to_numpy().flatten()
        y_test = test_data.select("is_fraud").to_numpy().flatten()
        
        # Train XGBoost with embeddings
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
        
        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = calculate_metrics(y_test, proba)
        
        print(f"  Results: AUC-PR={metrics['auc_pr']:.4f}, P@100={metrics['p@100']:.4f}")
        
        results.append({
            "window_idx": window_idx,
            "window_start": train_end,
            "train_size": len(train_data),
            "test_size": len(test_data),
            **metrics
        })
        
        window_idx += 1
        current_date += step_size
    
    # Save results
    os.makedirs("artifacts/results", exist_ok=True)
    results_df = pl.DataFrame(results)
    results_df.write_csv(f"artifacts/results/hybrid_{model_name}_expanding_results.csv")
    
    print("\n" + "="*60)
    print("EXPANDING WINDOW TRAINING COMPLETE")
    print("="*60)
    print(f"Total windows: {len(results)}")
    print(f"Mean AUC-PR: {float(results_df['auc_pr'].mean()):.4f}")
    print(f"Mean AUC-ROC: {float(results_df['auc_roc'].mean()):.4f}")
    print(f"Mean P@100: {float(results_df['p@100'].mean()):.4f}")
    print(f"Results saved to: artifacts/results/hybrid_{model_name}_expanding_results.csv")
    
    return results


def main(model_name: str = "hgt", window_days: int = 180, step_days: int = 7):
    train_hybrid_expanding(model_name=model_name, window_days=window_days, step_days=step_days)


if __name__ == "__main__":
    main()
