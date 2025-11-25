"""
Continuous Learning - Production Retraining

This is DIFFERENT from backtesting:
- Backtesting: 90-day sliding window (temporal validation)
- Production: Accumulate ALL historical data (maximize learning)
"""
import os
from datetime import datetime
from pathlib import Path
import polars as pl
import xgboost as xgb
import pickle

from src.models.train_baseline import feature_engineering, get_base_features


GRAPH_FEATURE_COLUMNS = [
    "contact_email_count", "shared_contact_email_count", "max_shared_contact_email",
    "contact_phone_count", "shared_contact_phone_count", "max_shared_contact_phone",
    "user_listing_count", "user_unique_ip_count", "shared_ip_user_count",
    "max_shared_ip_users", "listing_component_size", "listing_pagerank"
]

ADVANCED_GRAPH_FEATURE_COLUMNS = [
    "degree_total", "is_isolated", "unique_identifier_count",
    "neighbor_overlap_score", "avg_neighbor_degree"
]


def load_historical_data(data_path: str = "s3://fraud-training/historical/2023-2025.parquet") -> pl.DataFrame:
    """Load initial training data (2 years)."""
    print(f"Loading historical data from {data_path}")
    return pl.read_parquet(data_path)


def load_production_data(start_date: str, end_date: str = None) -> pl.DataFrame:
    """
    Load production data with labels from human reviews.
    
    Args:
        start_date: Start date (e.g., "2025-02-01")
        end_date: End date (default: today)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Loading production data from {start_date} to {end_date}")
    
    # This would query your Aurora DB or S3 for production labels
    # For now, placeholder:
    production_files = Path("s3://fraud-training/production/").glob(f"{start_date[:4]}*.parquet")
    
    if not production_files:
        print("No production data found")
        return pl.DataFrame()
    
    dfs = [pl.read_parquet(f) for f in production_files]
    return pl.concat(dfs)


def train_production_model(
    output_path: str = "artifacts/models/production/model_latest.pkl",
    include_production_data: bool = True
) -> dict:
    """
    Train production model with ACCUMULATING data.
    
    Unlike backtesting (90-day window), this uses ALL available data.
    
    Returns:
        dict with metrics and model path
    """
    # 1. Load historical data (2 years)
    historical = load_historical_data()
    print(f"Historical data: {len(historical):,} rows")
    
    # 2. Load production data (accumulated since launch)
    if include_production_data:
        production = load_production_data(start_date="2025-02-01")
        print(f"Production data: {len(production):,} rows")
        
        # Combine
        combined = pl.concat([historical, production])
        print(f"Combined training data: {len(combined):,} rows")
    else:
        combined = historical
    
    # 3. Engineer features
    combined = feature_engineering(combined, include_graph_features=True)
    
    # 4. Prepare features
    all_graph_cols = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS
    graph_columns = [col for col in all_graph_cols if col in combined.columns]
    features = get_base_features() + graph_columns
    target = "is_fraud"
    
    # 5. Split: Use last 10% as validation
    n = len(combined)
    train_size = int(n * 0.9)
    
    train_df = combined[:train_size]
    val_df = combined[train_size:]
    
    X_train = train_df.select(features).to_numpy()
    y_train = train_df.select(target).to_numpy().flatten()
    X_val = val_df.select(features).to_numpy()
    y_val = val_df.select(target).to_numpy().flatten()
    
    print(f"\nTraining set: {len(X_train):,} rows")
    print(f"Validation set: {len(X_val):,} rows")
    print(f"Fraud rate: {y_train.mean():.2%}")
    
    # 6. Train model
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        n_estimators=200,  # More trees for larger dataset
        max_depth=6,
        learning_rate=0.05,  # Lower LR for stability
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    # 7. Evaluate
    from src.utils.metrics import calculate_metrics
    
    y_pred = model.predict_proba(X_val)[:, 1]
    metrics = calculate_metrics(y_val, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Production Model Evaluation:")
    print(f"  AUC-PR:     {metrics['auc_pr']:.4f}")
    print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"  P@100:      {metrics['p@100']:.4f}")
    print(f"  Lift@100:   {metrics['lift@100']:.2f}")
    print(f"  Fraud Count: {metrics['fraud_count']}")
    print(f"{'='*50}\n")
    
    # 8. Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model_bundle = {
        'model': model,
        'features': features,
        'metrics': metrics,
        'training_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'historical_data_size': len(historical),
            'production_data_size': len(production) if include_production_data else 0,
            'trained_at': datetime.now().isoformat(),
            'feature_version': 'v2.1'
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"✅ Model saved to {output_path}")
    
    return {
        'metrics': metrics,
        'model_path': output_path,
        'training_rows': len(X_train)
    }


def compare_models(current_model_path: str, new_model_path: str) -> bool:
    """
    Compare new model vs current production model.
    
    Returns True if new model is better.
    """
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    # Load models
    with open(current_model_path, 'rb') as f:
        current = pickle.load(f)
    
    with open(new_model_path, 'rb') as f:
        new = pickle.load(f)
    
    current_auc = current['metrics']['auc_pr']
    new_auc = new['metrics']['auc_pr']
    
    improvement = new_auc - current_auc
    improvement_pct = (improvement / current_auc) * 100
    
    print(f"Current Model: AUC-PR = {current_auc:.4f}")
    print(f"New Model:     AUC-PR = {new_auc:.4f}")
    print(f"Improvement:   {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    # Deploy if improvement > 1%
    threshold = 0.01
    should_deploy = improvement > threshold
    
    if should_deploy:
        print(f"\n✅ NEW MODEL IS BETTER (>{threshold:.2%} improvement)")
        print("   → Recommend deployment")
    else:
        print(f"\n❌ NEW MODEL NOT SIGNIFICANTLY BETTER")
        print(f"   → Keep current model")
    
    print("="*50 + "\n")
    
    return should_deploy


if __name__ == "__main__":
    # Example usage
    result = train_production_model()
    
    # Compare with current
    current_model = "artifacts/models/production/model_current.pkl"
    new_model = result['model_path']
    
    if os.path.exists(current_model):
        should_deploy = compare_models(current_model, new_model)
        
        if should_deploy:
            print("Deploying new model...")
            # Copy new model to production location
            import shutil
            shutil.copy(new_model, current_model)
            print("✅ Deployment complete!")
    else:
        print("No current model found - this is the first deployment")
