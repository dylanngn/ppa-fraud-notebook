"""
Expanding Window Training - Baseline XGBoost

Trains with ACCUMULATING data to simulate production continuous learning.
This is different from sliding window backtesting.
"""
import os
from datetime import timedelta
from pathlib import Path

import polars as pl
import xgboost as xgb

from src.models.train_baseline import (
    load_data, 
    feature_engineering, 
    get_base_features,
    GRAPH_FEATURE_COLUMNS,
    ADVANCED_GRAPH_FEATURE_COLUMNS
)
from src.utils.metrics import calculate_metrics


def train_expanding_window(
    window_days: int = 180,  # Initial window (6 months)
    step_days: int = 7,
    save_models: bool = True,
    models_dir: str = "artifacts/models/baseline_expanding"
):
    """
    Train baseline with EXPANDING window (accumulating data).
    
    Unlike sliding window (90 days), this uses ALL historical data,
    simulating production continuous learning.
    
    Args:
        window_days: Initial training window (default: 180 days)
        step_days: Step size for evaluation
        save_models: Whether to save models
        models_dir: Where to save models
    """
    print("="*60)
    print("EXPANDING WINDOW TRAINING - Baseline XGBoost")
    print("="*60)
    print(f"Initial window: {window_days} days")
    print(f"Step size: {step_days} days")
    print(f"Model saves to: {models_dir}\n")
    
    # Load and prepare data
    df = load_data()
    df = feature_engineering(df, include_graph_features=True)
    df = df.sort("submission_at")
    
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    print(f"Data range: {start_date.date()} to {end_date.date()}")
    print(f"Total listings: {len(df):,}\n")
    
    # Initial training point
    current_date = start_date + timedelta(days=window_days)
    test_size = timedelta(days=14)
    step_size = timedelta(days=step_days)
    
    # Build feature list
    all_graph_cols = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    features = get_base_features() + graph_columns
    target = "is_fraud"
    
    if save_models:
        os.makedirs(models_dir, exist_ok=True)
    
    results = []
    window_idx = 0
    
    while current_date + test_size <= end_date:
        train_end = current_date
        test_end = current_date + test_size
        
        # EXPANDING WINDOW: Use ALL data from start to train_end
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
        
        X_train = train_data.select(features).to_numpy()
        y_train = train_data.select(target).to_numpy().flatten()
        X_test = test_data.select(features).to_numpy()
        y_test = test_data.select(target).to_numpy().flatten()
        
        # Train XGBoost
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
        
        # Save model
        if save_models:
            import pickle
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
        
        window_idx += 1
        current_date += step_size
    
    # Save results
    os.makedirs("artifacts/results", exist_ok=True)
    results_df = pl.DataFrame(results)
    results_df.write_csv("artifacts/results/baseline_expanding_results.csv")
    
    print("\n" + "="*60)
    print("EXPANDING WINDOW TRAINING COMPLETE")
    print("="*60)
    print(f"Total windows: {len(results)}")
    print(f"Mean AUC-PR: {float(results_df['auc_pr'].mean()):.4f}")
    print(f"Mean AUC-ROC: {float(results_df['auc_roc'].mean()):.4f}")
    print(f"Mean P@100: {float(results_df['p@100'].mean()):.4f}")
    print(f"Results saved to: artifacts/results/baseline_expanding_results.csv")
    
    return results


def main(window_days: int = 180, step_days: int = 7):
    train_expanding_window(window_days=window_days, step_days=step_days)


if __name__ == "__main__":
    main()
