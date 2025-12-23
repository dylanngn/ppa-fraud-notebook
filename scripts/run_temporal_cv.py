"""
5-Fold Temporal Cross-Validation for Thesis Defense

This script runs temporal cross-validation with a 7-day gap between train/test splits
to validate model performance stability across different temporal partitions.

Usage:
    python scripts/run_temporal_cv.py --model vanilla  # Vanilla XGBoost only
    python scripts/run_temporal_cv.py --model gnn      # GNN+XGBoost only
    python scripts/run_temporal_cv.py --model both     # Both models (default)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import mlflow

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score


def load_data():
    """Load merged events data"""
    data_path = Path("artifacts/merged_events.parquet")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df


def create_temporal_folds(df, n_splits=5, gap_days=7):
    """
    Create temporal folds with gap between train/test.

    Returns: list of (train_idx, test_idx) tuples
    """
    # Sort by date
    df_sorted = df.sort_values('date').reset_index(drop=True)
    total_samples = len(df_sorted)

    # Calculate fold size
    fold_size = total_samples // (n_splits + 1)
    gap_samples = int(gap_days * fold_size / 30)  # Approximate gap in samples

    folds = []
    for i in range(n_splits):
        # Progressive train size
        train_end_idx = fold_size * (i + 1)
        test_start_idx = train_end_idx + gap_samples
        test_end_idx = min(test_start_idx + fold_size, total_samples)

        if test_end_idx > test_start_idx:
            train_idx = np.arange(0, train_end_idx)
            test_idx = np.arange(test_start_idx, test_end_idx)
            folds.append((train_idx, test_idx))

    return folds


def train_vanilla_xgboost(X_train, y_train, X_test, y_test):
    """Train vanilla XGBoost model"""
    from xgboost import XGBClassifier

    # Best hyperparameters from HPO
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.085,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=10,
        random_state=42,
        eval_metric='aucpr'
    )

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return evaluate(y_test, y_pred_proba)


def train_gnn_xgboost(X_train, y_train, X_test, y_test, graph_train, graph_test):
    """Train GNN+XGBoost model"""
    # Note: This requires graph construction which is expensive
    # For now, return placeholder
    print("  [GNN training skipped - requires graph construction]")
    return {
        'auc_pr': np.nan,
        'auc_roc': np.nan,
        'precision': np.nan,
        'recall': np.nan
    }


def evaluate(y_true, y_pred_proba):
    """Compute evaluation metrics"""
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        'auc_pr': average_precision_score(y_true, y_pred_proba),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }


def prepare_features(df):
    """Extract features for XGBoost"""
    # Get numeric columns only (simplified)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target and ID columns
    feature_cols = [col for col in numeric_cols
                   if col not in ['is_fraud', 'LISTINGID', 'INSERTION_ID', 'fraud_score']]

    X = df[feature_cols].fillna(0)
    y = df['is_fraud'].fillna(0).astype(int)

    return X, y


def main():
    parser = argparse.ArgumentParser(description='Run temporal cross-validation')
    parser.add_argument('--model', choices=['vanilla', 'gnn', 'both'], default='both',
                       help='Which model(s) to evaluate')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds (default: 5)')
    parser.add_argument('--gap_days', type=int, default=7,
                       help='Gap between train/test in days (default: 7)')
    args = parser.parse_args()

    print("=== Temporal Cross-Validation for Thesis Defense ===\n")
    print(f"Configuration:")
    print(f"  Model(s): {args.model}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Gap: {args.gap_days} days\n")

    # Load data
    print("Loading data...")
    df = load_data()
    print(f"Total events: {len(df):,}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%\n")

    # Create folds
    print("Creating temporal folds...")
    folds = create_temporal_folds(df, n_splits=args.n_folds, gap_days=args.gap_days)
    print(f"Created {len(folds)} folds\n")

    # Prepare features
    X, y = prepare_features(df)
    print(f"Features: {X.shape[1]}\n")

    # Run CV for each model
    results = {'vanilla': [], 'gnn': []}

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"=== Fold {fold_idx}/{len(folds)} ===")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"  Train: {len(X_train):,} samples ({y_train.mean()*100:.2f}% fraud)")
        print(f"  Test: {len(X_test):,} samples ({y_test.mean()*100:.2f}% fraud)")

        # Train vanilla XGBoost
        if args.model in ['vanilla', 'both']:
            print(f"  Training Vanilla XGBoost...")
            metrics = train_vanilla_xgboost(X_train, y_train, X_test, y_test)
            results['vanilla'].append(metrics)
            print(f"    AUC-PR: {metrics['auc_pr']:.4f}")

        # Train GNN+XGBoost
        if args.model in ['gnn', 'both']:
            print(f"  Training GNN+XGBoost...")
            # Requires graph construction - expensive
            metrics = train_gnn_xgboost(X_train, y_train, X_test, y_test, None, None)
            results['gnn'].append(metrics)
            if not np.isnan(metrics['auc_pr']):
                print(f"    AUC-PR: {metrics['auc_pr']:.4f}")

        print()

    # Summarize results
    print("=== Cross-Validation Results ===\n")

    for model_name, metrics_list in results.items():
        if not metrics_list:
            continue

        df_results = pd.DataFrame(metrics_list)

        print(f"{model_name.upper()} XGBOOST:")
        for metric in ['auc_pr', 'auc_roc', 'precision', 'recall']:
            values = df_results[metric].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f} (n={len(values)})")

        print(f"\n  Fold-wise {model_name.upper()} AUC-PR:")
        for i, metrics in enumerate(metrics_list, 1):
            if not np.isnan(metrics['auc_pr']):
                print(f"    Fold {i}: {metrics['auc_pr']:.4f}")
        print()

    # Compare models
    if args.model == 'both':
        vanilla_scores = [m['auc_pr'] for m in results['vanilla'] if not np.isnan(m['auc_pr'])]
        gnn_scores = [m['auc_pr'] for m in results['gnn'] if not np.isnan(m['auc_pr'])]

        if vanilla_scores and gnn_scores and len(vanilla_scores) == len(gnn_scores):
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(gnn_scores, vanilla_scores)

            print("=== Statistical Comparison ===")
            print(f"Paired t-test (GNN vs Vanilla):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")

    print("\n=== CV Completed ===")
    print("Results saved to console. Redirect to file for permanent storage:")
    print("  python scripts/run_temporal_cv.py > cv_results.txt")


if __name__ == "__main__":
    main()
