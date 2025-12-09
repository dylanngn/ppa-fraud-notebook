"""
Experiment 10A: Unsupervised Anomaly Detection

Research Gap: Supervisor feedback (Section 2.1) requires comparison with unsupervised methods.

Methods:
    - Isolation Forest (HIGH priority)
    - One-Class SVM (HIGH priority)
    - Local Outlier Factor (MEDIUM priority)

Run with:
    python -m src.experiments.exp10_anomaly_detection

Output:
    - artifacts/unsupervised/anomaly_detection_results.csv
    - artifacts/unsupervised/anomaly_comparison.png
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from src.models.utils.common import setup_mlflow
from src.utils.hydra_utils import resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
OUTPUT_DIR = ARTIFACTS_DIR / "unsupervised"
RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"

# Constants
FRAUD_RATE = 0.082  # ~8.2% fraud rate in dataset


def load_data(min_coverage: float = 0.5) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """Load data with feature preprocessing."""
    logger.info(f"Loading data from raw_insertions.parquet...")
    
    df = pl.read_parquet(RAW_INSERTIONS)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Add fraud label
    df = df.with_columns(
        pl.col("fraud_flag").is_not_null().cast(pl.Int8).alias("is_fraud")
    )
    
    # Exclude non-feature columns
    exclude_patterns = {
        "object_reference", "insertion_id", "owner_id", "user_id",
        "submission_at", "fraud_flag", "is_fraud", "first_published_date",
        "listing_created_at", "account_created_at", "contact_emails_hash",
        "user_ip_address_hash", "seonFraudScore",
    }
    
    feature_cols = []
    for col in df.columns:
        if col in exclude_patterns:
            continue
        if any(pattern in col for pattern in ["_hash", "legacy.personId"]):
            continue
        feature_cols.append(col)
    
    # Filter by coverage
    selected_cols = []
    for col in feature_cols:
        coverage = 1 - (df[col].null_count() / len(df))
        if coverage >= min_coverage:
            selected_cols.append(col)
    
    logger.info(f"Features with ≥{min_coverage*100:.0f}% coverage: {len(selected_cols)}")
    
    # Convert to pandas
    pdf = df.select(selected_cols + ["is_fraud", "submission_at"]).to_pandas()
    
    # Handle categorical columns
    for col in selected_cols:
        if pdf[col].dtype == 'object' or pdf[col].dtype.name == 'category':
            pdf[col] = pdf[col].fillna('__MISSING__')
            pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))
    
    pdf = pdf.fillna(-999)
    
    X = pdf[selected_cols]
    y = pdf["is_fraud"]
    
    return X, selected_cols, y, pdf


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    """Calculate precision at top k predictions."""
    top_k_idx = np.argsort(scores)[-k:]
    return np.mean(y_true.iloc[top_k_idx] if hasattr(y_true, 'iloc') else y_true[top_k_idx])


def train_isolation_forest(X_train: pd.DataFrame, contamination: float = FRAUD_RATE) -> IsolationForest:
    """Train Isolation Forest."""
    logger.info(f"Training Isolation Forest (contamination={contamination})...")
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def train_one_class_svm(X_train: pd.DataFrame, nu: float = FRAUD_RATE) -> OneClassSVM:
    """Train One-Class SVM."""
    logger.info(f"Training One-Class SVM (nu={nu})...")
    # Scale features for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = OneClassSVM(
        nu=min(nu, 0.5),  # nu must be in (0, 0.5]
        kernel='rbf',
        gamma='scale'
    )
    model.fit(X_scaled)
    return model, scaler


def train_lof(X_train: pd.DataFrame, contamination: float = FRAUD_RATE) -> LocalOutlierFactor:
    """Train Local Outlier Factor."""
    logger.info(f"Training LOF (contamination={contamination})...")
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,  # Required for prediction on new data
        n_jobs=-1
    )
    model.fit(X_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   scaler=None, model_name: str = "") -> Dict:
    """Evaluate anomaly detection model."""
    logger.info(f"Evaluating {model_name}...")
    
    # Get anomaly scores
    if scaler is not None:
        X_scaled = scaler.transform(X_test)
        scores = -model.decision_function(X_scaled)
    else:
        scores = -model.decision_function(X_test)
    
    # Higher score = more anomalous
    auc_pr = average_precision_score(y_test, scores)
    auc_roc = roc_auc_score(y_test, scores)
    p_at_100 = precision_at_k(y_test.values, scores, k=100)
    p_at_500 = precision_at_k(y_test.values, scores, k=500)
    
    return {
        'method': model_name,
        'auc_pr': auc_pr,
        'auc_roc': auc_roc,
        'p_at_100': p_at_100,
        'p_at_500': p_at_500,
        'type': 'unsupervised'
    }


def run_anomaly_detection(config: DictConfig) -> pd.DataFrame:
    """Run full anomaly detection comparison."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 10A: UNSUPERVISED ANOMALY DETECTION")
    logger.info("=" * 70)
    
    # Load data
    X, feature_names, y, pdf = load_data(min_coverage=0.5)
    logger.info(f"Dataset: {len(X):,} samples, {len(feature_names)} features")
    logger.info(f"Fraud rate: {y.mean()*100:.2f}%")
    
    # Time-based split (use first 80% for training, last 20% for test)
    n_train = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    
    logger.info(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    
    # For semi-supervised: train only on non-fraud samples
    X_train_clean = X_train[y_train == 0]
    logger.info(f"Clean training set (non-fraud only): {len(X_train_clean):,} samples")
    
    results = []
    
    # 1. Isolation Forest
    try:
        iso_forest = train_isolation_forest(X_train_clean)
        iso_result = evaluate_model(iso_forest, X_test, y_test, model_name="Isolation Forest")
        results.append(iso_result)
        logger.info(f"  Isolation Forest: AUC-PR={iso_result['auc_pr']:.4f}, P@100={iso_result['p_at_100']:.4f}")
    except Exception as e:
        logger.error(f"Isolation Forest failed: {e}")
    
    # 2. One-Class SVM (slower, use subset if needed)
    try:
        # Use subset for SVM (it's slow on large datasets)
        max_samples = min(50000, len(X_train_clean))
        X_svm_train = X_train_clean.sample(n=max_samples, random_state=42)
        
        ocsvm, scaler = train_one_class_svm(X_svm_train)
        ocsvm_result = evaluate_model(ocsvm, X_test, y_test, scaler=scaler, model_name="One-Class SVM")
        results.append(ocsvm_result)
        logger.info(f"  One-Class SVM: AUC-PR={ocsvm_result['auc_pr']:.4f}, P@100={ocsvm_result['p_at_100']:.4f}")
    except Exception as e:
        logger.error(f"One-Class SVM failed: {e}")
    
    # 3. Local Outlier Factor
    try:
        # LOF needs smaller dataset
        max_samples = min(50000, len(X_train_clean))
        X_lof_train = X_train_clean.sample(n=max_samples, random_state=42)
        
        lof = train_lof(X_lof_train)
        lof_result = evaluate_model(lof, X_test, y_test, model_name="LOF")
        results.append(lof_result)
        logger.info(f"  LOF: AUC-PR={lof_result['auc_pr']:.4f}, P@100={lof_result['p_at_100']:.4f}")
    except Exception as e:
        logger.error(f"LOF failed: {e}")
    
    # 4. Add XGBoost supervised baseline for comparison (load from results registry)
    from src.experiments.results import get_metric, ExperimentID
    
    baseline_auc_pr = get_metric(ExperimentID.XGBOOST_BASELINE, "auc_pr", default=0.784)
    baseline_auc_roc = get_metric(ExperimentID.XGBOOST_BASELINE, "auc_roc", default=0.95)
    baseline_p100 = get_metric(ExperimentID.XGBOOST_BASELINE, "p_at_100", default=0.87)
    baseline_p500 = get_metric(ExperimentID.XGBOOST_BASELINE, "p_at_500", default=0.75)
    
    results.append({
        'method': 'XGBoost (supervised)',
        'auc_pr': baseline_auc_pr,
        'auc_roc': baseline_auc_roc,
        'p_at_100': baseline_p100,
        'p_at_500': baseline_p500,
        'type': 'supervised'
    })
    
    return pd.DataFrame(results)


def plot_comparison(results: pd.DataFrame, output_path: Path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors by type
    colors = {'unsupervised': '#3498db', 'supervised': '#e74c3c'}
    bar_colors = [colors[t] for t in results['type']]
    
    # AUC-PR comparison
    ax1 = axes[0]
    bars1 = ax1.barh(results['method'], results['auc_pr'], color=bar_colors)
    ax1.set_xlabel('AUC-PR')
    ax1.set_title('AUC-PR: Unsupervised vs Supervised')
    ax1.set_xlim(0, 1)
    for bar, val in zip(bars1, results['auc_pr']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.3f}', va='center')
    
    # P@100 comparison
    ax2 = axes[1]
    bars2 = ax2.barh(results['method'], results['p_at_100'], color=bar_colors)
    ax2.set_xlabel('P@100')
    ax2.set_title('Precision@100: Unsupervised vs Supervised')
    ax2.set_xlim(0, 1)
    for bar, val in zip(bars2, results['p_at_100']):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 10A: UNSUPERVISED ANOMALY DETECTION")
    logger.info("=" * 70)
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_name = cfg.get('experiment_name', 'unsupervised-pattern-exp10')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"exp10a_anomaly_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "unsupervised", "sub_experiment": "10A"}
    ):
        # Run anomaly detection
        results = run_anomaly_detection(cfg)
        
        # Log metrics
        for _, row in results.iterrows():
            prefix = row['method'].lower().replace(' ', '_').replace('(', '').replace(')', '')
            mlflow.log_metrics({
                f"{prefix}_auc_pr": row['auc_pr'],
                f"{prefix}_p_at_100": row['p_at_100']
            })
        
        # Save results
        results_path = OUTPUT_DIR / "anomaly_detection_results.csv"
        results.to_csv(results_path, index=False)
        logger.info(f"Saved: {results_path}")
        
        # Create visualization
        plot_path = OUTPUT_DIR / "anomaly_comparison.png"
        plot_comparison(results, plot_path)
        
        # Log artifacts
        mlflow.log_artifacts(str(OUTPUT_DIR))
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        print(results.to_string(index=False))
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        
        # Save to results registry
        from src.experiments.results import save_result, ExperimentID
        
        best_unsupervised_row = results[results['type'] == 'unsupervised'].sort_values('auc_pr', ascending=False).iloc[0]
        save_result(ExperimentID.EXP10A_ANOMALY, {
            "best_method": best_unsupervised_row['method'],
            "best_auc_pr": float(best_unsupervised_row['auc_pr']),
            "best_auc_roc": float(best_unsupervised_row['auc_roc']),
            "all_methods": results.to_dict(orient='records')
        })
        
        # Key finding
        best_unsupervised = results[results['type'] == 'unsupervised']['auc_pr'].max()
        supervised = results[results['type'] == 'supervised']['auc_pr'].iloc[0]
        gap = supervised - best_unsupervised
        logger.info(f"\n🔑 Key Finding: Supervised XGBoost outperforms best unsupervised by {gap:.3f} AUC-PR ({gap/best_unsupervised*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    main()
