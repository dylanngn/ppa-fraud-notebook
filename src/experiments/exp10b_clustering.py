"""
Experiment 10B: Clustering Analysis for Fraud Ring Discovery

Research Gap: Supervisor feedback (Section 2.2) requires clustering for fraud ring detection.

Methods:
    - K-Means (baseline)
    - DBSCAN (density-based, good for arbitrary shapes)
    - HDBSCAN (hierarchical density-based)

Run with:
    python -m src.experiments.exp10_clustering

Output:
    - artifacts/unsupervised/clustering_results.csv
    - artifacts/unsupervised/cluster_fraud_rates.png
    - artifacts/unsupervised/high_fraud_clusters.csv
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.models.utils.common import setup_mlflow
from src.utils.hydra_utils import resolve_path

# Try to import HDBSCAN (optional)
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logging.warning("HDBSCAN not installed. pip install hdbscan for better clustering.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
OUTPUT_DIR = ARTIFACTS_DIR / "unsupervised"
RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"


def load_data(min_coverage: float = 0.5) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """Load data with feature preprocessing."""
    logger.info("Loading data...")
    
    df = pl.read_parquet(RAW_INSERTIONS)
    
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
    
    # Convert to pandas (use object_reference as ID since insertion_id doesn't exist)
    id_col = "object_reference" if "object_reference" in df.columns else "listing.id"
    pdf = df.select(selected_cols + ["is_fraud"]).to_pandas()
    
    # Handle categorical columns
    for col in selected_cols:
        if pdf[col].dtype == 'object' or pdf[col].dtype.name == 'category':
            pdf[col] = pdf[col].fillna('__MISSING__')
            pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))
    
    pdf = pdf.fillna(-999)
    
    X = pdf[selected_cols]
    y = pdf["is_fraud"]
    
    return X, selected_cols, y, pdf


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 20) -> np.ndarray:
    """Run K-Means clustering."""
    logger.info(f"Running K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels


def run_dbscan(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 50) -> np.ndarray:
    """Run DBSCAN clustering."""
    logger.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)
    return labels


def run_hdbscan(X_scaled: np.ndarray, min_cluster_size: int = 100) -> np.ndarray:
    """Run HDBSCAN clustering."""
    if not HAS_HDBSCAN:
        logger.warning("HDBSCAN not available, skipping...")
        return None
    
    logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10)
    labels = clusterer.fit_predict(X_scaled)
    return labels


def analyze_clusters(labels: np.ndarray, y: pd.Series, method_name: str) -> pd.DataFrame:
    """Analyze fraud rate per cluster."""
    df = pd.DataFrame({
        'cluster': labels,
        'is_fraud': y.values
    })
    
    # Calculate stats per cluster
    cluster_stats = df.groupby('cluster').agg(
        n_samples=('is_fraud', 'count'),
        n_fraud=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).reset_index()
    
    cluster_stats['method'] = method_name
    cluster_stats['fraud_lift'] = cluster_stats['fraud_rate'] / y.mean()
    
    return cluster_stats


def run_clustering(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full clustering analysis."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 10B: CLUSTERING ANALYSIS")
    logger.info("=" * 70)
    
    # Load data
    X, feature_names, y, pdf = load_data(min_coverage=0.5)
    logger.info(f"Dataset: {len(X):,} samples, {len(feature_names)} features")
    logger.info(f"Overall fraud rate: {y.mean()*100:.2f}%")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce dimensionality for visualization and speed
    logger.info("Reducing dimensionality with PCA (50 components)...")
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    all_cluster_stats = []
    summary_results = []
    
    # 1. K-Means with different k values
    for k in [10, 20, 50]:
        labels = run_kmeans(X_pca, n_clusters=k)
        stats = analyze_clusters(labels, y, f"KMeans_k{k}")
        all_cluster_stats.append(stats)
        
        # Summary metrics
        n_high_fraud = (stats['fraud_rate'] > y.mean() * 2).sum()  # 2x average
        max_fraud_rate = stats['fraud_rate'].max()
        
        summary_results.append({
            'method': f'KMeans (k={k})',
            'n_clusters': k,
            'n_high_fraud_clusters': n_high_fraud,
            'max_fraud_rate': max_fraud_rate,
            'max_fraud_lift': max_fraud_rate / y.mean()
        })
        logger.info(f"  KMeans k={k}: {n_high_fraud} high-fraud clusters, max rate={max_fraud_rate:.2%}")
    
    # 2. DBSCAN
    # Use subset for DBSCAN (it's slow)
    sample_idx = np.random.choice(len(X_pca), min(50000, len(X_pca)), replace=False)
    X_sample = X_pca[sample_idx]
    y_sample = y.iloc[sample_idx]
    
    labels = run_dbscan(X_sample, eps=3.0, min_samples=50)
    stats = analyze_clusters(labels, y_sample, "DBSCAN")
    all_cluster_stats.append(stats)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    n_high_fraud = (stats[stats['cluster'] != -1]['fraud_rate'] > y.mean() * 2).sum()
    max_fraud_rate = stats[stats['cluster'] != -1]['fraud_rate'].max() if n_clusters > 0 else 0
    
    summary_results.append({
        'method': 'DBSCAN',
        'n_clusters': n_clusters,
        'n_high_fraud_clusters': n_high_fraud,
        'max_fraud_rate': max_fraud_rate,
        'max_fraud_lift': max_fraud_rate / y.mean() if max_fraud_rate > 0 else 0,
        'n_noise_points': n_noise
    })
    logger.info(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise, {n_high_fraud} high-fraud")
    
    # 3. HDBSCAN (if available)
    if HAS_HDBSCAN:
        labels = run_hdbscan(X_sample, min_cluster_size=100)
        if labels is not None:
            stats = analyze_clusters(labels, y_sample, "HDBSCAN")
            all_cluster_stats.append(stats)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            n_high_fraud = (stats[stats['cluster'] != -1]['fraud_rate'] > y.mean() * 2).sum()
            max_fraud_rate = stats[stats['cluster'] != -1]['fraud_rate'].max() if n_clusters > 0 else 0
            
            summary_results.append({
                'method': 'HDBSCAN',
                'n_clusters': n_clusters,
                'n_high_fraud_clusters': n_high_fraud,
                'max_fraud_rate': max_fraud_rate,
                'max_fraud_lift': max_fraud_rate / y.mean() if max_fraud_rate > 0 else 0,
                'n_noise_points': n_noise
            })
            logger.info(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise, {n_high_fraud} high-fraud")
    
    # Combine results
    all_stats_df = pd.concat(all_cluster_stats, ignore_index=True)
    summary_df = pd.DataFrame(summary_results)
    
    return summary_df, all_stats_df


def plot_cluster_fraud_rates(all_stats: pd.DataFrame, output_path: Path):
    """Create visualization of fraud rates by cluster."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = all_stats['method'].unique()
    
    for i, method in enumerate(methods[:4]):  # Max 4 subplots
        ax = axes[i // 2, i % 2]
        method_stats = all_stats[all_stats['method'] == method].sort_values('fraud_rate', ascending=False)
        
        # Color by fraud rate
        colors = ['#e74c3c' if r > 0.15 else '#f39c12' if r > 0.08 else '#3498db' 
                  for r in method_stats['fraud_rate']]
        
        ax.barh(range(len(method_stats)), method_stats['fraud_rate'], color=colors)
        ax.set_xlabel('Fraud Rate')
        ax.set_ylabel('Cluster')
        ax.set_title(f'{method}')
        ax.axvline(x=0.082, color='red', linestyle='--', alpha=0.5, label='Overall avg')
    
    plt.suptitle('Fraud Rate by Cluster', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 10B: CLUSTERING ANALYSIS")
    logger.info("=" * 70)
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_name = cfg.get('experiment_name', 'unsupervised-pattern-exp10')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"exp10b_clustering_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "unsupervised", "sub_experiment": "10B"}
    ):
        # Run clustering
        summary_df, all_stats_df = run_clustering(cfg)
        
        # Log metrics
        for _, row in summary_df.iterrows():
            prefix = row['method'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            mlflow.log_metrics({
                f"{prefix}_n_clusters": row['n_clusters'],
                f"{prefix}_max_fraud_lift": row['max_fraud_lift']
            })
        
        # Save results
        summary_path = OUTPUT_DIR / "clustering_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved: {summary_path}")
        
        stats_path = OUTPUT_DIR / "cluster_fraud_rates.csv"
        all_stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved: {stats_path}")
        
        # Identify high-fraud clusters
        high_fraud = all_stats_df[all_stats_df['fraud_lift'] > 2.0]
        if len(high_fraud) > 0:
            high_fraud_path = OUTPUT_DIR / "high_fraud_clusters.csv"
            high_fraud.to_csv(high_fraud_path, index=False)
            logger.info(f"Saved {len(high_fraud)} high-fraud clusters to: {high_fraud_path}")
        
        # Create visualization
        plot_path = OUTPUT_DIR / "cluster_fraud_rates.png"
        plot_cluster_fraud_rates(all_stats_df, plot_path)
        
        # Log artifacts
        mlflow.log_artifacts(str(OUTPUT_DIR))
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("CLUSTERING SUMMARY")
        logger.info("=" * 70)
        print(summary_df.to_string(index=False))
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")
        
        # Save to results registry
        from src.experiments.results import save_result, ExperimentID
        
        best_method_row = summary_df.sort_values('max_fraud_lift', ascending=False).iloc[0]
        save_result(ExperimentID.EXP10B_CLUSTERING, {
            "best_method": best_method_row['method'],
            "best_max_fraud_lift": float(best_method_row['max_fraud_lift']),
            "n_high_fraud_clusters": len(high_fraud),
            "methods_summary": summary_df.to_dict(orient='records')
        })
        
        # Key finding
        if len(high_fraud) > 0:
            best_lift = high_fraud['fraud_lift'].max()
            logger.info(f"\n🔑 Key Finding: Found {len(high_fraud)} high-fraud clusters with up to {best_lift:.1f}x fraud lift")
        else:
            logger.info("\n⚠️ No high-fraud clusters found (lift > 2x)")
    
    return summary_df


if __name__ == "__main__":
    main()
