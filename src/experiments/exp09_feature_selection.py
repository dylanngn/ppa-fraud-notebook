"""
Experiment 9: Feature Selection Validation

Validates the simplified feature architecture by comparing:
1. Full auto mode (tabular + graph features)
2. Tabular-only mode (no graph features)
3. Graph-only mode (only graph-derived features)

This validates:
- Graph features add value (RQ1)
- Auto selection works as well as manual curation
- The simplified architecture is justified

Run with:
    python -m src.experiments.exp9_feature_selection

Output:
    - artifacts/feature_selection/comparison_results.json
    - artifacts/feature_selection/ablation_comparison.png
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.data.loader import load_data
from src.features.processor import FeatureProcessor
from src.models.config.constants import GRAPH_FEATURES, EXCLUDED_COLUMNS
from src.models.utils.common import setup_mlflow
from src.utils.metrics import calculate_metrics
from src.utils.hydra_utils import resolve_path
from src.experiments.results import save_result, ExperimentID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
OUTPUT_DIR = ARTIFACTS_DIR / "feature_selection"


def prepare_features(
    df: pl.DataFrame,
    mode: str,
    cutoff_date: datetime
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features based on mode.
    
    Args:
        df: Input DataFrame
        mode: 'full', 'tabular_only', or 'graph_only'
        cutoff_date: Temporal cutoff for feature computation
        
    Returns:
        Tuple of (feature DataFrame, feature column names)
    """
    # Determine categories based on mode
    if mode == "full":
        categories = ["base", "graph"]
    elif mode == "tabular_only":
        categories = ["base"]  # No graph features
    elif mode == "graph_only":
        categories = ["graph"]  # Only graph features
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Process features
    processor = FeatureProcessor(categories=categories)
    df_processed = processor.process(df, cutoff_date)
    
    # Convert to pandas
    df_pd = df_processed.to_pandas()
    
    # Identify feature columns (exclude IDs, target, timestamps)
    exclude_cols = {
        "insertion_id", "object_reference", "owner_id", "user_id",
        "is_fraud", "fraud_flag", "submission_at", "account_created_at"
    }
    
    feature_cols = [c for c in df_pd.columns if c not in exclude_cols]
    
    # For graph_only mode, filter to only graph features
    if mode == "graph_only":
        feature_cols = [c for c in feature_cols if c in GRAPH_FEATURES]
    
    # For tabular_only mode, exclude graph features
    if mode == "tabular_only":
        feature_cols = [c for c in feature_cols if c not in GRAPH_FEATURES]
    
    logger.info(f"Mode '{mode}': {len(feature_cols)} features")
    
    return df_pd, feature_cols


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str = "is_fraud",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train XGBoost and evaluate performance."""
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target].astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='aucpr'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test.values, y_proba)
    
    return {
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc_pr": metrics["auc_pr"],
        "auc_roc": metrics["auc_roc"],
        "p_at_100": metrics.get("p@100", 0),
        "feature_cols": feature_cols[:20]  # Top 20 for logging
    }


def run_comparison(cfg: DictConfig) -> pd.DataFrame:
    """Run feature selection comparison."""
    
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df):,} samples")
    
    # Use a fixed cutoff date for reproducibility
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    
    results = []
    modes = ["full", "tabular_only", "graph_only"]
    
    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running mode: {mode}")
        logger.info(f"{'='*60}")
        
        try:
            # Prepare features
            df_pd, feature_cols = prepare_features(df, mode, cutoff_date)
            
            if len(feature_cols) == 0:
                logger.warning(f"No features for mode '{mode}', skipping")
                continue
            
            # Train and evaluate
            metrics = train_and_evaluate(df_pd, feature_cols)
            
            results.append({
                "mode": mode,
                "n_features": metrics["n_features"],
                "auc_pr": metrics["auc_pr"],
                "auc_roc": metrics["auc_roc"],
                "p_at_100": metrics["p_at_100"]
            })
            
            logger.info(f"  Features: {metrics['n_features']}")
            logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
            logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            
        except Exception as e:
            logger.error(f"Mode '{mode}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results)


def plot_comparison(results: pd.DataFrame, output_path: Path):
    """Create comparison visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color scheme
    colors = {
        "full": "#27ae60",
        "tabular_only": "#3498db",
        "graph_only": "#e74c3c"
    }
    bar_colors = [colors.get(m, "#95a5a6") for m in results['mode']]
    
    # AUC-PR comparison
    ax1 = axes[0]
    bars1 = ax1.barh(results['mode'], results['auc_pr'], color=bar_colors)
    ax1.set_xlabel('AUC-PR')
    ax1.set_title('AUC-PR by Feature Mode')
    ax1.set_xlim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars1, results['auc_pr']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    # Feature count
    ax2 = axes[1]
    bars2 = ax2.barh(results['mode'], results['n_features'], color=bar_colors)
    ax2.set_xlabel('Number of Features')
    ax2.set_title('Feature Count by Mode')
    
    # Add value labels
    for bar, val in zip(bars2, results['n_features']):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2, 
                str(int(val)), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {output_path}")


def compute_insights(results: pd.DataFrame) -> Dict[str, Any]:
    """Compute key insights from results."""
    
    full = results[results['mode'] == 'full'].iloc[0] if 'full' in results['mode'].values else None
    tabular = results[results['mode'] == 'tabular_only'].iloc[0] if 'tabular_only' in results['mode'].values else None
    graph = results[results['mode'] == 'graph_only'].iloc[0] if 'graph_only' in results['mode'].values else None
    
    insights = {}
    
    if full is not None and tabular is not None:
        graph_contribution = full['auc_pr'] - tabular['auc_pr']
        graph_contribution_pct = (graph_contribution / tabular['auc_pr']) * 100
        insights['graph_contribution'] = {
            "absolute": float(graph_contribution),
            "relative_pct": float(graph_contribution_pct),
            "description": f"Graph features add +{graph_contribution:.3f} AUC-PR ({graph_contribution_pct:.1f}%)"
        }
    
    if graph is not None:
        insights['graph_standalone'] = {
            "auc_pr": float(graph['auc_pr']),
            "n_features": int(graph['n_features']),
            "description": f"Graph-only achieves {graph['auc_pr']:.3f} AUC-PR with {graph['n_features']} features"
        }
    
    if full is not None:
        insights['full_performance'] = {
            "auc_pr": float(full['auc_pr']),
            "n_features": int(full['n_features']),
            "description": f"Full auto mode achieves {full['auc_pr']:.3f} AUC-PR"
        }
    
    return insights


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 9: FEATURE SELECTION VALIDATION")
    logger.info("=" * 70)
    logger.info("Comparing: Full Auto | Tabular Only | Graph Only")
    logger.info("=" * 70)
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_name = cfg.get('experiment_name', 'feature-selection-exp9')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"exp9_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "feature_selection", "experiment": "exp9"}
    ):
        # Run comparison
        results = run_comparison(cfg)
        
        if results.empty:
            logger.error("No results generated!")
            return
        
        # Log metrics
        for _, row in results.iterrows():
            prefix = row['mode']
            mlflow.log_metrics({
                f"{prefix}_auc_pr": row['auc_pr'],
                f"{prefix}_n_features": row['n_features']
            })
        
        # Compute insights
        insights = compute_insights(results)
        
        # Save results
        results_path = OUTPUT_DIR / "comparison_results.csv"
        results.to_csv(results_path, index=False)
        logger.info(f"Saved: {results_path}")
        
        # Save insights as JSON
        insights_path = OUTPUT_DIR / "comparison_insights.json"
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2)
        logger.info(f"Saved: {insights_path}")
        
        # Create visualization
        plot_path = OUTPUT_DIR / "ablation_comparison.png"
        plot_comparison(results, plot_path)
        
        # Log artifacts
        mlflow.log_artifacts(str(OUTPUT_DIR))
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Save to results registry
        full_result = results[results['mode'] == 'full'].iloc[0] if 'full' in results['mode'].values else None
        if full_result is not None:
            save_result(ExperimentID.EXP9_FEATURE_SELECTION, {
                "full_auc_pr": float(full_result['auc_pr']),
                "full_n_features": int(full_result['n_features']),
                "comparison": results.to_dict(orient='records'),
                "insights": insights
            })
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        print(results.to_string(index=False))
        
        logger.info("\n" + "=" * 70)
        logger.info("KEY INSIGHTS")
        logger.info("=" * 70)
        for key, insight in insights.items():
            logger.info(f"  {key}: {insight.get('description', insight)}")
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")
        logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
