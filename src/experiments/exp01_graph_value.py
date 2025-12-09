"""
Experiment 01: Graph Feature Value (RQ1)

Demonstrates that graph-derived features add significant value to fraud detection.
This is validated by comparing full model vs tabular-only model.

Run with:
    python -m src.experiments.exp01_graph_value

Output:
    - artifacts/results/exp01_graph_value.json
    - MLflow run with comparison metrics
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

import hydra
import mlflow
from omegaconf import DictConfig

from src.data.loader import load_data
from src.features.processor import FeatureProcessor
from src.models.config.constants import GRAPH_FEATURES
from src.models.utils.common import setup_mlflow
from src.experiments.results import save_result, ExperimentID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_comparison(cfg: DictConfig) -> Dict[str, Any]:
    """
    Compare full model (tabular + graph) vs tabular-only.
    
    This directly measures the contribution of graph features.
    """
    from src.experiments.exp09_feature_selection import prepare_features, train_and_evaluate
    
    logger.info("Loading data...")
    df = load_data()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    
    results = {}
    
    # Full model (tabular + graph)
    logger.info("\n" + "=" * 60)
    logger.info("Testing: Full Model (Tabular + Graph)")
    logger.info("=" * 60)
    df_pd, feature_cols = prepare_features(df, "full", cutoff_date)
    full_metrics = train_and_evaluate(df_pd, feature_cols)
    results['full'] = full_metrics
    logger.info(f"  AUC-PR: {full_metrics['auc_pr']:.4f}")
    
    # Tabular only (no graph)
    logger.info("\n" + "=" * 60)
    logger.info("Testing: Tabular Only (No Graph Features)")
    logger.info("=" * 60)
    df_pd, feature_cols = prepare_features(df, "tabular_only", cutoff_date)
    tabular_metrics = train_and_evaluate(df_pd, feature_cols)
    results['tabular_only'] = tabular_metrics
    logger.info(f"  AUC-PR: {tabular_metrics['auc_pr']:.4f}")
    
    # Calculate graph contribution
    graph_contribution = full_metrics['auc_pr'] - tabular_metrics['auc_pr']
    graph_contribution_pct = (graph_contribution / tabular_metrics['auc_pr']) * 100
    
    results['graph_contribution'] = {
        'absolute': graph_contribution,
        'relative_pct': graph_contribution_pct
    }
    
    return results


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 01: GRAPH FEATURE VALUE (RQ1)")
    logger.info("=" * 70)
    logger.info("Research Question: Do graph features add significant value?")
    logger.info("=" * 70)
    
    experiment_name = cfg.get('experiment_name', 'graph-value-rq1')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"exp01_graph_value_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment": "exp01", "rq": "RQ1"}
    ):
        results = run_comparison(cfg)
        
        # Log metrics
        mlflow.log_metrics({
            "full_auc_pr": results['full']['auc_pr'],
            "tabular_only_auc_pr": results['tabular_only']['auc_pr'],
            "graph_contribution_absolute": results['graph_contribution']['absolute'],
            "graph_contribution_pct": results['graph_contribution']['relative_pct']
        })
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Save to results registry
        save_result(ExperimentID.EXP1_GRAPH_VALUE, {
            "full_auc_pr": results['full']['auc_pr'],
            "tabular_only_auc_pr": results['tabular_only']['auc_pr'],
            "graph_contribution": results['graph_contribution']['absolute'],
            "graph_contribution_pct": results['graph_contribution']['relative_pct'],
            "n_graph_features": len(GRAPH_FEATURES)
        })
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Full model (tabular + graph): {results['full']['auc_pr']:.4f} AUC-PR")
        logger.info(f"Tabular only:                 {results['tabular_only']['auc_pr']:.4f} AUC-PR")
        logger.info(f"Graph contribution:           +{results['graph_contribution']['absolute']:.4f} (+{results['graph_contribution']['relative_pct']:.1f}%)")
        
        if results['graph_contribution']['absolute'] > 0.02:
            logger.info("\n✅ CONCLUSION: Graph features provide SIGNIFICANT value")
        elif results['graph_contribution']['absolute'] > 0:
            logger.info("\n⚠️ CONCLUSION: Graph features provide MARGINAL value")
        else:
            logger.info("\n❌ CONCLUSION: Graph features provide NO measurable value")
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
