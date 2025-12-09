"""
Experiment 02: Hybrid Architecture Comparison (RQ2)

Compares hybrid GNN-XGBoost architecture against alternatives:
- XGBoost with handcrafted graph features (baseline)
- SAGE Hybrid (SAGE embeddings + XGBoost)
- HGT Hybrid (HGT embeddings + XGBoost)

Run with:
    python -m src.experiments.exp02_architecture

Note: This experiment requires GNN models to be trained first:
    python -m src.models.gnn.sage
    python -m src.models.gnn.hgt

Output:
    - artifacts/results/exp02_architecture.json
    - MLflow run with comparison metrics
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import mlflow
import torch
from omegaconf import DictConfig

from src.models.utils.common import setup_mlflow
from src.experiments.results import save_result, get_metric, ExperimentID
from src.utils.hydra_utils import resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = resolve_path("artifacts")


def load_gnn_results() -> Dict[str, Optional[Dict[str, float]]]:
    """Load results from GNN training runs."""
    
    results = {}
    
    # Check for SAGE embeddings
    sage_path = ARTIFACTS_DIR / "embeddings_sage.pt"
    if sage_path.exists():
        # Load from MLflow or results registry
        sage_auc = get_metric("gnn_sage", "auc_pr")
        results['sage'] = {
            "auc_pr": sage_auc,
            "path": str(sage_path)
        } if sage_auc else None
        logger.info(f"SAGE embeddings found: {sage_path}")
    else:
        results['sage'] = None
        logger.warning("SAGE embeddings not found. Run: python -m src.models.gnn.sage")
    
    # Check for HGT embeddings
    hgt_path = ARTIFACTS_DIR / "embeddings_hgt.pt"
    if hgt_path.exists():
        hgt_auc = get_metric("gnn_hgt", "auc_pr")
        results['hgt'] = {
            "auc_pr": hgt_auc,
            "path": str(hgt_path)
        } if hgt_auc else None
        logger.info(f"HGT embeddings found: {hgt_path}")
    else:
        results['hgt'] = None
        logger.warning("HGT embeddings not found. Run: python -m src.models.gnn.hgt")
    
    return results


def get_xgboost_baseline() -> Dict[str, float]:
    """Get XGBoost baseline from results registry."""
    auc_pr = get_metric(ExperimentID.XGBOOST_BASELINE, "auc_pr", default=None)
    
    if auc_pr is None:
        logger.warning("XGBoost baseline not found. Run: python -m src.models.train")
        return {"auc_pr": None}
    
    return {
        "auc_pr": auc_pr,
        "auc_roc": get_metric(ExperimentID.XGBOOST_BASELINE, "auc_roc"),
        "p_at_100": get_metric(ExperimentID.XGBOOST_BASELINE, "p_at_100")
    }


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 02: HYBRID ARCHITECTURE COMPARISON (RQ2)")
    logger.info("=" * 70)
    logger.info("Research Question: How does hybrid GNN-XGBoost compare to alternatives?")
    logger.info("=" * 70)
    
    experiment_name = cfg.get('experiment_name', 'architecture-rq2')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"exp02_architecture_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment": "exp02", "rq": "RQ2"}
    ):
        # Load baseline
        logger.info("\n" + "=" * 60)
        logger.info("Loading XGBoost Baseline (Handcrafted Graph Features)")
        logger.info("=" * 60)
        xgb_baseline = get_xgboost_baseline()
        if xgb_baseline.get("auc_pr"):
            logger.info(f"  AUC-PR: {xgb_baseline['auc_pr']:.4f}")
        else:
            logger.warning("  Baseline not available. Run training first.")
        
        # Load GNN results
        logger.info("\n" + "=" * 60)
        logger.info("Loading GNN Hybrid Results")
        logger.info("=" * 60)
        gnn_results = load_gnn_results()
        
        # Compile comparison
        comparison = {
            "xgboost_baseline": xgb_baseline,
            "sage_hybrid": gnn_results.get('sage'),
            "hgt_hybrid": gnn_results.get('hgt')
        }
        
        # Log metrics
        if xgb_baseline.get("auc_pr"):
            mlflow.log_metric("xgboost_baseline_auc_pr", xgb_baseline["auc_pr"])
        if gnn_results.get('sage') and gnn_results['sage'].get("auc_pr"):
            mlflow.log_metric("sage_hybrid_auc_pr", gnn_results['sage']["auc_pr"])
        if gnn_results.get('hgt') and gnn_results['hgt'].get("auc_pr"):
            mlflow.log_metric("hgt_hybrid_auc_pr", gnn_results['hgt']["auc_pr"])
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Save to results registry
        save_result(ExperimentID.EXP2_ARCHITECTURE, comparison)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        
        print("\n| Model | AUC-PR | Status |")
        print("|-------|--------|--------|")
        
        if xgb_baseline.get("auc_pr"):
            print(f"| XGBoost (handcrafted) | {xgb_baseline['auc_pr']:.4f} | ✅ Baseline |")
        else:
            print("| XGBoost (handcrafted) | - | ❌ Not run |")
        
        if gnn_results.get('sage') and gnn_results['sage'].get("auc_pr"):
            delta = gnn_results['sage']['auc_pr'] - (xgb_baseline.get('auc_pr') or 0)
            print(f"| SAGE Hybrid | {gnn_results['sage']['auc_pr']:.4f} | {'+' if delta > 0 else ''}{delta:.4f} |")
        else:
            print("| SAGE Hybrid | - | ❌ Not run |")
        
        if gnn_results.get('hgt') and gnn_results['hgt'].get("auc_pr"):
            delta = gnn_results['hgt']['auc_pr'] - (xgb_baseline.get('auc_pr') or 0)
            print(f"| HGT Hybrid | {gnn_results['hgt']['auc_pr']:.4f} | {'+' if delta > 0 else ''}{delta:.4f} |")
        else:
            print("| HGT Hybrid | - | ❌ Not run |")
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")
        
        # Provide instructions if models not trained
        missing = []
        if not xgb_baseline.get("auc_pr"):
            missing.append("python -m src.models.train")
        if not gnn_results.get('sage'):
            missing.append("python -m src.models.gnn.sage")
        if not gnn_results.get('hgt'):
            missing.append("python -m src.models.gnn.hgt")
        
        if missing:
            logger.info("\n⚠️ To complete this experiment, run:")
            for cmd in missing:
                logger.info(f"    {cmd}")


if __name__ == "__main__":
    main()
