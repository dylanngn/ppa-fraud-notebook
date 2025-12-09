"""
Experiment 08: Feature Drift Detection & Field Audit

Combined experiment for drift monitoring and feature discovery:
- 8A: Drift Detection POC (using Evidently AI)
- 8B: Field Audit (analyze unused fields for fraud correlation)

Run with:
    python -m src.experiments.exp08_drift              # Run all
    python -m src.experiments.exp08_drift +mode=drift  # Drift only
    python -m src.experiments.exp08_drift +mode=audit  # Audit only

Output:
    - artifacts/drift_monitoring/drift_summary.json
    - artifacts/drift_monitoring/field_audit_report.json
    - artifacts/results/exp08_drift.json
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import hydra
import mlflow
from omegaconf import DictConfig

from src.models.utils.common import setup_mlflow
from src.experiments.results import save_result, ExperimentID
from src.utils.hydra_utils import resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = resolve_path("artifacts/drift_monitoring")


def run_drift_detection() -> Dict[str, Any]:
    """Run drift detection POC (8A)."""
    logger.info("\n" + "=" * 60)
    logger.info("8A: DRIFT DETECTION POC")
    logger.info("=" * 60)
    
    try:
        # Import and run drift detection
        from src.experiments.exp08_drift_poc import main as drift_main
        
        # Check for Evidently
        try:
            from evidently.report import Report
            logger.info("Evidently AI available - running drift analysis...")
            # Note: drift_main uses Hydra, so we run it separately
            logger.info("Run separately: python -m src.experiments.exp08_drift_poc")
            return {"status": "requires_separate_run", "command": "python -m src.experiments.exp08_drift_poc"}
        except ImportError:
            logger.warning("Evidently not installed. pip install evidently")
            return {"status": "skipped", "reason": "evidently not installed"}
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return {"status": "error", "error": str(e)}


def run_field_audit() -> Dict[str, Any]:
    """Run field audit (8B)."""
    logger.info("\n" + "=" * 60)
    logger.info("8B: FIELD AUDIT")
    logger.info("=" * 60)
    
    try:
        from src.experiments.exp08_field_audit import run_field_audit as audit_func
        
        # Run audit
        report = audit_func()
        
        if report:
            logger.info(f"Audit complete. Found {len(report.get('candidates', []))} candidate features.")
            return {"status": "completed", "n_candidates": len(report.get('candidates', []))}
        else:
            return {"status": "completed", "n_candidates": 0}
    except Exception as e:
        logger.error(f"Field audit failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 08: DRIFT DETECTION & FIELD AUDIT")
    logger.info("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    experiment_name = cfg.get('experiment_name', 'drift-monitoring')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    mode = cfg.get('mode', 'all')  # 'all', 'drift', or 'audit'
    
    with mlflow.start_run(
        run_name=f"exp08_drift_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment": "exp08", "mode": mode}
    ):
        results = {}
        
        if mode in ['all', 'drift']:
            results['drift'] = run_drift_detection()
        
        if mode in ['all', 'audit']:
            results['audit'] = run_field_audit()
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Save to results registry
        save_result(ExperimentID.EXP8_DRIFT, results)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        
        for name, result in results.items():
            status = result.get('status', 'unknown')
            logger.info(f"  {name}: {status}")
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")
        logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
