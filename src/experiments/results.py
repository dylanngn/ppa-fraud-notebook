"""
Experiment Results Registry.

Provides consistent saving and loading of experiment results.
All experiments should use this module to store their key metrics.

Usage:
    # Saving results (in experiment script)
    from src.experiments.results import save_result
    save_result("exp9_feature_selection", {
        "auc_pr": 0.784,
        "auc_roc": 0.95,
        ...
    })
    
    # Loading results (in another experiment)
    from src.experiments.results import load_result, get_metric
    exp9 = load_result("exp9_feature_selection")
    baseline_auc = get_metric("exp9_feature_selection", "auc_pr")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

RESULTS_DIR = resolve_path("artifacts/results")


def _ensure_results_dir():
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(
    experiment_id: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save experiment results to a JSON file.
    
    Args:
        experiment_id: Unique identifier (e.g., "exp9_feature_selection")
        metrics: Dictionary of metrics to save
        metadata: Optional additional metadata
        
    Returns:
        Path to saved results file
    """
    _ensure_results_dir()
    
    result = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "metadata": metadata or {}
    }
    
    filepath = RESULTS_DIR / f"{experiment_id}.json"
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"Saved results to {filepath}")
    return filepath


def load_result(experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from JSON file.
    
    Args:
        experiment_id: Unique identifier (e.g., "exp9_feature_selection")
        
    Returns:
        Result dictionary or None if not found
    """
    filepath = RESULTS_DIR / f"{experiment_id}.json"
    
    if not filepath.exists():
        logger.warning(f"Results not found: {filepath}")
        return None
    
    with open(filepath) as f:
        return json.load(f)


def get_metric(
    experiment_id: str,
    metric_name: str,
    default: Any = None
) -> Any:
    """
    Get a specific metric from experiment results.
    
    Args:
        experiment_id: Unique identifier (e.g., "exp9_feature_selection")
        metric_name: Name of metric to retrieve
        default: Default value if not found
        
    Returns:
        Metric value or default
        
    Example:
        baseline_auc = get_metric("exp9_feature_selection", "auc_pr", default=0.78)
    """
    result = load_result(experiment_id)
    if result is None:
        return default
    
    return result.get("metrics", {}).get(metric_name, default)


def list_results() -> Dict[str, Dict[str, Any]]:
    """
    List all saved experiment results.
    
    Returns:
        Dictionary mapping experiment_id to result summary
    """
    _ensure_results_dir()
    
    results = {}
    for filepath in RESULTS_DIR.glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                results[data["experiment_id"]] = {
                    "timestamp": data.get("timestamp"),
                    "metrics": list(data.get("metrics", {}).keys())
                }
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
    
    return results


# =============================================================================
# CANONICAL EXPERIMENT IDS
# =============================================================================
# Use these constants to ensure consistent naming across experiments

class ExperimentID:
    """Canonical experiment identifiers (zero-padded for consistent sorting)."""
    EXP1_GRAPH_VALUE = "exp01_graph_value"
    EXP2_ARCHITECTURE = "exp02_architecture"
    EXP3_HYPEROPT = "exp03_hyperopt"
    EXP4_CONCEPT_DRIFT = "exp04_concept_drift"
    EXP5_SHAP = "exp05_shap"
    EXP6_BUSINESS_VALUE = "exp06_business_value"
    EXP7_PRODUCTION = "exp07_production"
    EXP8_DRIFT = "exp08_drift"
    EXP9_FEATURE_SELECTION = "exp09_feature_selection"
    EXP10A_ANOMALY = "exp10a_anomaly"
    EXP10B_CLUSTERING = "exp10b_clustering"
    EXP10C_ASSOCIATION = "exp10c_association"
    
    # Baselines
    XGBOOST_BASELINE = "xgboost_baseline"
    SEON_BASELINE = "seon_baseline"
