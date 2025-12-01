"""
Seon Baseline Evaluation Module

Usage:
    # Generate static baseline (run once after ETL)
    python -m src.utils.evaluate_seon
    
    # Load baseline for comparison
    from src.utils.evaluate_seon import load_seon_baseline
    baseline = load_seon_baseline()
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import polars as pl

from src.utils.hydra_utils import resolve_path
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)

NODES_LISTING = resolve_path("artifacts/nodes_listing.parquet")
SEON_BASELINE_PATH = resolve_path("artifacts/seon_baseline.json")


def calculate_seon_predictions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Seon prediction labels based on approval criteria.
    
    Logic:
    - seon_approved=true → predict legitimate (0)
    - seon_approved=false → predict fraud (1)
    - Fallback: Use first_published_date if seon_approved is null
    
    Returns:
        DataFrame with 'seon_prediction' column (0/1 or null)
    """
    # Primary: Use seon_approved directly
    df = df.with_columns([
        pl.when(pl.col("seon_approved").is_not_null())
        .then((~pl.col("seon_approved")).cast(pl.Int8))  # Rejected = 1 (fraud)
        .otherwise(None)
        .alias("seon_prediction_primary")
    ])
    
    # Fallback: Use first_published_date
    df = df.with_columns([
        pl.when(pl.col("seon_prediction_primary").is_null())
        .then(
            pl.when(
                pl.col("fraud_flag").is_not_null() & pl.col("first_published_date").is_null()
            ).then(pl.lit(1))  # Blocked fraud
            .when(pl.col("first_published_date").is_not_null())
            .then(pl.lit(0))  # Published = not blocked
            .otherwise(None)
        )
        .otherwise(pl.col("seon_prediction_primary"))
        .alias("seon_prediction")
    ])
    
    return df


def evaluate_seon_on_window(
    df: pl.DataFrame,
    window_start: datetime,
    window_end: datetime
) -> Optional[Dict]:
    """
    Evaluate Seon on a single time window.
    
    Uses centralized calculate_metrics for consistency with model evaluation.
    """
    # Filter to window
    window_data = df.filter(
        (pl.col("submission_at") >= window_start) & 
        (pl.col("submission_at") < window_end)
    )
    
    if len(window_data) == 0:
        return None
    
    # Filter to valid predictions
    eval_data = window_data.filter(pl.col("seon_prediction").is_not_null())
    
    if len(eval_data) == 0:
        return None
    
    # Extract arrays
    y_pred = eval_data["seon_prediction"].to_numpy().astype(float)
    y_true = eval_data["is_fraud"].cast(pl.Int8).to_numpy()
    
    if len(np.unique(y_true)) < 2:
        return None
    
    # Use centralized metrics (with confusion matrix for Seon binary classifier)
    metrics = calculate_metrics(y_true, y_pred, include_confusion_matrix=True)
    
    return {
        "window_start": str(window_start.date()),
        "window_end": str(window_end.date()),
        "total_listings": len(window_data),
        "evaluated": len(eval_data),
        **metrics
    }


def generate_seon_baseline(
    initial_window_days: int = 90,
    step_days: int = 14
) -> Dict:
    """
    Generate Seon baseline evaluation and store as static JSON.
    
    This should be run ONCE after ETL, not for every model comparison.
    
    Args:
        initial_window_days: Days to skip before starting evaluation
        step_days: Step size between evaluation windows
        
    Returns:
        Baseline dict with aggregate metrics and per-window results
    """
    logger.info("Generating Seon baseline (one-time computation)...")
    
    # Load data
    if not NODES_LISTING.exists():
        raise FileNotFoundError("nodes_listing.parquet not found. Run ETL first.")
    
    df = pl.read_parquet(NODES_LISTING)
    
    # Ensure seon_approved column exists
    if "seon_approved" not in df.columns:
        logger.warning("seon_approved not found. Using fallback mode only.")
        df = df.with_columns(pl.lit(None).cast(pl.Boolean).alias("seon_approved"))
    
    # Calculate predictions
    df = calculate_seon_predictions(df)
    df = df.sort("submission_at")
    
    # Window parameters
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    evaluation_start = start_date + timedelta(days=initial_window_days)
    test_size = timedelta(days=14)
    step_size = timedelta(days=step_days)
    
    # Evaluate per window
    current_date = evaluation_start
    window_results = []
    
    while current_date + test_size <= end_date:
        result = evaluate_seon_on_window(df, current_date, current_date + test_size)
        if result:
            window_results.append(result)
        current_date += step_size
    
    if not window_results:
        raise ValueError("No valid evaluation windows. Check data coverage.")
    
    # Aggregate metrics
    agg_metrics = {}
    metric_keys = ["auc_pr", "auc_roc", "precision", "recall", "f1_score", 
                   "p@100", "catch_rate", "false_alarm_rate"]
    
    for key in metric_keys:
        values = [w[key] for w in window_results if key in w]
        if values:
            agg_metrics[f"mean_{key}"] = float(np.mean(values))
            agg_metrics[f"std_{key}"] = float(np.std(values))
    
    # Build baseline
    baseline = {
        "generated_at": datetime.now().isoformat(),
        "data_range": {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
        },
        "config": {
            "initial_window_days": initial_window_days,
            "step_days": step_days,
            "test_window_days": 14,
        },
        "total_listings": len(df),
        "fraud_rate": float(df["is_fraud"].mean()),
        "num_windows": len(window_results),
        "aggregate_metrics": agg_metrics,
        "window_results": window_results,
    }
    
    # Save to file
    SEON_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEON_BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    
    logger.info(f"Saved Seon baseline to {SEON_BASELINE_PATH}")
    logger.info(f"  Windows: {len(window_results)}")
    logger.info(f"  Mean AUC-PR: {agg_metrics.get('mean_auc_pr', 0):.4f}")
    logger.info(f"  Mean P@100: {agg_metrics.get('mean_p@100', 0):.4f}")
    logger.info(f"  Mean Recall: {agg_metrics.get('mean_recall', 0):.4f}")
    
    return baseline


def load_seon_baseline() -> Optional[Dict]:
    """
    Load pre-computed Seon baseline from static file.
    
    Returns:
        Baseline dict or None if not generated yet
    """
    if not SEON_BASELINE_PATH.exists():
        logger.warning("Seon baseline not found. Run: python -m src.utils.evaluate_seon")
        return None
    
    with open(SEON_BASELINE_PATH) as f:
        return json.load(f)


def get_seon_metrics_for_comparison() -> Dict[str, float]:
    """
    Get Seon aggregate metrics for model comparison.
    
    Returns dict with mean metrics that match model output format.
    """
    baseline = load_seon_baseline()
    if baseline is None:
        return {"error": "Seon baseline not found"}
    
    agg = baseline.get("aggregate_metrics", {})
    
    return {
        "mean_auc_pr": agg.get("mean_auc_pr", 0),
        "mean_auc_roc": agg.get("mean_auc_roc", 0),
        "mean_p_at_100": agg.get("mean_p@100", 0),
        "mean_recall": agg.get("mean_recall", 0),
        "mean_precision": agg.get("mean_precision", 0),
        "mean_f1_score": agg.get("mean_f1_score", 0),
        "num_windows": baseline.get("num_windows", 0),
    }


def main():
    """Generate Seon baseline (CLI entry point)."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate static Seon baseline for model comparison")
    parser.add_argument("--initial-window-days", type=int, default=90, help="Days to skip before evaluation")
    parser.add_argument("--step-days", type=int, default=14, help="Step size between windows")
    
    args = parser.parse_args()
    
    generate_seon_baseline(
        initial_window_days=args.initial_window_days,
        step_days=args.step_days
    )


if __name__ == "__main__":
    main()
