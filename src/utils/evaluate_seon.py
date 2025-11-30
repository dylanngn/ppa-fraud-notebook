"""
Seon Baseline Evaluation Module

This module evaluates the performance of the Seon fraud detection system
that is currently in production. Seon operates BEFORE listing publication,
making binary approve/reject decisions.

Performance metrics are calculated based on:
1. Primary: seon_approved boolean (extracted from auto_approval_criteria_json in ETL)
2. Fallback: first_published_date comparison (when Seon data unavailable)

This represents the TRUE BASELINE we want to beat with our research.

Note: This is evaluation-only (no training). The "window" strategy is used
to evaluate Seon on different time periods to match our model evaluation approach.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import mlflow
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.metrics import calculate_metrics

def parse_seon_approval(auto_approval_criteria_col: pl.Series) -> pl.Series:
    """
    Parse auto_approval_criteria to extract seonApproved boolean.
    
    The structure is: auto_approval_criteria.criteria.criteria.seonApproved
    
    Args:
        auto_approval_criteria_col: Polars Series containing JSON/text data
        
    Returns:
        Boolean Series indicating Seon approval status (null if unavailable)
    """
    def extract_seon_approved(criteria_value):
        """Extract seonApproved from nested JSON structure."""
        if criteria_value is None:
            return None
        
        try:
            # Handle if it's already a dict (from JSONB)
            if isinstance(criteria_value, dict):
                data = criteria_value
            else:
                # Parse as JSON string
                data = json.loads(criteria_value)
            
            # Navigate nested structure: criteria.criteria.seonApproved
            criteria = data.get('criteria', {})
            if criteria is None:
                return None
                
            inner_criteria = criteria.get('criteria', {})
            if inner_criteria is None:
                return None
                
            seon_approved = inner_criteria.get('seonApproved')
            return seon_approved if isinstance(seon_approved, bool) else None
            
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None
    
    # Apply extraction function
    return auto_approval_criteria_col.map_elements(
        extract_seon_approved, 
        return_dtype=pl.Boolean
    )


def calculate_seon_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Seon prediction labels based on approval criteria and fraud flags.
    
    Logic:
    Primary (if seon_approved is available):
        1. seon_approved=true AND fraud_flag=null → TN (correctly allowed)
        2. seon_approved=true AND fraud_flag!=null → FN (missed fraud)
        3. seon_approved=false AND fraud_flag!=null → TP (caught fraud)
        4. seon_approved=false AND fraud_flag=null → FP (false alarm)
    
    Fallback (if seon_approved is null, use first_published_date):
        1. first_published_date < fraud_flag → FN (published then flagged)
        2. fraud_flag!=null AND first_published_date=null → TP (blocked fraud)
        3. fraud_flag=null AND first_published_date!=null → TN (published, no fraud)
        (Note: Cannot detect FP in fallback mode)
    
    Args:
        df: DataFrame with columns: seon_approved, fraud_flag, first_published_date
        
    Returns:
        DataFrame with added columns:
            - seon_prediction: 1 (reject) or 0 (approve)
            - seon_prediction_source: "seon_approved" or "published_date_fallback"
    """
    # seon_approved is already extracted from auto_approval_criteria_json in ETL process
    # No need to parse - just use the existing column
    
    # Primary logic: Use seon_approved when available
    df = df.with_columns([
        pl.when(pl.col("seon_approved").is_not_null())
        .then(
            # Seon rejected (false) = predict fraud (1)
            # Seon approved (true) = predict legitimate (0)
            (~pl.col("seon_approved")).cast(pl.Int8)
        )
        .otherwise(None)
        .alias("seon_prediction_primary")
    ])
    
    # Fallback logic: Use first_published_date
    df = df.with_columns([
        pl.when(pl.col("seon_prediction_primary").is_null())
        .then(
            pl.when(
                # If fraud_flag exists but not published → Seon blocked it (TP)
                pl.col("fraud_flag").is_not_null() & pl.col("first_published_date").is_null()
            ).then(pl.lit(1))  # Predict fraud
            .when(
                # If published then flagged → Seon missed it (FN) 
                # But we mark as "approved" (0) since Seon let it through
                pl.col("first_published_date").is_not_null()
            ).then(pl.lit(0))  # Predict legitimate
            .otherwise(None)  # Cannot determine
        )
        .otherwise(pl.col("seon_prediction_primary"))
        .alias("seon_prediction_fallback")
    ])
    
    # Final prediction and source tracking
    df = df.with_columns([
        pl.coalesce([
            pl.col("seon_prediction_primary"),
            pl.col("seon_prediction_fallback")
        ]).alias("seon_prediction"),
        
        pl.when(pl.col("seon_prediction_primary").is_not_null())
        .then(pl.lit("seon_approved"))
        .when(pl.col("seon_prediction_fallback").is_not_null())
        .then(pl.lit("published_date_fallback"))
        .otherwise(pl.lit("unavailable"))
        .alias("seon_prediction_source")
    ])
    
    return df


def evaluate_seon_performance(
    df: pl.DataFrame,
    include_fallback: bool = True
) -> Dict:
    """
    Evaluate Seon's performance on the dataset.
    
    Args:
        df: DataFrame with fraud_flag, seon_approved, first_published_date
        include_fallback: Whether to include fallback predictions in metrics
        
    Returns:
        Dictionary with performance metrics and source breakdown
    """
    # Calculate Seon predictions
    df = calculate_seon_labels(df)
    
    # Filter to valid predictions
    if include_fallback:
        df_eval = df.filter(pl.col("seon_prediction").is_not_null())
    else:
        df_eval = df.filter(pl.col("seon_prediction_source") == "seon_approved")
    
    if len(df_eval) == 0:
        return {
            "error": "No valid Seon predictions found",
            "total_listings": len(df),
            "seon_approved_available": 0,
            "fallback_used": 0
        }
    
    # Extract predictions and labels
    y_pred = df_eval["seon_prediction"].to_numpy()
    y_true = df_eval["is_fraud"].cast(pl.Int8).to_numpy()
    
    # Calculate confusion matrix
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AUC-PR and AUC-ROC (using probabilities = predictions for binary case)
    # For binary predictions, we use predictions as probabilities
    y_pred_proba = y_pred.astype(float)
    if len(np.unique(y_true)) > 1:
        auc_pr = average_precision_score(y_true, y_pred_proba)
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    else:
        auc_pr = 0.0
        auc_roc = 0.0
    
    # Source breakdown
    source_counts = df_eval["seon_prediction_source"].value_counts()
    source_dict = {row["seon_prediction_source"]: row["count"] 
                   for row in source_counts.to_dicts()}
    
    return {
        "total_listings": len(df),
        "evaluated_listings": len(df_eval),
        "seon_approved_available": source_dict.get("seon_approved", 0),
        "fallback_used": source_dict.get("published_date_fallback", 0),
        "unavailable": len(df) - len(df_eval),
        
        # Confusion matrix
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        
        # Standard metrics
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_pr": float(auc_pr),
        "auc_roc": float(auc_roc),
        
        # Fraud-specific metrics
        "fraud_count": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "catch_rate": float(recall),  # Same as recall but named for fraud context
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }


def evaluate_seon_evaluation_windows(
    df: pl.DataFrame,
    evaluation_start_days: int = 90,
    step_days: int = 14,
    include_fallback: bool = True
) -> List[Dict]:
    """
    Evaluate Seon performance using sliding test windows to match model evaluation.
    
    Note: This is evaluation-only (no training). Seon uses sliding test windows
    that match exactly how our trained models are evaluated on test sets.
    
    Models use:
    - Training: Accumulating window (all historical data)
    - Test: Sliding window (fixed 14-day windows that move forward)
    
    Seon evaluation matches the model test windows (sliding), since there's no training.
    
    Args:
        df: DataFrame with fraud_flag, seon_approved, first_published_date, submission_at
        evaluation_start_days: Days to skip before starting evaluation (matches model initial_window_days)
        step_days: Step size between evaluation windows (matches model step_days)
        include_fallback: Whether to include fallback predictions
        
    Returns:
        List of dictionaries with metrics per evaluation window
    """
    print("Evaluating Seon with sliding test windows (matching model evaluation)...")
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Calculate Seon predictions once
    df = calculate_seon_labels(df)
    
    # Define windows - matching model evaluation exactly
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    # Start evaluation after initial period (matches model's initial_window_days)
    evaluation_start = start_date + timedelta(days=evaluation_start_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=14)  # Match model evaluation test window size
    
    current_date = evaluation_start
    
    results = []
    window_idx = 0
    
    while current_date + test_size <= end_date:
        test_start = current_date
        test_end = current_date + test_size
        
        # Get test window data (matching our model evaluation)
        test_data = df.filter(
            (pl.col("submission_at") >= test_start) & 
            (pl.col("submission_at") < test_end)
        )
        
        if len(test_data) == 0:
            current_date += step_size
            window_idx += 1
            continue
        
        # Filter to valid predictions
        if include_fallback:
            test_eval = test_data.filter(pl.col("seon_prediction").is_not_null())
        else:
            test_eval = test_data.filter(pl.col("seon_prediction_source") == "seon_approved")
        
        if len(test_eval) == 0:
            print(f"Window {test_start.date()} - {test_end.date()}: No valid Seon predictions")
            current_date += step_size
            window_idx += 1
            continue
        
        # Extract predictions and labels
        y_pred = test_eval["seon_prediction"].to_numpy()
        y_true = test_eval["is_fraud"].cast(pl.Int8).to_numpy()
        
        # Calculate confusion matrix
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        
        total = tp + tn + fp + fn
        
        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC-PR and AUC-ROC
        y_pred_proba = y_pred.astype(float)
        if len(np.unique(y_true)) > 1:
            auc_pr = average_precision_score(y_true, y_pred_proba)
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        else:
            auc_pr = 0.0
            auc_roc = 0.0
        
        # Calculate P@100 and other top-K metrics
        metrics = calculate_metrics(y_true, y_pred_proba)
        
        # Source counts
        source_counts = test_eval["seon_prediction_source"].value_counts()
        source_dict = {row["seon_prediction_source"]: row["count"] 
                      for row in source_counts.to_dicts()}
        
        print(f"Window {test_start.date()} - {test_end.date()}: "
              f"Precision = {precision:.4f}, "
              f"Recall = {recall:.4f}, "
              f"AUC-PR = {auc_pr:.4f}, "
              f"F1 = {f1:.4f}, "
              f"Fraud Count = {y_true.sum()}")
        
        results.append({
            "window_idx": window_idx,
            "window_start": test_start,
            "window_end": test_end,
            "total_test": len(test_data),
            "evaluated": len(test_eval),
            "seon_approved_available": source_dict.get("seon_approved", 0),
            "fallback_used": source_dict.get("published_date_fallback", 0),
            
            # Confusion matrix
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            
            # Metrics
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "p_at_100": metrics.get("p@100", 0.0),
            "lift_at_100": metrics.get("lift@100", 0.0),
            "fraud_count": int(y_true.sum()),
            "fraud_rate": float(y_true.mean()),
            "catch_rate": recall,
            "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        })
        
        current_date += step_size
        window_idx += 1
    
    return results


def run_seon_evaluation(
    evaluation_start_days: int = 90,
    step_days: int = 14,
    include_fallback: bool = True,
    log_to_mlflow: bool = True,
    results_filename: Optional[str] = None
) -> Optional[List[Dict]]:
    """
    Run Seon evaluation and save results, optionally logging to MLflow.
    
    Args:
        evaluation_start_days: Days to skip before starting evaluation
        step_days: Step size between evaluation windows
        include_fallback: Whether to use fallback predictions
        log_to_mlflow: Whether to log metrics to MLflow for comparison
        results_filename: Output file path (default: artifacts/results/seon_baseline_results.csv)
        
    Returns:
        List of evaluation results or None if error
    """
    if results_filename is None:
        results_filename = "artifacts/results/seon_baseline_results.csv"
    
    # Load listing data
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Error: nodes_listing.parquet not found. Run ETL first.")
        return None
    
    print("Loading listing data...")
    df = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    # Check for required columns (seon_approved is extracted from flattened data in create_graph_artifacts)
    required_cols = ["fraud_flag", "first_published_date", "submission_at", "is_fraud"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # seon_approved may not exist if field is 100% null (will use fallback mode)
    if "seon_approved" not in df.columns:
        print("Warning: seon_approved column not found. Will use fallback mode (first_published_date) only.")
        df = df.with_columns(pl.lit(None).cast(pl.Boolean).alias("seon_approved"))
    
    print(f"Total listings: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()}")
    
    # Run evaluation windows
    results = evaluate_seon_evaluation_windows(
        df,
        evaluation_start_days=evaluation_start_days,
        step_days=step_days,
        include_fallback=include_fallback
    )
    
    if not results:
        print("No results generated.")
        return None
    
    # Save results
    os.makedirs("artifacts/results", exist_ok=True)
    results_df = pl.DataFrame(results)
    results_df.write_csv(results_filename)
    
    # Calculate aggregate metrics
    mean_precision = float(results_df['precision'].mean())
    mean_recall = float(results_df['recall'].mean())
    mean_f1 = float(results_df['f1_score'].mean())
    mean_auc_pr = float(results_df['auc_pr'].mean())
    mean_auc_roc = float(results_df['auc_roc'].mean())
    mean_p100 = float(results_df['p_at_100'].mean())
    mean_catch_rate = float(results_df['catch_rate'].mean())
    mean_false_alarm_rate = float(results_df['false_alarm_rate'].mean())
    
    # Log to MLflow if requested
    if log_to_mlflow:
        print("\nLogging to MLflow...")
        mlflow.set_tracking_uri("sqlite:///fraud-detection-mlflow.db")
        mlflow.set_experiment("ppa-fraud-detection")
        
        with mlflow.start_run(
            run_name=f"seon_baseline_{datetime.now().strftime('%Y%m%d_%H%M')}",
            tags={
                "model_type": "seon_baseline",
                "evaluation_mode": "sliding_test_windows",
                "baseline": "true"
            }
        ) as run:
            # Log parameters
            mlflow.log_params({
                "evaluation_start_days": evaluation_start_days,
                "step_days": step_days,
                "include_fallback": include_fallback,
                "num_windows": len(results),
                "total_listings": len(df),
                "fraud_rate": float(df['is_fraud'].mean()),
            })
            
            # Log aggregate metrics
            mlflow.log_metrics({
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_f1_score": mean_f1,
                "mean_auc_pr": mean_auc_pr,
                "mean_auc_roc": mean_auc_roc,
                "mean_p_at_100": mean_p100,
                "mean_catch_rate": mean_catch_rate,
                "mean_false_alarm_rate": mean_false_alarm_rate,
            })
            
            # Log results CSV as artifact
            mlflow.log_artifact(results_filename, artifact_path="results")
            
            # Log dataset info
            try:
                dataset = mlflow.data.from_polars(
                    df.head(1000),  # Sample for dataset logging
                    name="seon_baseline_evaluation",
                    targets="is_fraud"
                )
                mlflow.log_input(dataset, context="evaluation")
            except Exception as e:
                print(f"Warning: Could not log dataset: {e}")
            
            print(f"✓ Logged to MLflow run: {run.info.run_id}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SEON BASELINE PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Windows evaluated: {len(results)}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Mean AUC-PR: {mean_auc_pr:.4f}")
    print(f"Mean AUC-ROC: {mean_auc_roc:.4f}")
    print(f"Mean P@100: {mean_p100:.4f}")
    print(f"Mean Catch Rate: {mean_catch_rate:.4f}")
    print(f"Mean False Alarm Rate: {mean_false_alarm_rate:.4f}")
    
    coverage_pct = (results_df['seon_approved_available'].sum() / 
                    results_df['total_test'].sum() * 100)
    print(f"\nSeon Data Coverage: {coverage_pct:.1f}%")
    print(f"Fallback Used: {results_df['fallback_used'].sum()} cases")
    
    print(f"\nResults saved to: {results_filename}")
    if log_to_mlflow:
        print("Metrics logged to MLflow for comparison with trained models")
    print(f"{'='*80}")
    
    return results


def main(
    evaluation_start_days: int = typer.Option(90, help="Days to skip before starting evaluation"),
    step_days: int = typer.Option(14, help="Step size between evaluation windows"),
    include_fallback: bool = typer.Option(True, help="Whether to include fallback predictions"),
    log_to_mlflow: bool = typer.Option(True, help="Whether to log metrics to MLflow"),
    results_filename: Optional[str] = typer.Option(None, help="Output file path for results CSV")
):
    """
    Evaluate Seon performance using sliding test windows.
    
    Matches model evaluation strategy for fair comparison.
    """
    run_seon_evaluation(
        evaluation_start_days=evaluation_start_days,
        step_days=step_days,
        include_fallback=include_fallback,
        log_to_mlflow=log_to_mlflow,
        results_filename=results_filename
    )


if __name__ == "__main__":
    import typer
    typer.run(main)

