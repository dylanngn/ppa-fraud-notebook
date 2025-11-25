"""
Seon Baseline Evaluation Module

This module evaluates the performance of the Seon fraud detection system
that is currently in production. Seon operates BEFORE listing publication,
making binary approve/reject decisions.

Performance metrics are calculated based on:
1. Primary: seon_approved boolean (extracted from auto_approval_criteria_json in ETL)
2. Fallback: first_published_date comparison (when Seon data unavailable)

This represents the TRUE BASELINE we want to beat with our research.
"""

import json
import os
from datetime import timedelta
from typing import Dict, List

import numpy as np
import polars as pl

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
        
        # Fraud-specific metrics
        "fraud_count": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "catch_rate": float(recall),  # Same as recall but named for fraud context
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }


def evaluate_seon_sliding_window(
    df: pl.DataFrame,
    window_days: int = 90,
    step_days: int = 14,
    include_fallback: bool = True
) -> List[Dict]:
    """
    Evaluate Seon performance using sliding window to match our model evaluation.
    
    Args:
        df: DataFrame with fraud_flag, seon_approved, first_published_date, submission_at
        window_days: Size of training window (for consistency, not used by Seon)
        step_days: Step size for sliding window
        include_fallback: Whether to include fallback predictions
        
    Returns:
        List of dictionaries with metrics per window
    """
    print("Evaluating Seon with sliding window...")
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Calculate Seon predictions once
    df = calculate_seon_labels(df)
    
    # Define windows
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=14)
    
    current_date = start_date + window_size
    
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
        
        # Source counts
        source_counts = test_eval["seon_prediction_source"].value_counts()
        source_dict = {row["seon_prediction_source"]: row["count"] 
                      for row in source_counts.to_dicts()}
        
        print(f"Window {test_start.date()} - {test_end.date()}: "
              f"Precision = {precision:.4f}, "
              f"Recall = {recall:.4f}, "
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
            "fraud_count": int(y_true.sum()),
            "fraud_rate": float(y_true.mean()),
            "catch_rate": recall,
            "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        })
        
        current_date += step_size
        window_idx += 1
    
    return results


def run_seon_evaluation(
    window_days: int = 90,
    step_days: int = 14,
    include_fallback: bool = True,
    results_filename: str = "artifacts/results/seon_baseline_results.csv"
):
    """
    Run Seon evaluation and save results.
    
    Args:
        window_days: Window size in days
        step_days: Step size in days
        include_fallback: Whether to use fallback predictions
        results_filename: Output file path
    """
    # Load listing data
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Error: nodes_listing.parquet not found. Run ETL first.")
        return None
    
    print("Loading listing data...")
    df = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    # Check for required columns (seon_approved is already extracted in ETL)
    required_cols = ["seon_approved", "fraud_flag", "first_published_date", "submission_at", "is_fraud"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    print(f"Total listings: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()}")
    
    # Run sliding window evaluation
    results = evaluate_seon_sliding_window(
        df,
        window_days=window_days,
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
    
    print(f"\n{'='*80}")
    print("SEON BASELINE PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Windows evaluated: {len(results)}")
    print(f"Mean Precision: {float(results_df['precision'].mean()):.4f}")
    print(f"Mean Recall: {float(results_df['recall'].mean()):.4f}")
    print(f"Mean F1 Score: {float(results_df['f1_score'].mean()):.4f}")
    print(f"Mean Catch Rate: {float(results_df['catch_rate'].mean()):.4f}")
    print(f"Mean False Alarm Rate: {float(results_df['false_alarm_rate'].mean()):.4f}")
    
    coverage_pct = (results_df['seon_approved_available'].sum() / 
                    results_df['total_test'].sum() * 100)
    print(f"\nSeon Data Coverage: {coverage_pct:.1f}%")
    print(f"Fallback Used: {results_df['fallback_used'].sum()} cases")
    
    print(f"\nResults saved to: {results_filename}")
    print(f"{'='*80}")
    
    return results


def main(window_days: int = 90, step_days: int = 14, include_fallback: bool = True):
    """Main entry point for CLI."""
    run_seon_evaluation(window_days, step_days, include_fallback)


if __name__ == "__main__":
    import typer
    typer.run(main)

