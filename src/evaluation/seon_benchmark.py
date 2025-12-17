"""
SEON Benchmark Evaluation.

Compares model performance against SEON baselines:
1. SEON State (DECLINE = fraud) - rule-based binary classifier
2. SEON fraud_score - continuous risk score

Usage:
    from src.evaluation.seon_benchmark import evaluate_seon_benchmark
    results = evaluate_seon_benchmark(test_df)
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import polars as pl
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)


@dataclass
class SEONBenchmarkResult:
    """Results from SEON benchmark evaluation."""
    
    # SEON State (DECLINE = fraud)
    state_precision: float
    state_recall: float
    state_f1: float
    state_tp: int
    state_fp: int
    state_tn: int
    state_fn: int
    
    # SEON fraud_score
    score_auc_roc: float
    score_auc_pr: float
    
    # Fraud rates by state
    decline_fraud_rate: float
    approve_fraud_rate: float
    review_fraud_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seon_state_precision": self.state_precision,
            "seon_state_recall": self.state_recall,
            "seon_state_f1": self.state_f1,
            "seon_score_auc_roc": self.score_auc_roc,
            "seon_score_auc_pr": self.score_auc_pr,
        }


def evaluate_seon_benchmark(
    test_df: pl.DataFrame,
    label_column: str = "is_fraud",
    state_column: str = "state",
    score_column: str = "fraud_score",
) -> SEONBenchmarkResult:
    """
    Evaluate SEON's performance as a baseline.
    
    Args:
        test_df: Test DataFrame with labels and SEON columns
        label_column: Name of fraud label column
        state_column: Name of SEON state column
        score_column: Name of SEON fraud score column
        
    Returns:
        SEONBenchmarkResult with all metrics
    """
    # Fraud rates by state
    state_rates = {}
    for state in ["DECLINE", "APPROVE", "REVIEW"]:
        subset = test_df.filter(pl.col(state_column) == state)
        if subset.height > 0:
            rate = subset[label_column].sum() / subset.height
            state_rates[state.lower()] = rate
        else:
            state_rates[state.lower()] = 0.0
    
    # SEON State metrics (exclude REVIEW - manual queue)
    test_binary = test_df.filter(pl.col(state_column).is_in(["APPROVE", "DECLINE"]))
    
    if test_binary.height == 0:
        logger.warning("No APPROVE/DECLINE events in test set")
        return None
    
    y_true = test_binary[label_column].to_numpy()
    y_pred_state = (test_binary[state_column] == "DECLINE").to_numpy().astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_state).ravel()
    
    state_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    state_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    state_f1 = 2 * state_precision * state_recall / (state_precision + state_recall) \
        if (state_precision + state_recall) > 0 else 0
    
    # SEON score metrics
    test_score = test_df.filter(pl.col(score_column).is_not_null())
    
    if test_score.height == 0:
        logger.warning("No fraud_score values in test set")
        score_auc_roc = 0.0
        score_auc_pr = 0.0
    else:
        y_true_score = test_score[label_column].to_numpy()
        y_score = test_score[score_column].to_numpy() / 100.0  # Normalize to 0-1
        
        score_auc_roc = roc_auc_score(y_true_score, y_score)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true_score, y_score)
        score_auc_pr = auc(recall_arr, precision_arr)
    
    return SEONBenchmarkResult(
        state_precision=state_precision,
        state_recall=state_recall,
        state_f1=state_f1,
        state_tp=int(tp),
        state_fp=int(fp),
        state_tn=int(tn),
        state_fn=int(fn),
        score_auc_roc=score_auc_roc,
        score_auc_pr=score_auc_pr,
        decline_fraud_rate=state_rates.get("decline", 0),
        approve_fraud_rate=state_rates.get("approve", 0),
        review_fraud_rate=state_rates.get("review", 0),
    )


def compare_model_vs_seon(
    model_metrics: Dict[str, float],
    seon_result: SEONBenchmarkResult,
) -> str:
    """
    Generate comparison summary between model and SEON.
    
    Args:
        model_metrics: Dict with model's metrics (auc_pr, auc_roc, precision, recall, f1_score)
        seon_result: SEON benchmark results
        
    Returns:
        Formatted comparison string
    """
    lines = [
        "",
        "=" * 60,
        "MODEL vs SEON COMPARISON",
        "=" * 60,
        "",
        "| Metric        | Our Model | SEON State | SEON Score |",
        "|---------------|-----------|------------|------------|",
    ]
    
    # AUC-PR
    model_auc_pr = model_metrics.get("auc_pr", 0)
    lines.append(
        f"| AUC-PR        | {model_auc_pr:>9.4f} | {'N/A':>10} | {seon_result.score_auc_pr:>10.4f} |"
    )
    
    # AUC-ROC
    model_auc_roc = model_metrics.get("auc_roc", 0)
    lines.append(
        f"| AUC-ROC       | {model_auc_roc:>9.4f} | {'N/A':>10} | {seon_result.score_auc_roc:>10.4f} |"
    )
    
    # Precision
    model_precision = model_metrics.get("precision", 0)
    lines.append(
        f"| Precision     | {model_precision:>9.4f} | {seon_result.state_precision:>10.4f} | {'varies':>10} |"
    )
    
    # Recall
    model_recall = model_metrics.get("recall", 0)
    lines.append(
        f"| Recall        | {model_recall:>9.4f} | {seon_result.state_recall:>10.4f} | {'varies':>10} |"
    )
    
    # F1
    model_f1 = model_metrics.get("f1_score", 0)
    lines.append(
        f"| F1 Score      | {model_f1:>9.4f} | {seon_result.state_f1:>10.4f} | {'varies':>10} |"
    )
    
    lines.append("")
    lines.append("Notes:")
    lines.append(f"  - SEON State: DECLINE = predict fraud (high recall, aggressive)")
    lines.append(f"  - SEON State FP: {seon_result.state_fp} false alarms")
    lines.append(f"  - SEON State FN: {seon_result.state_fn} missed fraud")
    lines.append(f"  - Model AUC-PR improvement: {(model_auc_pr / seon_result.score_auc_pr - 1) * 100:.1f}%")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    from datetime import datetime, timezone
    from src.features import FeatureStore, FEATURE_SCHEMA, TemporalSplitter, TemporalSplitConfig, LabelPropagation
    
    logging.basicConfig(level=logging.INFO)
    
    train_end = datetime(2025, 6, 1, tzinfo=timezone.utc)
    test_end = datetime(2025, 7, 1, tzinfo=timezone.utc)
    
    store = FeatureStore("artifacts/merged_events.parquet")
    df = store.load_data().collect()
    df = df.filter(pl.col(FEATURE_SCHEMA.time_column) >= datetime(2024, 12, 1, tzinfo=timezone.utc))
    
    split_config = TemporalSplitConfig(
        time_column=FEATURE_SCHEMA.time_column,
        insertion_column=FEATURE_SCHEMA.temporal_config.insertion_id_column,
        label_column=FEATURE_SCHEMA.target,
        fraud_flag_column=FEATURE_SCHEMA.raw_target_source,
        gap_days=7,
        label_propagation=LabelPropagation.POINT_IN_TIME,
    )
    splitter = TemporalSplitter(split_config)
    _, test_df = splitter.split(df, train_end, test_end)
    
    result = evaluate_seon_benchmark(test_df)
    
    print("SEON Benchmark Results:")
    print(f"  State Precision: {result.state_precision:.4f}")
    print(f"  State Recall: {result.state_recall:.4f}")
    print(f"  State F1: {result.state_f1:.4f}")
    print(f"  Score AUC-ROC: {result.score_auc_roc:.4f}")
    print(f"  Score AUC-PR: {result.score_auc_pr:.4f}")
