"""
Metrics for Fraud Detection Evaluation.

Includes:
- AUC-PR (primary optimization metric)
- AUC-ROC
- Precision/Recall at top-K
- Tier-based metrics at multiple thresholds
- Lift metrics
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict, List, Optional


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_confusion_matrix: bool = False,
    include_tier_metrics: bool = True,
) -> Dict[str, float]:
    """
    Calculate comprehensive fraud detection metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted probabilities (0.0-1.0)
        include_confusion_matrix: Include confusion matrix at threshold 0.5
        include_tier_metrics: Include precision/recall at multiple thresholds
        
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # ==========================================
    # Core AUC Metrics (threshold-agnostic)
    # ==========================================
    if len(np.unique(y_true)) > 1:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_pred))
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
    else:
        metrics["auc_pr"] = 0.0
        metrics["auc_roc"] = 0.0

    # ==========================================
    # Fraud Statistics
    # ==========================================
    total_positives = int(np.sum(y_true))
    total_negatives = int(len(y_true) - total_positives)
    total_samples = len(y_true)
    fraud_rate = total_positives / total_samples if total_samples > 0 else 0.0
    
    metrics["fraud_count"] = total_positives
    metrics["fraud_rate"] = float(fraud_rate)
    
    # ==========================================
    # Top-K Metrics (ranking quality)
    # ==========================================
    sorted_indices = np.argsort(y_pred)[::-1]
    
    for k in [50, 100, 200]:
        if len(y_pred) >= k:
            top_k_indices = sorted_indices[:k]
            top_k_labels = y_true[top_k_indices]
            
            hits = int(np.sum(top_k_labels))
            
            precision_k = hits / k
            recall_k = hits / total_positives if total_positives > 0 else 0.0
            lift_k = precision_k / fraud_rate if fraud_rate > 0 else 0.0
            
            metrics[f"p_at_{k}"] = float(precision_k)
            metrics[f"r_at_{k}"] = float(recall_k)
            metrics[f"lift_at_{k}"] = float(lift_k)
        else:
            metrics[f"p_at_{k}"] = 0.0
            metrics[f"r_at_{k}"] = 0.0
            metrics[f"lift_at_{k}"] = 0.0
    
    # ==========================================
    # Tier-Based Metrics (at multiple thresholds)
    # ==========================================
    if include_tier_metrics:
        tier_metrics = calculate_tier_metrics(y_true, y_pred)
        metrics.update(tier_metrics)
    
    # ==========================================
    # Confusion Matrix at threshold 0.5
    # ==========================================
    if include_confusion_matrix:
        cm_metrics = calculate_confusion_metrics(y_true, y_pred, threshold=0.5)
        metrics.update(cm_metrics)
    
    return metrics


def calculate_tier_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 at multiple thresholds.
    
    Useful for:
    - Understanding performance across risk tiers
    - Choosing operational threshold
    - Comparing model behavior at different operating points
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]
    
    metrics = {}
    total_positives = int(np.sum(y_true))
    total_negatives = int(len(y_true) - total_positives)
    
    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)
        
        tp = int(np.sum((y_pred_binary == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred_binary == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred_binary == 0) & (y_true == 1)))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Count how many would be flagged at this threshold
        flagged_count = int(np.sum(y_pred_binary))
        flagged_pct = flagged_count / len(y_pred) * 100
        
        t_str = str(t).replace(".", "")
        metrics[f"p_at_t{t_str}"] = float(precision)
        metrics[f"r_at_t{t_str}"] = float(recall)
        metrics[f"f1_at_t{t_str}"] = float(f1)
        metrics[f"flagged_at_t{t_str}"] = flagged_count
        metrics[f"flagged_pct_t{t_str}"] = float(flagged_pct)
    
    return metrics


def calculate_confusion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate confusion matrix and derived metrics at a specific threshold."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tp = int(np.sum((y_pred_binary == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred_binary == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred_binary == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred_binary == 0) & (y_true == 1)))
    
    total = tp + tn + fp + fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / total) if total > 0 else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "catch_rate": float(recall),  # Alias for fraud context
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Dict[str, float]:
    """
    Find optimal threshold based on target metric.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        metric: "f1", "precision", or "recall"
        min_precision: Minimum precision constraint
        min_recall: Minimum recall constraint
        
    Returns:
        Dict with optimal threshold and metrics at that point
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_threshold = 0.5
    best_value = 0.0
    best_metrics = {}
    
    total_positives = int(np.sum(y_true))
    
    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Check constraints
        if min_precision is not None and precision < min_precision:
            continue
        if min_recall is not None and recall < min_recall:
            continue
        
        # Get target metric value
        if metric == "f1":
            value = f1
        elif metric == "precision":
            value = precision
        elif metric == "recall":
            value = recall
        else:
            value = f1
        
        if value > best_value:
            best_value = value
            best_threshold = t
            best_metrics = {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
    
    return best_metrics


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable table."""
    lines = []
    
    # Core metrics
    lines.append("=== Core Metrics ===")
    lines.append(f"AUC-PR:  {metrics.get('auc_pr', 0):.4f}")
    lines.append(f"AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
    lines.append("")
    
    # Tier metrics
    lines.append("=== Performance by Threshold ===")
    lines.append("Threshold | Precision | Recall | F1     | Flagged")
    lines.append("----------|-----------|--------|--------|--------")
    
    for t in ["03", "05", "07", "09"]:
        p = metrics.get(f"p_at_t{t}", 0)
        r = metrics.get(f"r_at_t{t}", 0)
        f1 = metrics.get(f"f1_at_t{t}", 0)
        flagged = metrics.get(f"flagged_at_t{t}", 0)
        t_display = f"0.{t[1]}" if t.startswith("0") else t
        lines.append(f"    {t_display}   | {p:>8.1%} | {r:>5.1%} | {f1:.3f} | {flagged:>6}")
    
    lines.append("")
    
    # Top-K metrics
    lines.append("=== Top-K Metrics ===")
    for k in [50, 100, 200]:
        p = metrics.get(f"p_at_{k}", 0)
        r = metrics.get(f"r_at_{k}", 0)
        lift = metrics.get(f"lift_at_{k}", 0)
        lines.append(f"Top-{k}: P={p:.1%}, R={r:.1%}, Lift={lift:.1f}x")
    
    return "\n".join(lines)
