import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_confusion_matrix: bool = False
) -> Dict[str, float]:
    metrics = {}
    
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Basic AUC metrics
    if len(np.unique(y_true)) > 1:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_pred))
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
    else:
        metrics["auc_pr"] = 0.0
        metrics["auc_roc"] = 0.0

    # Fraud statistics
    total_positives = int(np.sum(y_true))
    total_samples = len(y_true)
    fraud_rate = total_positives / total_samples if total_samples > 0 else 0.0
    
    metrics["fraud_count"] = total_positives
    metrics["fraud_rate"] = float(fraud_rate)
    
    # Top-K Metrics
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
    
    # Optional: Confusion matrix and derived metrics (for binary classifiers like Seon)
    if include_confusion_matrix:
        # Threshold predictions at 0.5 for confusion matrix
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        tp = int(np.sum((y_pred_binary == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred_binary == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred_binary == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred_binary == 0) & (y_true == 1)))
        
        metrics["tp"] = tp
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn
        
        total = tp + tn + fp + fn
        metrics["accuracy"] = float((tp + tn) / total) if total > 0 else 0.0
        metrics["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        precision = metrics["precision"]
        recall = metrics["recall"]
        metrics["f1_score"] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Fraud-specific naming
        metrics["catch_rate"] = metrics["recall"]  # Alias for fraud context
        metrics["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return metrics
