"""
Centralized metrics module for fraud detection evaluation.

All evaluation metrics should be computed through this module to ensure consistency.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_confusion_matrix: bool = False
) -> Dict[str, float]:
    """
    Calculates all fraud detection metrics from predictions.
    
    This is the SINGLE SOURCE OF TRUTH for all model evaluation metrics.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted probabilities or scores (higher = more likely fraud)
        include_confusion_matrix: If True, include tp/tn/fp/fn for binary predictions
        
    Returns:
        Dict with metrics:
        - auc_pr, auc_roc: Area under PR and ROC curves
        - p@K, r@K, lift@K: Precision, Recall, Lift at K (K=50,100,200)
        - fraud_count, fraud_rate: Ground truth statistics
        - (optional) accuracy, precision, recall, f1_score, tp, tn, fp, fn
    """
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
            
            metrics[f"p@{k}"] = float(precision_k)
            metrics[f"r@{k}"] = float(recall_k)
            metrics[f"lift@{k}"] = float(lift_k)
        else:
            metrics[f"p@{k}"] = 0.0
            metrics[f"r@{k}"] = 0.0
            metrics[f"lift@{k}"] = 0.0
    
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


def measure_inference_latency(
    model,
    X_sample: np.ndarray,
    n_iterations: int = 100
) -> Dict[str, float]:
    """
    Measure inference latency for a model.
    
    Args:
        model: Model with predict_proba or predict method
        X_sample: Sample input data (small batch, e.g. 100 rows)
        n_iterations: Number of iterations to average
        
    Returns:
        Dict with latency metrics in milliseconds
    """
    import time
    
    # Determine prediction method
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    elif hasattr(model, 'predict'):
        predict_fn = model.predict
    else:
        return {"error": "Model has no predict method"}
    
    # Warmup
    for _ in range(5):
        _ = predict_fn(X_sample)
    
    # Measure
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = predict_fn(X_sample)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "batch_size": len(X_sample),
        "latency_per_sample_ms": float(np.mean(latencies) / len(X_sample)),
    }

