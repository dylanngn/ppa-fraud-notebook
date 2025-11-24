import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_metrics(y_true, y_pred):
    """
    Calculates AUC-PR, AUC-ROC, Precision@K, Recall@K, and Lift@K.
    K values: 50, 100, 200.
    """
    metrics = {}
    
    # Basic AUC
    if len(np.unique(y_true)) > 1:
        metrics["auc_pr"] = average_precision_score(y_true, y_pred)
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
    else:
        metrics["auc_pr"] = 0.0
        metrics["auc_roc"] = 0.0

    # Global Fraud Rate
    total_positives = sum(y_true)
    global_rate = total_positives / len(y_true) if len(y_true) > 0 else 0.0
    
    # Top-K Metrics
    sorted_indices = np.argsort(y_pred)[::-1]
    
    for k in [50, 100, 200]:
        if len(y_pred) >= k:
            top_k_indices = sorted_indices[:k]
            top_k_labels = y_true[top_k_indices]
            
            hits = sum(top_k_labels)
            
            precision_k = hits / k
            recall_k = hits / total_positives if total_positives > 0 else 0.0
            lift_k = precision_k / global_rate if global_rate > 0 else 0.0
            
            metrics[f"p@{k}"] = precision_k
            metrics[f"r@{k}"] = recall_k
            metrics[f"lift@{k}"] = lift_k
        else:
            metrics[f"p@{k}"] = 0.0
            metrics[f"r@{k}"] = 0.0
            metrics[f"lift@{k}"] = 0.0
            
    metrics["fraud_count"] = int(total_positives)
    
    return metrics

