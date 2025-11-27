"""
MLflow Model Loader Utilities

Helper functions to load models from MLflow by window index and model type.
Replaces the old pickle-based model loading system.
"""

from typing import Optional, Dict, List, Tuple
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np


def find_window_run(
    model_type: str,
    window_index: int,
    experiment_name: str = "ppa-fraud-detection",
    parent_run_tag: str = "training_mode",
    parent_run_tag_value: str = "accumulating_window"
) -> Optional[str]:
    """
    Find MLflow run ID for a specific model type and window index.
    
    Args:
        model_type: Model type (e.g., "baseline", "baseline_graph", "hybrid_hgt")
        window_index: Window index to find (-1 for latest)
        experiment_name: MLflow experiment name
        parent_run_tag: Tag to filter parent runs
        parent_run_tag_value: Value for parent run tag
        
    Returns:
        Run ID string if found, None otherwise
    """
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        
        # Search for parent runs with matching model_type tag
        parent_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_type = '{model_type}' AND tags.{parent_run_tag} = '{parent_run_tag_value}'",
            order_by=["start_time DESC"],
            max_results=100
        )
        
        if not parent_runs:
            print(f"No parent runs found for model_type='{model_type}'")
            return None
        
        # Get the most recent parent run
        parent_run = parent_runs[0]
        parent_run_id = parent_run.info.run_id
        
        # Search for nested runs (child runs) with matching window_index
        child_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' AND params.window_index = '{window_index}'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if child_runs:
            return child_runs[0].info.run_id
        
        # If window_index not found, try to find latest window
        if window_index == -1:
            # Get all child runs and find the one with highest window_index
            all_child_runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                order_by=["start_time DESC"],
                max_results=1000
            )
            
            if all_child_runs:
                # Find run with highest window_index
                max_window_idx = -1
                latest_run = None
                for run in all_child_runs:
                    try:
                        win_idx = int(run.data.params.get("window_index", "-1"))
                        if win_idx > max_window_idx:
                            max_window_idx = win_idx
                            latest_run = run
                    except (ValueError, TypeError):
                        continue
                
                if latest_run:
                    return latest_run.info.run_id
        
        print(f"Window {window_index} not found for model_type='{model_type}'")
        return None
        
    except Exception as e:
        print(f"Error finding window run: {e}")
        return None


def load_model_from_mlflow(
    model_type: str,
    window_index: int = -1,
    experiment_name: str = "ppa-fraud-detection"
) -> Optional[object]:
    """
    Load XGBoost model from MLflow by model type and window index.
    
    Args:
        model_type: Model type (e.g., "baseline", "baseline_graph", "hybrid_hgt")
        window_index: Window index (-1 for latest)
        experiment_name: MLflow experiment name
        
    Returns:
        Loaded model object, or None if not found
    """
    run_id = find_window_run(model_type, window_index, experiment_name)
    
    if run_id is None:
        return None
    
    try:
        # Load model using MLflow
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.xgboost.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None


def get_window_info(
    model_type: str,
    window_index: int = -1,
    experiment_name: str = "ppa-fraud-detection"
) -> Optional[Dict]:
    """
    Get window information (metrics, params) from MLflow run.
    
    Args:
        model_type: Model type
        window_index: Window index (-1 for latest)
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with window info, metrics, and features, or None
    """
    run_id = find_window_run(model_type, window_index, experiment_name)
    
    if run_id is None:
        return None
    
    try:
        client = MlflowClient()
        run = client.get_run(run_id)
        
        # Extract window info from params
        window_info = {
            "window_idx": int(run.data.params.get("window_index", -1)),
            "window_start": run.data.params.get("window_start", ""),
            "window_end": run.data.params.get("window_end", ""),
            "test_start": run.data.params.get("test_start", ""),
            "test_end": run.data.params.get("test_end", ""),
            "train_size": int(run.data.params.get("train_size", 0)),
            "test_size": int(run.data.params.get("test_size", 0)),
        }
        
        # Extract metrics
        metrics = {
            "auc_pr": run.data.metrics.get("auc_pr", 0.0),
            "auc_roc": run.data.metrics.get("auc_roc", 0.0),
            "p@100": run.data.metrics.get("p_at_100", 0.0),
            "lift@100": run.data.metrics.get("lift_at_100", 0.0),
            "fraud_count": int(run.data.metrics.get("fraud_count", 0)),
        }
        
        # Try to get feature list from model artifact
        # Note: Features are logged via autolog, may need to extract from model
        features = []  # Will be populated if available
        
        return {
            "window_info": window_info,
            "metrics": metrics,
            "features": features,
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/model"
        }
    except Exception as e:
        print(f"Error getting window info: {e}")
        return None


def list_available_windows(
    model_type: str,
    experiment_name: str = "ppa-fraud-detection"
) -> List[int]:
    """
    List all available window indices for a model type.
    
    Args:
        model_type: Model type
        experiment_name: MLflow experiment name
        
    Returns:
        List of window indices (sorted)
    """
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        
        # Find parent run
        parent_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_type = '{model_type}' AND tags.training_mode = 'accumulating_window'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not parent_runs:
            return []
        
        parent_run_id = parent_runs[0].info.run_id
        
        # Get all child runs
        child_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["start_time ASC"],
            max_results=1000
        )
        
        window_indices = []
        for run in child_runs:
            try:
                win_idx = int(run.data.params.get("window_index", "-1"))
                if win_idx >= 0:
                    window_indices.append(win_idx)
            except (ValueError, TypeError):
                continue
        
        return sorted(window_indices)
        
    except Exception as e:
        print(f"Error listing windows: {e}")
        return []


def load_model_bundle(
    model_type: str,
    window_index: int = -1,
    experiment_name: str = "ppa-fraud-detection"
) -> Optional[Dict]:
    """
    Load complete model bundle (model + info) from MLflow.
    Replaces the old load_saved_model() function that loaded from pickle.
    
    Args:
        model_type: Model type
        window_index: Window index (-1 for latest)
        experiment_name: MLflow experiment name
        
    Returns:
        Dictionary with 'model', 'window_info', 'metrics', 'features', 'run_id'
    """
    # Get window info
    info = get_window_info(model_type, window_index, experiment_name)
    if info is None:
        return None
    
    # Load model
    model = load_model_from_mlflow(model_type, window_index, experiment_name)
    if model is None:
        return None
    
    # Try to extract features from model
    # XGBoost models have get_booster().feature_names
    try:
        if hasattr(model, 'get_booster'):
            features = model.get_booster().feature_names
            if features:
                info["features"] = features
    except Exception:
        pass
    
    return {
        "model": model,
        "window_info": info["window_info"],
        "metrics": info["metrics"],
        "features": info["features"],
        "run_id": info["run_id"],
        "model_uri": info["model_uri"]
    }

