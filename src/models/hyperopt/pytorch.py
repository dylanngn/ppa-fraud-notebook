"""
Hyperparameter optimization for PyTorch GNN models using Optuna and MLflow.
Follows MLflow best practices for hyperparameter tuning.

Currently supports: GraphSAGE
"""
import logging
from typing import Any, Callable, Dict, Optional

import hydra
import mlflow
import optuna
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

GRAPH_PT = resolve_path("artifacts/graph.pt")

from src.models.utils.common import get_device, setup_mlflow, filter_graph_by_time
from src.utils.metrics import calculate_metrics
from src.utils.temporal_split import TemporalTrainTestSplitter


def create_pytorch_objective(
    model_class,
    data,
    epochs: int = 25,
    split_percent: float = 0.8,
    metric_weights: Optional[Dict[str, float]] = None
) -> Callable:
    """
    Create Optuna objective function for GraphSAGE hyperparameter optimization.
    
    Args:
        model_class: PyTorch model class (SAGEWrapper)
        data: Full graph data
        epochs: Number of training epochs
        split_percent: Train/test split percentage
        metric_weights: Weights for composite metric (default: 0.7 * AUC-PR + 0.3 * P@100)
    
    Returns:
        Objective function for Optuna
    """
    if metric_weights is None:
        metric_weights = {"auc_pr": 0.7, "p@100": 0.3}
    
    device = get_device()
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for GraphSAGE hyperparameter tuning.
        """
        # Define hyperparameter search space
        params = {
            "hidden_channels": trial.suggest_int("hidden_channels", 32, 128, step=16),
            "out_channels": trial.suggest_int("out_channels", 32, 128, step=16),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }
        
        # Create nested MLflow run for this trial
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("epochs", epochs)
            
            try:
                # Temporal split using utility
                timestamps = data['listing'].timestamp.numpy()
                splitter = TemporalTrainTestSplitter(
                    timestamps=timestamps,
                    split_percent=split_percent,
                    start_threshold_percentile=10
                )
                train_mask, test_mask = splitter.split()
                split_info = splitter.get_split_info()
                split_time = split_info['split_time']
                
                # Create training subgraph
                train_data = filter_graph_by_time(data, split_time)
                train_data = train_data.to(device)
                
                # Initialize model (GraphSAGE only)
                model = model_class(
                    metadata=train_data.metadata(),
                    hidden_channels=params["hidden_channels"],
                    out_channels=params["out_channels"],
                    num_layers=params["num_layers"],
                ).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
                
                train_mask_device = torch.from_numpy(train_mask).to(device)
                
                # Training loop
                best_loss = float('inf')
                for epoch in range(1, epochs + 1):
                    model.train()
                    optimizer.zero_grad()
                    
                    out = model.predict(train_data.x_dict, train_data.edge_index_dict)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        out[train_mask_device], 
                        train_data['listing'].y[train_mask_device].float().view(-1, 1)
                    )
                    
                    loss.backward()
                    optimizer.step()
                    
                    if loss < best_loss:
                        best_loss = loss
                
                mlflow.log_metric("best_train_loss", best_loss.item())
                
                # Evaluate on test split
                if test_mask.sum() > 0:
                    full_data = data.to(device)
                    test_mask_device = torch.from_numpy(test_mask).to(device)
                    
                    model.eval()
                    with torch.no_grad():
                        test_out = model.predict(full_data.x_dict, full_data.edge_index_dict)
                        test_pred = test_out[test_mask_device].sigmoid().cpu().numpy().flatten()
                        test_y = data['listing'].y[test_mask_device].cpu().numpy()
                    
                    # Calculate metrics
                    metrics = calculate_metrics(test_y, test_pred)
                    
                    # Calculate composite metric
                    composite_score = (
                        metric_weights.get("auc_pr", 0.7) * metrics["auc_pr"] +
                        metric_weights.get("p@100", 0.3) * metrics["p@100"]
                    )
                    
                    mlflow.log_metrics({
                        "test_auc_pr": metrics["auc_pr"],
                        "test_auc_roc": metrics["auc_roc"],
                        "test_p_at_100": metrics["p@100"],
                        "test_lift_at_100": metrics["lift@100"],
                        "composite_score": composite_score,
                    })
                    
                    return -composite_score
                else:
                    return float('inf')
                    
            except Exception as e:
                mlflow.log_param("error", str(e))
                return float('inf')
    
    return objective


def optimize_pytorch_hyperparameters(
    model_class,
    experiment_name: str,
    n_trials: int = 50,
    epochs: int = 25,
    split_percent: float = 0.8,
    timeout_minutes: Optional[int] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for GraphSAGE.
    
    Args:
        model_class: PyTorch model class (SAGEWrapper)
        experiment_name: MLflow experiment name
        n_trials: Number of Optuna trials
        epochs: Number of training epochs per trial
        split_percent: Train/test split percentage
        timeout_minutes: Optional timeout in minutes
        metric_weights: Weights for composite metric
        study_name: Optional Optuna study name
    
    Returns:
        Dictionary with best parameters and metrics
    """
    # Setup MLflow
    setup_mlflow(experiment_name)
    
    # Load graph data
    data = torch.load(GRAPH_PT, weights_only=False)
    
    # Create objective function
    objective = create_pytorch_objective(
        model_class=model_class,
        data=data,
        epochs=epochs,
        split_percent=split_percent,
        metric_weights=metric_weights
    )
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name or f"pytorch_hyperopt_{experiment_name}",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    model_name = model_class.__name__.lower().replace("wrapper", "")
    with mlflow.start_run(run_name=f"{model_name}_hyperparameter_optimization"):
        mlflow.log_params({
            "n_trials": n_trials,
            "epochs": epochs,
            "split_percent": split_percent,
        })
        
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout_minutes * 60 if timeout_minutes else None,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            pass
        
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_composite_score", -study.best_value)
        
    return {
        "best_params": study.best_params,
        "best_score": -study.best_value,
        "n_trials": len(study.trials),
    }


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Hyperparameter optimization for GraphSAGE using Optuna.
    
    Usage:
        python -m src.models.hyperopt.pytorch
    """
    from src.models.gnn.sage import SAGEWrapper
    
    # Get config values
    n_trials = cfg.get("hyperopt", {}).get("n_trials", 50)
    epochs = cfg.get("gnn", {}).get("epochs", 25)
    timeout_minutes = cfg.get("hyperopt", {}).get("timeout_minutes", None)
    
    model_class = SAGEWrapper
    
    logger.info(f"Starting GraphSAGE hyperparameter optimization...")
    logger.info(f"  Experiment: {cfg.experiment_name}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Epochs per trial: {epochs}")
    
    result = optimize_pytorch_hyperparameters(
        model_class=model_class,
        experiment_name=cfg.experiment_name,
        n_trials=n_trials,
        epochs=epochs,
        timeout_minutes=timeout_minutes
    )
    
    logger.info("Optimization complete!")
    logger.info(f"  Best parameters: {result['best_params']}")
    logger.info(f"  Best score: {result['best_score']:.4f}")
    
    return result


if __name__ == "__main__":
    main()
