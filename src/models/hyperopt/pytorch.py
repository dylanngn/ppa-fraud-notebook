"""
Hyperparameter optimization for PyTorch GNN models using Optuna and MLflow.
Follows MLflow best practices for hyperparameter tuning.
"""
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import typer
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from src.models.utils.common import get_device, setup_mlflow, filter_graph_by_time
from src.models.config.experiment_config import ExperimentConfig
from src.utils.metrics import calculate_metrics


def create_pytorch_objective(
    model_class,
    data,
    epochs: int = 25,
    split_percent: float = 0.8,
    metric_weights: Optional[Dict[str, float]] = None
) -> Callable:
    """
    Create Optuna objective function for PyTorch GNN hyperparameter optimization.
    
    Args:
        model_class: PyTorch model class (e.g., HGTWrapper, SAGEWrapper)
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
        Optuna objective function for PyTorch hyperparameter tuning.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Negative composite metric score (for minimization)
        """
        # Define hyperparameter search space
        params = {
            "hidden_channels": trial.suggest_int("hidden_channels", 32, 128, step=16),
            "out_channels": trial.suggest_int("out_channels", 32, 128, step=16),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }
        
        # For HGT, add num_heads
        if "HGT" in model_class.__name__:
            params["num_heads"] = trial.suggest_int("num_heads", 2, 8, step=2)
        
        # Create nested MLflow run for this trial
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("epochs", epochs)
            
            try:
                # Temporal split
                timestamps = data['listing'].timestamp.numpy()
                start_threshold = datetime(2023, 1, 1).timestamp() * 1e9
                valid_mask = timestamps >= start_threshold
                valid_timestamps = timestamps[valid_mask]
                
                if len(valid_timestamps) == 0:
                    valid_timestamps = timestamps
                
                split_time = valid_timestamps[int(len(valid_timestamps) * split_percent)]
                
                # Create training subgraph
                train_data, train_edge_times = filter_graph_by_time(data, split_time, return_edge_times=True)
                train_data = train_data.to(device)
                
                # Move edge times to device
                train_edge_times_device = {
                    k: v.to(device) if v is not None else None 
                    for k, v in train_edge_times.items()
                }
                
                # Initialize model
                if "HGT" in model_class.__name__:
                    model = model_class(
                        metadata=train_data.metadata(),
                        hidden_channels=params["hidden_channels"],
                        out_channels=params["out_channels"],
                        num_heads=params["num_heads"],
                        num_layers=params["num_layers"],
                    ).to(device)
                else:
                    model = model_class(
                        metadata=train_data.metadata(),
                        hidden_channels=params["hidden_channels"],
                        out_channels=params["out_channels"],
                        num_layers=params["num_layers"],
                    ).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
                
                train_mask = ((train_data['listing'].timestamp <= split_time) & 
                             (train_data['listing'].timestamp >= start_threshold)).to(device)
                
                # Training loop
                best_loss = float('inf')
                for epoch in range(1, epochs + 1):
                    model.train()
                    optimizer.zero_grad()
                    
                    if "HGT" in model_class.__name__:
                        out = model.predict(
                            train_data.x_dict, 
                            train_data.edge_index_dict, 
                            train_edge_times_device
                        )
                    else:
                        out = model.predict(train_data.x_dict, train_data.edge_index_dict)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        out[train_mask], 
                        train_data['listing'].y[train_mask].float().view(-1, 1)
                    )
                    
                    loss.backward()
                    optimizer.step()
                    
                    if loss < best_loss:
                        best_loss = loss
                
                mlflow.log_metric("best_train_loss", best_loss.item())
                
                # Evaluate on test split
                test_mask = ((data['listing'].timestamp > split_time) & 
                            (data['listing'].timestamp >= start_threshold))
                
                if test_mask.sum() > 0:
                    full_data = data.to(device)
                    test_mask_device = test_mask.to(device)
                    
                    # Prepare edge times for full data
                    full_edge_times_device = {}
                    for edge_type in data.edge_index_dict.keys():
                        if 'timestamp' in data[edge_type]:
                            full_edge_times_device[edge_type] = data[edge_type].timestamp.to(device)
                        else:
                            full_edge_times_device[edge_type] = None
                    
                    model.eval()
                    with torch.no_grad():
                        if "HGT" in model_class.__name__:
                            test_out = model.predict(
                                full_data.x_dict, 
                                full_data.edge_index_dict, 
                                full_edge_times_device
                            )
                        else:
                            test_out = model.predict(full_data.x_dict, full_data.edge_index_dict)
                        
                        test_pred = test_out[test_mask_device].sigmoid().cpu().numpy().flatten()
                        test_y = data['listing'].y[test_mask].cpu().numpy()
                    
                    # Calculate metrics
                    metrics = calculate_metrics(test_y, test_pred)
                    
                    # Calculate composite metric
                    composite_score = (
                        metric_weights.get("auc_pr", 0.7) * metrics["auc_pr"] +
                        metric_weights.get("p@100", 0.3) * metrics["p@100"]
                    )
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "test_auc_pr": metrics["auc_pr"],
                        "test_auc_roc": metrics["auc_roc"],
                        "test_p_at_100": metrics["p@100"],
                        "test_lift_at_100": metrics["lift@100"],
                        "composite_score": composite_score,
                    })
                    
                    # Report to Optuna (minimize negative score = maximize score)
                    return -composite_score
                else:
                    return float('inf')
                    
            except Exception as e:
                mlflow.log_param("error", str(e))
                return float('inf')
    
    return objective


def optimize_pytorch_hyperparameters(
    model_class,
    config: ExperimentConfig,
    n_trials: int = 50,
    epochs: int = 25,
    split_percent: float = 0.8,
    timeout_minutes: Optional[int] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for PyTorch GNN models.
    
    Args:
        model_class: PyTorch model class (e.g., HGTWrapper, SAGEWrapper)
        config: Experiment configuration
        n_trials: Number of Optuna trials
        epochs: Number of training epochs per trial
        split_percent: Train/test split percentage
        timeout_minutes: Optional timeout in minutes
        metric_weights: Weights for composite metric
        study_name: Optional Optuna study name
    
    Returns:
        Dictionary with best parameters and metrics
    """
    import torch
    
    # Setup MLflow
    setup_mlflow(config.experiment_name)
    
    # Load graph data
    data = torch.load("artifacts/graph.pt", weights_only=False)
    
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
        direction="minimize",  # Minimize negative composite score = maximize score
        study_name=study_name or f"pytorch_hyperopt_{config.experiment_name}",
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
        
        # Log best parameters
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_composite_score", -study.best_value)
        
    return {
            "best_params": study.best_params,
            "best_score": -study.best_value,
            "n_trials": len(study.trials),
        }


def main(
    model_type: str = typer.Option(..., help="Model type: 'hgt' or 'sage'"),
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials"),
    epochs: int = typer.Option(25, help="Number of training epochs per trial"),
    timeout_minutes: Optional[int] = typer.Option(None, help="Optional timeout in minutes"),
):
    """
    Hyperparameter optimization for PyTorch GNN models using Optuna.
    
    Tunes GNN hyperparameters to maximize: 0.7 * AUC-PR + 0.3 * P@100
    """
    from src.models.gnn.hgt import HGTWrapper
    from src.models.gnn.sage import SAGEWrapper
    from src.models.config.experiment_config import ExperimentConfig
    
    model_class = HGTWrapper if model_type.lower() == "hgt" else SAGEWrapper
    
    config = ExperimentConfig(experiment_name=experiment_name)
    
    result = optimize_pytorch_hyperparameters(
        model_class=model_class,
        config=config,
        n_trials=n_trials,
        epochs=epochs,
        timeout_minutes=timeout_minutes
    )
    
    typer.echo(f"Best parameters: {result['best_params']}")
    typer.echo(f"Best score: {result['best_score']:.4f}")
    
    return result


if __name__ == "__main__":
    import typer
    typer.run(main)

