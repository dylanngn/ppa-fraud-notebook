"""
Experiment 3: Hyperparameter Optimization

Hyperparameter optimization for XGBoost models using Optuna and MLflow.
Follows MLflow best practices for hyperparameter tuning.

Run with: python -m src.experiments.exp3_hyperopt experiment_name=xgboost-hyperopt
"""
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import hydra
import mlflow
import optuna
import polars as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from src.models.xgboost.trainer import train_accumulating_window
from src.features.definitions.base import compute_base_features
from src.data.loader import load_data


def create_xgboost_objective(
    df: pl.DataFrame,
    base_config: DictConfig,
    n_windows: int = 5,
    metric_weights: Dict[str, float] = None
) -> Callable:
    """
    Create Optuna objective function for XGBoost hyperparameter optimization.
    
    Args:
        df: Training DataFrame
        base_config: Hydra config to use as base
        n_windows: Number of evaluation windows per trial (for faster optimization)
        metric_weights: Weights for composite metric (default: 0.7 * AUC-PR + 0.3 * P@100)
    
    Returns:
        Objective function for Optuna
    """
    if metric_weights is None:
        metric_weights = {"auc_pr": 0.7, "p@100": 0.3}
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for XGBoost hyperparameter tuning.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Composite metric score (higher is better)
        """
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        }
        
        # Create trial config by overriding base config params
        trial_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        trial_config.model.name = "xgboost_hyperopt"
        trial_config.model.params = OmegaConf.create(params)
        
        # Use larger step size for faster optimization
        step_days_optimization = max(base_config.model.training.step_days * 4, 28)
        trial_config.model.training.step_days = step_days_optimization
        
        # Create nested MLflow run for this trial
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            
            try:
                result = train_accumulating_window(
                    df=df,
                    config=trial_config,
                    max_windows=n_windows,  # Limit windows for faster optimization
                    skip_mlflow_run=True  # Already inside MLflow run context
                )
                
                if not result.get("results"):
                    return float('-inf')  # Invalid trial
                
                # Calculate composite metric
                results_df = pl.DataFrame(result["results"])
                mean_auc_pr = float(results_df["auc_pr"].mean())
                mean_p100 = float(results_df["p@100"].mean())
                
                composite_score = (
                    metric_weights.get("auc_pr", 0.7) * mean_auc_pr +
                    metric_weights.get("p@100", 0.3) * mean_p100
                )
                
                # Log metrics
                mlflow.log_metrics({
                    "mean_auc_pr": mean_auc_pr,
                    "mean_p_at_100": mean_p100,
                    "composite_score": composite_score,
                })
                
                # Report to Optuna (minimize negative score = maximize score)
                return -composite_score
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                return float('inf')  # Invalid trial
    
    return objective


def optimize_xgboost_hyperparameters(
    config: DictConfig,
    n_trials: int = 100,
    n_windows: int = 5,
    timeout_minutes: Optional[int] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for XGBoost models.
    
    Args:
        config: Hydra configuration
        n_trials: Number of Optuna trials
        n_windows: Number of evaluation windows per trial
        timeout_minutes: Optional timeout in minutes
        metric_weights: Weights for composite metric
        study_name: Optional Optuna study name
    
    Returns:
        Dictionary with best parameters and metrics
    """
    from src.models.utils.common import setup_mlflow
    setup_mlflow(config.experiment_name)
    
    # Load and prepare data
    df = load_data()
    df = compute_base_features(df, cutoff_date=datetime.now(), config=None)
    
    # Create objective function
    objective = create_xgboost_objective(
        df=df,
        base_config=config,
        n_windows=n_windows,
        metric_weights=metric_weights
    )
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",  # Minimize negative composite score = maximize score
        study_name=study_name or f"xgboost_hyperopt_{config.experiment_name}",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    with mlflow.start_run(run_name="xgboost_hyperparameter_optimization"):
        mlflow.log_params({
            "n_trials": n_trials,
            "n_windows": n_windows,
            "initial_window_days": config.model.training.initial_window_days,
            "step_days": config.model.training.step_days,
            "feature_categories": str(config.features.categories),
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


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Experiment 3: Hyperparameter Optimization
    
    Run with: python -m src.experiments.exp3_hyperopt experiment_name=xgboost-hyperopt
    """
    # Get optimization parameters from config or use defaults
    n_trials = cfg.get("hyperopt", {}).get("n_trials", 100)
    n_windows = cfg.get("hyperopt", {}).get("n_windows", 5)
    timeout_minutes = cfg.get("hyperopt", {}).get("timeout_minutes", None)
    
    logger.info("="*60)
    logger.info("EXPERIMENT 3: HYPERPARAMETER OPTIMIZATION")
    logger.info("="*60)
    logger.info(f"  Experiment: {cfg.experiment_name}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Windows per trial: {n_windows}")
    logger.info(f"  Feature categories: {cfg.features.categories}")
    
    result = optimize_xgboost_hyperparameters(
        config=cfg,
        n_trials=n_trials,
        n_windows=n_windows,
        timeout_minutes=timeout_minutes
    )
    
    logger.info("Optimization complete!")
    logger.info(f"  Best parameters: {result['best_params']}")
    logger.info(f"  Best score: {result['best_score']:.4f}")
    
    return result


if __name__ == "__main__":
    main()

