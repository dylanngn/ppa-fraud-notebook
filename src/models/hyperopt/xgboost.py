"""
Hyperparameter optimization for XGBoost models using Optuna and MLflow.
Follows MLflow best practices for hyperparameter tuning.
"""
import optuna
import polars as pl
import xgboost as xgb
import mlflow
import typer
from typing import Dict, Any, Callable, Optional

from src.models.training_window import train_accumulating_window
from src.models.config.experiment_config import ExperimentConfig
from src.models.feature_engineering import load_data, add_base_tabular_features


def create_xgboost_objective(
    df: pl.DataFrame,
    config: ExperimentConfig,
    n_windows: int = 5,
    metric_weights: Dict[str, float] = None
) -> Callable:
    """
    Create Optuna objective function for XGBoost hyperparameter optimization.
    
    Args:
        df: Training DataFrame
        config: Experiment configuration
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
        
        # Create nested MLflow run for this trial
        with mlflow.start_run(nested=True):
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)
            
            # Create config with these hyperparameters
            trial_config = ExperimentConfig(
                experiment_name=config.experiment_name,
                initial_window_days=config.initial_window_days,
                step_days=config.step_days,
                feature_categories=config.feature_categories,
                xgb_params=params
            )
            
            # Train with limited windows for faster optimization
            # Use larger step size to reduce number of windows
            step_days_optimization = max(config.step_days * 4, 28)  # At least 28 days
            
            # Create trial config with optimized step size
            trial_config_optimized = ExperimentConfig(
                experiment_name=config.experiment_name,
                initial_window_days=config.initial_window_days,
                step_days=step_days_optimization,
                feature_categories=config.feature_categories,
                xgb_params=params
            )
            
            try:
                result = train_accumulating_window(
                    df=df,
                    initial_window_days=config.initial_window_days,
                    step_days=step_days_optimization,
                    extra_features=[],
                    embedding_generator=None,
                    model_name="xgboost_hyperopt",
                    config=trial_config_optimized,
                    max_windows=n_windows  # Limit windows for faster optimization
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
    config: ExperimentConfig,
    n_trials: int = 100,
    n_windows: int = 5,
    timeout_minutes: Optional[int] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization for XGBoost models.
    
    Args:
        config: Experiment configuration
        n_trials: Number of Optuna trials
        n_windows: Number of evaluation windows per trial
        timeout_minutes: Optional timeout in minutes
        metric_weights: Weights for composite metric
        study_name: Optional Optuna study name
    
    Returns:
        Dictionary with best parameters and metrics
    """
    # Setup MLflow
    from src.models.utils.common import setup_mlflow
    setup_mlflow(config.experiment_name)
    
    # Load data
    df = load_data()
    df = add_base_tabular_features(df)
    
    # Create objective function
    objective = create_xgboost_objective(
        df=df,
        config=config,
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
            "initial_window_days": config.initial_window_days,
            "step_days": config.step_days,
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
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    initial_window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Step size between evaluation windows in days"),
    n_trials: int = typer.Option(100, help="Number of Optuna trials"),
    n_windows: int = typer.Option(5, help="Number of evaluation windows per trial"),
    timeout_minutes: Optional[int] = typer.Option(None, help="Optional timeout in minutes"),
    feature_categories: Optional[str] = typer.Option(
        None,
        help="Comma-separated feature categories: base,graph,advanced_graph,time_weighted,interaction,text"
    ),
):
    """
    Hyperparameter optimization for XGBoost models using Optuna.
    
    Tunes XGBoost hyperparameters to maximize: 0.7 * AUC-PR + 0.3 * P@100
    """
    from src.models.config.experiment_config import ExperimentConfig, FeatureCategory
    
    # Parse feature categories
    categories = None
    if feature_categories:
        categories = [FeatureCategory(cat.strip()) for cat in feature_categories.split(",")]
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        initial_window_days=initial_window_days,
        step_days=step_days,
        feature_categories=categories
    )
    
    result = optimize_xgboost_hyperparameters(
        config=config,
        n_trials=n_trials,
        n_windows=n_windows,
        timeout_minutes=timeout_minutes
    )
    
    typer.echo(f"Best parameters: {result['best_params']}")
    typer.echo(f"Best score: {result['best_score']:.4f}")
    
    return result


if __name__ == "__main__":
    import typer
    typer.run(main)

