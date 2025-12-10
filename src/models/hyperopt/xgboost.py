"""
XGBoost Hyperparameter Optimization using Optuna

Optimizes XGBoost hyperparameters using temporal cross-validation.
Uses accumulating window strategy to maintain temporal order.

Usage:
    # Run optimization
    PYTHONPATH=. python src/models/hyperopt/xgboost.py

    # With custom settings
    PYTHONPATH=. python src/models/hyperopt/xgboost.py \
        n_trials=100 \
        initial_window_days=180 \
        max_windows=5

Results are logged to MLflow and best parameters are saved.
"""

import logging
import hydra
from omegaconf import DictConfig
import optuna
import mlflow
import polars as pl
from typing import Dict, Any

from src.data.training_loader import load_data
from src.features.xgboost.processor import FeatureProcessor
from src.utils.temporal_split import AccumulatingWindowSplitter
from src.models.xgb_trainer.trainer import train_single_window
from src.models.utils.common import setup_mlflow

logger = logging.getLogger(__name__)


def create_xgboost_objective(
    df: pl.DataFrame,
    feature_processor: FeatureProcessor,
    target_col: str,
    initial_window_days: int = 180,
    step_days: int = 7,
    max_windows: int = 3,
):
    """
    Create Optuna objective function for XGBoost hyperparameter optimization.
    
    Args:
        df: Input DataFrame with temporal column
        feature_processor: Configured feature processor
        target_col: Target column name
        initial_window_days: Days in initial training window
        step_days: Days between successive windows
        max_windows: Maximum number of windows to evaluate
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.
        
        Trains XGBoost with suggested hyperparameters across multiple
        temporal windows and returns average AUC-PR.
        """
        
        # Suggest hyperparameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "n_jobs": -1,
            "random_state": 42,
        }
        
        # Create temporal splitter
        splitter = AccumulatingWindowSplitter(
            df=df,
            initial_window_days=initial_window_days,
            step_days=step_days,
            test_days=14,
            time_column="submission_at",
            max_windows=max_windows
        )
        
        # Evaluate across windows
        auc_pr_scores = []
        
        for window_idx, train_df, test_df, window_info in splitter.split():
            # Process features
            train_processed = feature_processor.process(train_df)
            test_processed = feature_processor.process(test_df)
            
            # Get feature columns
            feature_cols = [col for col in train_processed.columns 
                          if col not in ['insertion_id', target_col, 'submission_at']]
            
            # Train model
            result = train_single_window(
                train_df=train_processed.to_pandas(),
                test_df=test_processed.to_pandas(),
                feature_cols=feature_cols,
                target_col=target_col,
                xgb_params=params,
                window_idx=window_idx,
                log_model=False,  # Don't log models during hyperopt
                nested=False  # No nested runs during hyperopt
            )
            
            auc_pr_scores.append(result['auc_pr'])
            
            # Report intermediate value for pruning
            trial.report(result['auc_pr'], window_idx)
            
            # Prune if not promising
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return average AUC-PR across windows
        avg_auc_pr = sum(auc_pr_scores) / len(auc_pr_scores)
        
        logger.info(f"Trial {trial.number}: avg_auc_pr={avg_auc_pr:.4f}")
        
        return avg_auc_pr
    
    return objective


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def optimize_xgboost_hyperparameters(cfg: DictConfig):
    """
    Run XGBoost hyperparameter optimization with Optuna.
    
    Configuration can be overridden via command line:
        python src/models/hyperopt/xgboost.py n_trials=100
    """
    # Get optimization settings
    n_trials = cfg.get("n_trials", 50)
    initial_window_days = cfg.model.training.get("initial_window_days", 180)
    step_days = cfg.model.training.get("step_days", 7)
    max_windows = cfg.get("max_windows", 3)  # Fewer windows for faster optimization
    
    logger.info("=" * 70)
    logger.info("XGBoost Hyperparameter Optimization")
    logger.info("=" * 70)
    logger.info(f"N Trials: {n_trials}")
    logger.info(f"Initial Window: {initial_window_days} days")
    logger.info(f"Step: {step_days} days")
    logger.info(f"Max Windows: {max_windows}")
    
    # Setup MLflow
    setup_mlflow(cfg.experiment_name)
    
    # Load data
    logger.info("Loading data...")
    df = load_data()
    logger.info(f"Loaded {len(df):,} samples")
    
    # Setup feature processor
    feature_processor = FeatureProcessor.from_config(cfg.features)
    
    # Create objective function
    objective = create_xgboost_objective(
        df=df,
        feature_processor=feature_processor,
        target_col="is_fraud",
        initial_window_days=initial_window_days,
        step_days=step_days,
        max_windows=max_windows
    )
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",  # Maximize AUC-PR
        study_name="xgboost_hyperopt",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )
    
    # Run optimization
    with mlflow.start_run(run_name="xgboost_hyperopt"):
        logger.info("Starting optimization...")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Log best parameters
        logger.info("=" * 70)
        logger.info("Optimization Complete")
        logger.info("=" * 70)
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best avg_auc_pr: {study.best_value:.4f}")
        logger.info("Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Log to MLflow
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_avg_auc_pr", study.best_value)
        
        # Log optimization history
        import plotly.graph_objects as go
        
        # Create optimization history plot
        trials_df = study.trials_dataframe()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trials_df['number'],
            y=trials_df['value'],
            mode='markers',
            name='Trial AUC-PR'
        ))
        fig.add_trace(go.Scatter(
            x=trials_df['number'],
            y=trials_df['value'].cummax(),
            mode='lines',
            name='Best AUC-PR'
        ))
        fig.update_layout(
            title='Hyperparameter Optimization Progress',
            xaxis_title='Trial',
            yaxis_title='Avg AUC-PR'
        )
        mlflow.log_figure(fig, "optimization_history.html")
        
        # Save best params to YAML format
        yaml_str = "# XGBoost Best Parameters (Optuna)\n"
        yaml_str += "# Generated by hyperparameter optimization\n\n"
        yaml_str += "params:\n"
        for key, value in study.best_params.items():
            yaml_str += f"  {key}: {value}\n"
        
        mlflow.log_text(yaml_str, "best_params.yaml")
        
        logger.info("=" * 70)
        logger.info("Results logged to MLflow")
        logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
        logger.info("=" * 70)
    
    return study


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    optimize_xgboost_hyperparameters()
