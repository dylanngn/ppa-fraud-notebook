"""
Main Training Entry Point.

Simple single-model training with fixed train/test split.
Supports Optuna HPO sweeps via hydra-optuna-sweeper.
"""

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
import os
import mlflow

from src.training.pipeline import SingleTrainingPipeline
from src.features.store import FeatureStore

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(cfg: DictConfig) -> float:
    """
    Main training entry point.
    
    Returns:
        float: The optimization metric (auc_pr) for Optuna HPO.
               Higher is better (direction=maximize in sweeper config).
    """
    print(OmegaConf.to_yaml(cfg))
    original_cwd = get_original_cwd()
    
    if "experiment" in cfg:
        if "tracking_uri" in cfg.experiment:
            tracking_uri = cfg.experiment.tracking_uri
            if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
                db_path = tracking_uri.replace("sqlite:///", "")
                tracking_uri = f"sqlite:///{os.path.join(original_cwd, db_path)}"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        artifact_location = cfg.experiment.get("artifact_location", "mlruns")
        if not os.path.isabs(artifact_location):
            artifact_location = os.path.join(original_cwd, artifact_location)
        
        if "name" in cfg.experiment:
            experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
            if experiment is None:
                mlflow.create_experiment(cfg.experiment.name, artifact_location=artifact_location)
            mlflow.set_experiment(cfg.experiment.name)
    
    data_path = cfg.data.path
    if not os.path.isabs(data_path):
        data_path = os.path.join(original_cwd, data_path)
    
    feature_store = FeatureStore(data_path)
    pipeline = SingleTrainingPipeline(OmegaConf.to_container(cfg, resolve=True), feature_store)
    
    variant = cfg.model.variant
    n_est = cfg.model.xgboost.n_estimators
    max_d = cfg.model.xgboost.max_depth
    lr = cfg.model.xgboost.learning_rate
    
    run_name = f"{variant}_{n_est}est_d{max_d}_lr{lr}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model.variant": cfg.model.variant,
            "model.n_estimators": n_est,
            "model.max_depth": max_d,
            "model.learning_rate": lr,
            "model.min_child_weight": cfg.model.xgboost.min_child_weight,
            "model.subsample": cfg.model.xgboost.subsample,
            "model.colsample_bytree": cfg.model.xgboost.colsample_bytree,
            "training.gap_days": cfg.training.gap_days,
            "data.train_end_date": cfg.data.train_end_date,
            "data.test_end_date": cfg.data.test_end_date,
        })
        
        result = pipeline.run()
        optimization_metric = result.get("auc_pr", 0.0) if result else 0.0
        logger.info(f"Optimization metric (auc_pr): {optimization_metric}")
        
        return optimization_metric


if __name__ == "__main__":
    main()
