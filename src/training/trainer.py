"""
Main Training Entry Point.
"""

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
from src.training.accumulated import AccumulatedTrainingPipeline
from src.data.feature_store import FeatureStore
import os

logger = logging.getLogger(__name__)

import mlflow

@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Get original working directory (before Hydra changed it)
    original_cwd = get_original_cwd()
    
    # Setup MLflow Experiment with absolute paths
    # Hydra changes cwd, so relative paths break MLflow artifact storage
    if "experiment" in cfg:
        if "tracking_uri" in cfg.experiment:
            tracking_uri = cfg.experiment.tracking_uri
            # Convert relative sqlite path to absolute
            if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
                db_path = tracking_uri.replace("sqlite:///", "")
                abs_db_path = os.path.join(original_cwd, db_path)
                tracking_uri = f"sqlite:///{abs_db_path}"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Set artifact location to absolute path (prevents Hydra cwd issues)
        artifact_location = cfg.experiment.get("artifact_location", "mlruns")
        if not os.path.isabs(artifact_location):
            artifact_location = os.path.join(original_cwd, artifact_location)
        
        if "name" in cfg.experiment:
            # Create/get experiment with explicit artifact location
            experiment = mlflow.get_experiment_by_name(cfg.experiment.name)
            if experiment is None:
                mlflow.create_experiment(cfg.experiment.name, artifact_location=artifact_location)
            mlflow.set_experiment(cfg.experiment.name)
    
    # Resolve Paths
    data_path = cfg.data.path
    
    # Setup Feature Store
    feature_store = FeatureStore(data_path)
    
    # Initialize Pipeline
    pipeline = AccumulatedTrainingPipeline(
        OmegaConf.to_container(cfg, resolve=True),
        feature_store
    )
    
    # Run with named parent run
    run_name = f"{cfg.model.variant}_{cfg.data.start_date}_to_{cfg.data.end_date}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "variant": cfg.model.variant,
            "data_start": cfg.data.start_date,
            "data_end": cfg.data.end_date,
        })
        pipeline.run()

if __name__ == "__main__":
    main()
