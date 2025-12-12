"""
Main Training Entry Point.
"""

import hydra
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
    
    # Setup MLflow Experiment
    if "experiment" in cfg:
        if "tracking_uri" in cfg.experiment:
            mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
        if "name" in cfg.experiment:
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
    
    # Run
    pipeline.run()

if __name__ == "__main__":
    main()
