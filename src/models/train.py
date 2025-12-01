"""
Main training entry point using Hydra.
"""
import hydra
from omegaconf import DictConfig
import logging

from src.models.xgboost.trainer import train_accumulating_window
from src.data.loader import load_data

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.
    """
    logger.info("Starting training...")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Features: {cfg.features.categories}")
    
    # Load data
    df = load_data()
    
    # Train
    result = train_accumulating_window(
        df=df,
        config=cfg,
        max_windows=cfg.model.training.max_windows
    )
    
    logger.info(f"Training complete. Mean AUC-PR: {result['mean_auc_pr']:.4f}")
    logger.info(f"Best AUC-PR: {result['best_auc_pr']:.4f}")
    
    return result

if __name__ == "__main__":
    main()
