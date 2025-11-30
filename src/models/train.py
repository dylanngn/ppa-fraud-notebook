"""
Main training entry point using Hydra.
"""
import hydra
from omegaconf import DictConfig
import polars as pl
import logging

from src.models.xgboost.trainer import train_accumulating_window
from src.features.definitions.base import compute_base_features
from datetime import datetime

logger = logging.getLogger(__name__)

def load_data() -> pl.DataFrame:
    """Load and join listing and user data."""
    df_listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    return df_listings.join(df_users, on="user_id", how="left")

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
    
    # Apply base features (these are independent of graph)
    df = compute_base_features(df, cutoff_date=datetime.now(), config=None)
    
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
