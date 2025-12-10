"""
Configuration validation for feature engineering.

Validates Hydra config before training to catch errors early.
"""
import logging
from omegaconf import DictConfig
from src.features.xgboost.registry import FeatureRegistry

logger = logging.getLogger(__name__)


def validate_feature_config(cfg: DictConfig) -> None:
    """
    Validate feature configuration before training.
    
    Checks that all categories are registered in FeatureRegistry.
    
    Args:
        cfg: Hydra configuration with features.categories
        
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    # Validate categories
    if hasattr(cfg.features, "categories") and cfg.features.categories:
        available_categories = set(FeatureRegistry.list_categories())
        unknown_categories = set(cfg.features.categories) - available_categories
        if unknown_categories:
            errors.append(
                f"Unknown feature categories: {sorted(unknown_categories)}. "
                f"Available: {sorted(available_categories)}"
            )
    
    if errors:
        raise ValueError("Feature configuration validation failed:\n" + "\n".join(errors))
    
    logger.info(f"Feature config validated: categories={list(cfg.features.categories)}")
