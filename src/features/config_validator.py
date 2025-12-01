"""
Configuration validation for feature engineering.

Validates Hydra config before training to catch errors early.
"""
import logging
from omegaconf import DictConfig
from src.models.config.constants import FEATURE_GROUPS
from src.features.registry import FeatureRegistry

logger = logging.getLogger(__name__)


def validate_feature_config(cfg: DictConfig) -> None:
    """
    Validate feature configuration before training.
    
    Checks:
    1. All include_groups exist in FEATURE_GROUPS
    2. All categories are registered in FeatureRegistry
    
    Args:
        cfg: Hydra configuration with features.include_groups and features.categories
        
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    # Validate include_groups
    if hasattr(cfg.features, "include_groups") and cfg.features.include_groups:
        unknown_groups = set(cfg.features.include_groups) - set(FEATURE_GROUPS.keys())
        if unknown_groups:
            errors.append(
                f"Unknown feature groups: {sorted(unknown_groups)}. "
                f"Available: {sorted(FEATURE_GROUPS.keys())}"
            )
    
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
    
    logger.info("Feature configuration validated successfully")

