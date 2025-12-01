"""
Feature Processor.
Generates feature matrix based on configuration.

Supports Hydra configuration with:
- categories: List of feature categories to compute (base, graph, etc.)
- include_groups: List of feature groups to INCLUDE
"""
import polars as pl
from datetime import datetime
from typing import List, Optional, Set
import logging
from omegaconf import DictConfig
from src.features.registry import FeatureRegistry
from src.models.config.constants import FEATURE_GROUPS

# Side-effect imports: these register feature generators via @FeatureRegistry.register()
# The modules use decorators that run at import time to populate the registry.
# Without these imports, FeatureRegistry.list_categories() would return empty.
import src.features.definitions.base  # noqa: F401 - registers "base" category
import src.features.definitions.graph  # noqa: F401 - registers "graph", "advanced_graph", "time_weighted"
import src.features.definitions.text  # noqa: F401 - registers "text" category

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Feature processor that supports Hydra configuration profiles.
    
    Usage:
        # With Hydra config (preferred - using include_groups)
        processor = FeatureProcessor.from_config(cfg.features)
        
        # With explicit parameters (include_groups preferred)
        processor = FeatureProcessor(
            categories=["base", "graph"],
            include_groups=["core_numerical", "boolean_all", "graph_all"]
        )
    """
    
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        include_groups: Optional[List[str]] = None,
    ):
        """
        Initialize feature processor.
        
        Args:
            categories: Feature categories to compute. Defaults to all available.
            include_groups: Feature groups to include (from constants.FEATURE_GROUPS).
                           This is the PREFERRED way to select features.
        """
        self.categories = categories or FeatureRegistry.list_categories()
        self.include_groups = include_groups
        
        # Build the set of features to include based on groups
        self.include_features: Optional[Set[str]] = None
        if include_groups:
            self.include_features = set()
            for group_name in include_groups:
                if group_name in FEATURE_GROUPS:
                    self.include_features.update(FEATURE_GROUPS[group_name])
                    logger.debug(f"Including group '{group_name}': {len(FEATURE_GROUPS[group_name])} features")
                else:
                    logger.warning(f"Feature group '{group_name}' not found in FEATURE_GROUPS. "
                                   f"Available groups: {list(FEATURE_GROUPS.keys())}")
            logger.info(f"Including {len(self.include_features)} features from {len(include_groups)} groups")
        
    @classmethod
    def from_config(cls, config: DictConfig) -> "FeatureProcessor":
        """
        Create processor from Hydra config.
        
        Args:
            config: Hydra feature configuration with:
                   - 'categories': Feature generators to run
                   - 'include_groups': Feature groups to include
            
        Returns:
            Configured FeatureProcessor instance
        """
        categories = list(config.get("categories", [])) or None
        include_groups = list(config.get("include_groups", [])) or None

        return cls(categories=categories, include_groups=include_groups)
        
    def process(self, df: pl.DataFrame, cutoff_date: datetime) -> pl.DataFrame:
        """
        Generate features for the given dataframe and cutoff date.
        
        Args:
            df: Input DataFrame with raw listing data
            cutoff_date: Temporal cutoff for feature computation
            
        Returns:
            DataFrame with computed features (filtered by include_groups)
        """
        logger.info(f"Generating features for cutoff date: {cutoff_date}")
        logger.info(f"Categories: {self.categories}")
        
        for category in self.categories:
            if category in FeatureRegistry.list_categories():
                logger.info(f"Computing {category} features...")
                generator = FeatureRegistry.get(category)
                df = generator(df, cutoff_date, None)
            else:
                logger.warning(f"Feature category '{category}' not found in registry. Skipping.")
        
        # Apply feature selection
        if self.include_features is not None:
            # INCLUDE mode (preferred): Only keep features in include_groups
            # Always keep ID columns and target columns
            always_keep = {"insertion_id", "object_reference", "is_fraud", "submission_at", "fraud_flag"}
            features_to_keep = self.include_features | always_keep
            
            available_features = set(df.columns)
            selected_features = available_features & features_to_keep
            
            # Warn about features in groups that don't exist in data
            missing_features = self.include_features - available_features
            if missing_features:
                logger.debug(f"Features in include_groups not found in data: {sorted(missing_features)}")
            
            # Select only the features we want
            cols_to_keep = [col for col in df.columns if col in selected_features]
            logger.info(f"Selecting {len(cols_to_keep)} features based on include_groups")
            df = df.select(cols_to_keep)
            
        return df
    
    def get_feature_summary(self) -> dict:
        """
        Get a summary of feature configuration for logging/debugging.
        
        Returns:
            Dictionary with feature configuration summary
        """
        return {
            "categories": self.categories,
            "include_groups": self.include_groups,
            "include_features_count": len(self.include_features) if self.include_features else None,
            "mode": "include_groups" if self.include_features else "all",
        }
