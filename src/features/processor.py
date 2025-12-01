"""
Feature Processor.
Generates feature matrix based on configuration.

Supports Hydra configuration with:
- categories: List of feature categories to include (base, graph, etc.)
- exclude: List of specific feature columns to exclude from final output
"""
import polars as pl
from datetime import datetime
from typing import List, Optional, Set
import logging
from omegaconf import DictConfig
from src.features.registry import FeatureRegistry
# Import definitions to register them
import src.features.definitions.base
import src.features.definitions.graph
import src.features.definitions.text

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Feature processor that supports Hydra configuration profiles.
    
    Usage:
        # With Hydra config
        processor = FeatureProcessor.from_config(cfg.features)
        
        # With explicit parameters
        processor = FeatureProcessor(
            categories=["base", "graph"],
            exclude=["is_new", "has_elevator"]
        )
    """
    
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ):
        """
        Initialize feature processor.
        
        Args:
            categories: Feature categories to compute. Defaults to all available.
            exclude: Feature columns to exclude from final output.
        """
        self.categories = categories or FeatureRegistry.list_categories()
        self.exclude: Set[str] = set(exclude or [])
        
        if self.exclude:
            logger.info(f"Excluding {len(self.exclude)} features: {sorted(self.exclude)}")
    
    @classmethod
    def from_config(cls, config: DictConfig) -> "FeatureProcessor":
        """
        Create processor from Hydra config.
        
        Args:
            config: Hydra feature configuration with 'categories' and 'exclude' keys
            
        Returns:
            Configured FeatureProcessor instance
        """
        categories = list(config.get("categories", [])) or None
        exclude = list(config.get("exclude", [])) or None
        return cls(categories=categories, exclude=exclude)
        
    def process(self, df: pl.DataFrame, cutoff_date: datetime) -> pl.DataFrame:
        """
        Generate features for the given dataframe and cutoff date.
        
        Args:
            df: Input DataFrame with raw listing data
            cutoff_date: Temporal cutoff for feature computation
            
        Returns:
            DataFrame with computed features (excluding configured exclusions)
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
        
        # Apply exclusions
        if self.exclude:
            cols_to_drop = [col for col in self.exclude if col in df.columns]
            if cols_to_drop:
                logger.info(f"Dropping {len(cols_to_drop)} excluded features: {cols_to_drop}")
                df = df.drop(cols_to_drop)
                
        return df
