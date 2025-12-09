"""
Feature Processor.

SIMPLIFIED (2025-12-09 after Exp 9 validation):
- Uses all tabular columns minus exclusions (auto mode)
- Graph features explicitly computed via "graph" category
"""
import polars as pl
from datetime import datetime
from typing import List, Optional, Set
import logging
from omegaconf import DictConfig
from src.features.registry import FeatureRegistry
from src.models.config.constants import EXCLUDED_COLUMNS, EXCLUDED_PATTERNS

# Side-effect imports: register feature generators
import src.features.definitions.base  # noqa: F401 - registers "base" category
import src.features.definitions.graph  # noqa: F401 - registers "graph" category

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Feature processor with auto-tabular mode.
    
    Uses all tabular columns minus exclusions + explicit graph features.
    
    Usage:
        processor = FeatureProcessor()
        df = processor.process(raw_df, cutoff_date)
    """
    
    def __init__(self, categories: Optional[List[str]] = None):
        """
        Initialize feature processor.
        
        Args:
            categories: Feature categories to compute. Defaults to ["base", "graph"].
        """
        self.categories = categories or ["base", "graph"]
        
    @classmethod
    def from_config(cls, config: DictConfig) -> "FeatureProcessor":
        """
        Create processor from Hydra config.
        
        Args:
            config: Hydra feature configuration with 'categories' list
            
        Returns:
            Configured FeatureProcessor instance
        """
        categories = list(config.get("categories", [])) or None
        return cls(categories=categories)
    
    def _is_excluded(self, col: str) -> bool:
        """Check if a column should be excluded."""
        if col in EXCLUDED_COLUMNS:
            return True
        for pattern in EXCLUDED_PATTERNS:
            if pattern in col:
                return True
        return False
        
    def process(self, df: pl.DataFrame, cutoff_date: datetime) -> pl.DataFrame:
        """
        Generate features for the given dataframe and cutoff date.
        
        Args:
            df: Input DataFrame with raw listing data
            cutoff_date: Temporal cutoff for feature computation
            
        Returns:
            DataFrame with computed features
        """
        logger.info(f"Processing features for cutoff: {cutoff_date}")
        logger.info(f"Categories: {self.categories}")
        
        # Run feature generators
        for category in self.categories:
            if category in FeatureRegistry.list_categories():
                logger.info(f"Computing {category} features...")
                generator = FeatureRegistry.get(category)
                df = generator(df, cutoff_date, None)
            else:
                logger.warning(f"Category '{category}' not found. Available: {FeatureRegistry.list_categories()}")
        
        # Filter out excluded columns
        always_keep = {"insertion_id", "object_reference", "is_fraud", "submission_at", "fraud_flag"}
        
        cols_to_keep = []
        excluded_count = 0
        for col in df.columns:
            if col in always_keep:
                cols_to_keep.append(col)
            elif self._is_excluded(col):
                excluded_count += 1
            else:
                cols_to_keep.append(col)
        
        logger.info(f"Keeping {len(cols_to_keep)} columns, excluded {excluded_count}")
        return df.select(cols_to_keep)
    
    def get_feature_summary(self) -> dict:
        """Get a summary of feature configuration."""
        return {
            "categories": self.categories,
            "mode": "auto",
        }
