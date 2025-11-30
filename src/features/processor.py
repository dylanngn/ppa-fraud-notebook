"""
Feature Processor.
Generates feature matrix based on configuration.
"""
import polars as pl
from datetime import datetime
from typing import List, Optional
import logging
from src.features.registry import FeatureRegistry
# Import definitions to register them
import src.features.definitions.base
import src.features.definitions.graph
import src.features.definitions.text

logger = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self, config: Optional[List[str]] = None):
        self.config = config or FeatureRegistry.list_categories()
        
    def process(self, df: pl.DataFrame, cutoff_date: datetime) -> pl.DataFrame:
        """
        Generate features for the given dataframe and cutoff date.
        """
        logger.info(f"Generating features for cutoff date: {cutoff_date}")
        
        for category in self.config:
            if category in FeatureRegistry.list_categories():
                logger.info(f"Computing {category} features...")
                generator = FeatureRegistry.get(category)
                df = generator(df, cutoff_date, None)
            else:
                logger.warning(f"Feature category '{category}' not found in registry. Skipping.")
                
        return df
