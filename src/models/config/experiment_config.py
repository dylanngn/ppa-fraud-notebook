"""
Experiment configuration for model training.

Enhanced with signature validation for fair model comparison.
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class FeatureCategory(str, Enum):
    """Feature categories that can be enabled/disabled."""
    BASE = "base"
    GRAPH = "graph"
    ADVANCED_GRAPH = "advanced_graph"
    TIME_WEIGHTED = "time_weighted"
    INTERACTION = "interaction"
    TEXT = "text"


@dataclass
class ExperimentConfig:
    """
    Configuration for model training experiments.
    
    Attributes:
        experiment_name: MLflow experiment name
        initial_window_days: Initial training window size in days
        step_days: Step size between evaluation windows in days
        feature_categories: List of feature categories to include
        xgb_params: Optional XGBoost hyperparameters (if None, uses defaults)
    """
    experiment_name: str = "ppa-fraud-detection"
    initial_window_days: int = 180
    step_days: int = 7
    feature_categories: Optional[List[FeatureCategory]] = None
    xgb_params: Optional[dict] = None
    
    def __post_init__(self):
        """Set default feature categories if not provided."""
        if self.feature_categories is None:
            self.feature_categories = list(FeatureCategory)
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns based on enabled categories.
        
        Returns:
            List of feature column names
        """
        from src.models.config.constants import (
            BASE_FEATURES,
            GRAPH_FEATURE_COLUMNS,
            ADVANCED_GRAPH_FEATURE_COLUMNS,
            TIME_WEIGHTED_FEATURE_COLUMNS,
            INTERACTION_FEATURE_COLUMNS,
            TEXT_FEATURE_COLUMNS,
        )
        
        features = []
        
        if FeatureCategory.BASE in self.feature_categories:
            features.extend(BASE_FEATURES)
        if FeatureCategory.GRAPH in self.feature_categories:
            features.extend(GRAPH_FEATURE_COLUMNS)
        if FeatureCategory.ADVANCED_GRAPH in self.feature_categories:
            features.extend(ADVANCED_GRAPH_FEATURE_COLUMNS)
        if FeatureCategory.TIME_WEIGHTED in self.feature_categories:
            features.extend(TIME_WEIGHTED_FEATURE_COLUMNS)
        if FeatureCategory.INTERACTION in self.feature_categories:
            features.extend(INTERACTION_FEATURE_COLUMNS)
        if FeatureCategory.TEXT in self.feature_categories:
            features.extend(TEXT_FEATURE_COLUMNS)
        
        return features
    
    def validate_signature_compatibility(
        self, 
        model_uris: dict,
        sample_data=None
    ) -> dict:
        """
        Validate that models trained with this config have compatible signatures.
        
        This ensures fair comparison across baseline and hybrid models.
        
        Args:
            model_uris: Dictionary mapping model names to MLflow URIs
            sample_data: Optional sample data for validation
        
        Returns:
            Validation results dictionary
        """
        from src.models.utils.signature_validation import compare_model_signatures
        
        return compare_model_signatures(model_uris, self)
    
    def get_signature_info(self) -> dict:
        """
        Get signature metadata for this experiment configuration.
        
        Returns:
            Dictionary with signature information:
            - expected_features: List of feature names
            - feature_categories: List of enabled categories
            - feature_count: Total number of features
        """
        features = self.get_feature_columns()
        return {
            "expected_features": features,
            "feature_categories": [cat.value for cat in self.feature_categories],
            "feature_count": len(features),
            "experiment_name": self.experiment_name
        }

