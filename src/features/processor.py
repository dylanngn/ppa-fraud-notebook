"""
Feature Processor.
Generates feature matrix based on configuration.

Supports Hydra configuration with:
- categories: List of feature categories to compute (base, graph, etc.)
- include_groups: List of feature groups to INCLUDE
- Dynamic candidates from conf/features/candidates.yaml (Feature Discovery Pipeline)
"""
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
import logging
from omegaconf import DictConfig, OmegaConf
from src.features.registry import FeatureRegistry
from src.models.config.constants import FEATURE_GROUPS

# Side-effect imports: these register feature generators via @FeatureRegistry.register()
# The modules use decorators that run at import time to populate the registry.
# Without these imports, FeatureRegistry.list_categories() would return empty.
import src.features.definitions.base  # noqa: F401 - registers "base" category
import src.features.definitions.graph  # noqa: F401 - registers "graph", "advanced_graph", "time_weighted"
import src.features.definitions.text  # noqa: F401 - registers "text" category

logger = logging.getLogger(__name__)

# Path to candidates config (for Feature Discovery Pipeline integration)
CANDIDATES_CONFIG_PATH = Path(__file__).parent.parent.parent / "conf" / "features" / "candidates.yaml"


class FeatureProcessor:
    """
    Feature processor that supports Hydra configuration profiles.
    
    Supports three sources of features:
    1. Static feature groups from constants.py (via include_groups)
    2. Dynamic candidates from conf/features/candidates.yaml (Feature Discovery Pipeline)
    3. Categories for feature generators (base, graph, etc.)
    
    Usage:
        # With Hydra config (preferred - using include_groups)
        processor = FeatureProcessor.from_config(cfg.features)
        
        # With explicit parameters (include_groups preferred)
        processor = FeatureProcessor(
            categories=["base", "graph"],
            include_groups=["core_numerical", "boolean_all", "graph_all"]
        )
        
        # With dynamic candidates from Feature Discovery Pipeline
        processor = FeatureProcessor(
            categories=["base", "graph"],
            include_groups=["core_numerical", "boolean_all"],
            use_candidates=True  # Include approved candidates
        )
    """
    
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        include_groups: Optional[List[str]] = None,
        use_candidates: bool = False,
    ):
        """
        Initialize feature processor.
        
        Args:
            categories: Feature categories to compute. Defaults to all available.
            include_groups: Feature groups to include (from constants.FEATURE_GROUPS).
                           This is the PREFERRED way to select features.
            use_candidates: Whether to include approved candidates from
                           conf/features/candidates.yaml (Feature Discovery Pipeline).
        """
        self.categories = categories or FeatureRegistry.list_categories()
        self.include_groups = include_groups
        self.use_candidates = use_candidates
        self.candidate_features: List[Dict[str, Any]] = []
        
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
        
        # Load approved candidates if enabled
        if use_candidates:
            self._load_approved_candidates()
    
    def _load_approved_candidates(self) -> None:
        """Load approved feature candidates from candidates.yaml."""
        if not CANDIDATES_CONFIG_PATH.exists():
            logger.debug("No candidates.yaml found, skipping dynamic candidates")
            return
        
        try:
            config = OmegaConf.load(CANDIDATES_CONFIG_PATH)
            candidates = config.get("candidates", [])
            
            # Filter to approved or integrated candidates
            approved = [
                dict(c) for c in candidates 
                if c.get("status") in ("approved", "integrated")
            ]
            
            if approved:
                self.candidate_features = approved
                candidate_names = [c.get("feature_name") for c in approved]
                
                # Add to include_features set
                if self.include_features is None:
                    self.include_features = set()
                self.include_features.update(candidate_names)
                
                logger.info(f"Loaded {len(approved)} approved candidates from Feature Discovery Pipeline")
                for c in approved:
                    logger.debug(f"  - {c.get('feature_name')} (corr={c.get('correlation', 0):.3f})")
            
        except Exception as e:
            logger.warning(f"Failed to load candidates.yaml: {e}")
        
    @classmethod
    def from_config(cls, config: DictConfig) -> "FeatureProcessor":
        """
        Create processor from Hydra config.
        
        Args:
            config: Hydra feature configuration with:
                   - 'categories': Feature generators to run
                   - 'include_groups': Feature groups to include
                   - 'use_candidates': Whether to include approved candidates
                                      from Feature Discovery Pipeline
            
        Returns:
            Configured FeatureProcessor instance
        """
        categories = list(config.get("categories", [])) or None
        include_groups = list(config.get("include_groups", [])) or None
        use_candidates = config.get("use_candidates", False)

        return cls(
            categories=categories, 
            include_groups=include_groups,
            use_candidates=use_candidates
        )
        
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
            "use_candidates": self.use_candidates,
            "candidate_features_count": len(self.candidate_features),
            "candidate_features": [c.get("feature_name") for c in self.candidate_features],
        }
    
    def get_approved_candidates(self) -> List[Dict[str, Any]]:
        """
        Get list of approved candidates currently loaded.
        
        Returns:
            List of approved candidate feature configs from candidates.yaml
        """
        return self.candidate_features
