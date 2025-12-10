"""
Feature Processor.

SIMPLIFIED (2025-12-09 after Exp 9 validation):
- Uses all tabular columns minus exclusions (auto mode)
- Graph features explicitly computed via "graph" category
- Encodes categorical/string columns for XGBoost compatibility
"""
import polars as pl
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import logging
from omegaconf import DictConfig
from src.features.xgboost.registry import FeatureRegistry
from src.models.config.constants import EXCLUDED_COLUMNS, EXCLUDED_PATTERNS

# Side-effect imports: register feature generators
import src.features.xgboost.definitions.base  # noqa: F401 - registers "base" category
import src.features.xgboost.definitions.graph  # noqa: F401 - registers "graph" category

logger = logging.getLogger(__name__)

# Constants for feature processing
MAX_CARDINALITY = 200  # Max unique values for label encoding (higher = likely an ID, will be dropped)
ENCODING_NULL_VALUE = -1  # Value to use for null/missing categorical values after encoding

# Metadata columns that should never be used as features
META_COLUMNS = {
    "insertion_id",
    "object_reference", 
    "is_fraud",
    "submission_at",
    "fraud_flag",
    "user_id",
    "owner_id"
}


class FeatureProcessor:
    """
    Feature processor with auto-tabular mode.
    
    Uses all tabular columns minus exclusions + explicit graph features.
    Automatically encodes categorical columns for XGBoost compatibility.
    
    Usage:
        processor = FeatureProcessor()
        df, feature_cols = processor.process(raw_df, cutoff_date)
    """
    
    def __init__(self, categories: Optional[List[str]] = None, 
                 predetermined_schema: Optional[Dict[str, str]] = None):
        """
        Initialize feature processor.
        
        Args:
            categories: Feature categories to compute. Defaults to ["base", "graph"].
            predetermined_schema: Optional dict of {column: action} where action is 
                                 'keep', 'encode', or 'drop'. Used to ensure train/test 
                                 consistency.
        """
        self.categories = categories or ["base", "graph"]
        self._encoders: Dict[str, Dict[str, int]] = {}  # col -> {value: encoded_int}
        self.predetermined_schema = predetermined_schema
        
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
    
    def _encode_categorical_columns(self, df: pl.DataFrame, keep_cols: set) -> Tuple[pl.DataFrame, int, int]:
        """
        Encode string/categorical columns to integers for XGBoost.
        
        Strategy:
        - Low cardinality (≤ MAX_CARDINALITY unique values): Label encode
        - High cardinality: Drop (likely IDs or free text)
        - Boolean: Convert to int
        - Datetime: Keep if in keep_cols (for temporal split), otherwise drop
        
        If predetermined_schema is set, follows that schema for consistency.
        
        Args:
            df: DataFrame to encode
            keep_cols: Set of column names to always keep (e.g., submission_at)
        
        Returns:
            Tuple of (encoded DataFrame, encoded_count, dropped_count)
        """
        encoded_count = 0
        dropped_cols = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # If we have a predetermined schema, follow it
            if self.predetermined_schema and col in self.predetermined_schema:
                action = self.predetermined_schema[col]
                if action == 'drop':
                    dropped_cols.append(col)
                    continue
                elif action == 'encode':
                    # Encode using existing encoder or create new one
                    if col not in self._encoders:
                        unique_vals = df[col].drop_nulls().unique().sort().to_list()
                        self._encoders[col] = {v: i for i, v in enumerate(unique_vals)}
                    
                    df = df.with_columns(
                        pl.col(col).replace(self._encoders[col], default=ENCODING_NULL_VALUE).alias(col)
                    )
                    encoded_count += 1
                    continue
                # action == 'keep' falls through to normal logic
            
            # Skip numeric columns (already good)
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64):
                continue
            
            # Boolean -> int
            if dtype == pl.Boolean:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
                encoded_count += 1
                continue
            
            # Datetime -> keep if in keep_cols (needed for temporal split)
            if dtype in (pl.Datetime, pl.Date, pl.Time):
                if col not in keep_cols:
                    dropped_cols.append(col)
                continue
            
            # String/Utf8 -> check cardinality and encode (if no predetermined schema)
            if dtype == pl.Utf8 or dtype == pl.String:
                n_unique = df[col].n_unique()
                
                if n_unique <= MAX_CARDINALITY:
                    # Label encode
                    unique_vals = df[col].drop_nulls().unique().sort().to_list()
                    self._encoders[col] = {v: i for i, v in enumerate(unique_vals)}
                    
                    # Map values to integers (null -> ENCODING_NULL_VALUE)
                    df = df.with_columns(
                        pl.col(col).replace(self._encoders[col], default=ENCODING_NULL_VALUE).alias(col)
                    )
                    encoded_count += 1
                else:
                    # Too many unique values - drop
                    dropped_cols.append(col)
                continue
            
            # Any other type -> drop
            dropped_cols.append(col)
        
        # Drop high-cardinality and unsupported columns
        if dropped_cols:
            df = df.drop(dropped_cols)
        
        return df, encoded_count, len(dropped_cols)
        
    def process(self, df: pl.DataFrame, cutoff_date: datetime, 
                expected_columns: Optional[List[str]] = None) -> Tuple[pl.DataFrame, List[str]]:
        """
        Generate ML-ready features for the given dataframe and cutoff date.
        
        Args:
            df: Input DataFrame with raw listing data
            cutoff_date: Temporal cutoff for feature computation
            expected_columns: Optional list of columns that should be present after processing.
                            If provided, missing columns will be added with nulls.
            
        Returns:
            Tuple of (DataFrame with ML-ready features, list of feature column names)
        """
        logger.info(f"Processing features for cutoff: {cutoff_date}")
        logger.info(f"Categories: {self.categories}")
        logger.info(f"Input columns: {len(df.columns)}")
        
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
        
        df = df.select(cols_to_keep)
        logger.info(f"After exclusions: {len(cols_to_keep)} columns (excluded {excluded_count})")
        
        # If expected_columns provided, ensure all are present (add missing with nulls)
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Adding {len(missing_cols)} missing columns with nulls: {list(missing_cols)[:5]}...")
                for col in missing_cols:
                    df = df.with_columns(pl.lit(None).alias(col))
        
        # Encode categorical columns for XGBoost compatibility
        # Pass always_keep so we don't drop submission_at (needed for temporal split)
        df, encoded_count, dropped_count = self._encode_categorical_columns(df, always_keep)
        logger.info(f"Encoded {encoded_count} categorical columns, dropped {dropped_count} high-cardinality/unsupported")
        
        # Identify feature columns (everything except metadata)
        feature_cols = [c for c in df.columns if c not in META_COLUMNS]
        
        logger.info(f"Final: {len(feature_cols)} ML-ready features")
        
        return df, feature_cols
    
    @staticmethod
    def determine_encoding_schema(df: pl.DataFrame, keep_cols: set) -> Dict[str, str]:
        """
        Determine which columns should be kept, encoded, or dropped.
        
        This should be run on the FULL dataset (before train/test split) to ensure
        train and test have consistent columns.
        
        Args:
            df: DataFrame to analyze
            keep_cols: Set of column names to always keep (e.g., submission_at)
        
        Returns:
            Dict mapping column name to action: 'keep', 'encode', or 'drop'
        """
        schema = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Numeric columns - keep as is
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64):
                schema[col] = 'keep'
                continue
            
            # Boolean - will be encoded to int
            if dtype == pl.Boolean:
                schema[col] = 'encode'
                continue
            
            # Datetime - keep if in keep_cols, drop otherwise
            if dtype in (pl.Datetime, pl.Date, pl.Time):
                schema[col] = 'keep' if col in keep_cols else 'drop'
                continue
            
            # String/Utf8 - check cardinality
            if dtype == pl.Utf8 or dtype == pl.String:
                n_unique = df[col].n_unique()
                schema[col] = 'encode' if n_unique <= MAX_CARDINALITY else 'drop'
                continue
            
            # Any other type - drop
            schema[col] = 'drop'
        
        return schema
    
    def get_feature_summary(self) -> dict:
        """Get a summary of feature configuration."""
        return {
            "categories": self.categories,
            "mode": "auto",
            "encoders": {k: len(v) for k, v in self._encoders.items()},
        }
