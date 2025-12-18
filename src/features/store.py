"""
Feature Store for Fraud Detection Pipeline.
"""

import logging
from typing import Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl
import numpy as np

from src.features.schema import (
    FEATURE_SCHEMA,
    ModelVariant,
    LabelPropagation,
)

logger = logging.getLogger(__name__)


class FeatureStore:
    """Central access point for feature data."""
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self._df: Optional[pl.LazyFrame] = None
        self._schema: Optional[list] = None
    
    def load_data(self) -> pl.LazyFrame:
        """Load and preprocess raw data."""
        if self._df is not None:
            return self._df
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.raw_data_path}")
        
        logger.info(f"Loading data from {self.raw_data_path}")
        df = pl.scan_parquet(self.raw_data_path)
        self._schema = df.collect_schema().names()
        
        # Parse time column
        time_col = FEATURE_SCHEMA.time_column
        if time_col in self._schema:
            df = df.with_columns(
                pl.col(time_col)
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f %z", strict=False)
                .alias(time_col)
            )
        
        # Coalesce price features
        df = self._coalesce_price_features(df)
        
        # Create temporal features
        df = self._create_temporal_features(df, time_col)
        
        # Cast boolean SEON features
        df = self._cast_seon_booleans(df)
        
        self._df = df
        return self._df
    
    def _coalesce_price_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Coalesce price/area columns into unified features."""
        available = self._schema or []
        
        price_cols = [
            "LISTING_PRICES_BUY_PRICE",
            "LISTING_PRICES_RENT_GROSS",
            "LISTING_PRICES_RENT_NET",
        ]
        
        area_cols = [
            "LISTING_CHARACTERISTICS_LIVINGSPACE",
            "LISTING_CHARACTERISTICS_LOTSIZE",
        ]
        
        def safe_coalesce(cols: List[str], alias: str):
            valid = [c for c in cols if c in available]
            if valid:
                return pl.coalesce([pl.col(c).cast(pl.Float64, strict=False) for c in valid]).alias(alias)
            return pl.lit(None, dtype=pl.Float64).alias(alias)
        
        df = df.with_columns([
            safe_coalesce(price_cols, "feature_price"),
            safe_coalesce(area_cols, "feature_area"),
        ])
        
        return df
    
    def _create_temporal_features(self, df: pl.LazyFrame, time_col: str) -> pl.LazyFrame:
        """Create temporal features from timestamp."""
        if time_col not in (self._schema or []):
            return df
        
        df = df.with_columns([
            pl.col(time_col).dt.hour().alias("event_hour"),
            pl.col(time_col).dt.weekday().alias("event_weekday"),
            (pl.col(time_col).dt.weekday() >= 5).cast(pl.Int8).alias("event_is_weekend"),
            ((pl.col(time_col).dt.hour() >= 9) & (pl.col(time_col).dt.hour() < 17))
            .cast(pl.Int8).alias("event_is_business_hours"),
        ])
        
        return df
    
    def _cast_seon_booleans(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Cast SEON boolean columns to integers."""
        available = self._schema or []
        
        # Use all boolean features from schema
        bool_cols = list(FEATURE_SCHEMA.all_seon_boolean)
        
        cast_exprs = [
                    pl.col(col).cast(pl.Int8, strict=False).alias(col)
            for col in bool_cols if col in available
        ]
        
        if cast_exprs:
            df = df.with_columns(cast_exprs)
        
        return df
    
    def get_features_for_variant(
        self,
        df: pl.LazyFrame,
        variant: ModelVariant,
    ) -> pl.LazyFrame:
        """Select features for a specific model variant."""
        feature_cols = FEATURE_SCHEMA.get_features_for_variant(variant)
        
        cols_to_select = (
            list(FEATURE_SCHEMA.id_columns) +
            [FEATURE_SCHEMA.time_column] +
            [FEATURE_SCHEMA.target] +
            feature_cols
        )
        cols_to_select.append(FEATURE_SCHEMA.benchmark_column)
        
        if variant == ModelVariant.GNN_XGBOOST:
            cols_to_select.extend(FEATURE_SCHEMA.get_graph_identity_columns())
        
        cols_to_select = list(dict.fromkeys(cols_to_select))
        
        schema_keys = df.collect_schema().names()
        final_cols = [c for c in cols_to_select if c in schema_keys]
        
        missing = set(cols_to_select) - set(final_cols)
        computed_later = {"gnn_emb", "velocity", "insertions_last", "neighbor", "degree", "pagerank"}
        real_missing = {c for c in missing if not any(p in c for p in computed_later)}
        
        if real_missing:
            logger.warning(f"Missing columns: {real_missing}")
        
        return df.select(final_cols)
    
    def prepare_train_test_data(
        self,
        train_end: datetime,
        test_end: datetime,
        gap_days: int = 7,
        label_propagation: LabelPropagation = LabelPropagation.INSERTION_LEVEL,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Prepare train/test splits with temporal ordering."""
        df = self.load_data().collect()
        time_col = FEATURE_SCHEMA.time_column
        target = FEATURE_SCHEMA.target
        insertion_col = FEATURE_SCHEMA.temporal_config.insertion_id_column
        
        # Apply label propagation
        if label_propagation == LabelPropagation.INSERTION_LEVEL and insertion_col in df.columns:
            fraud_insertions = (
                df.filter(pl.col(target) == 1)
                .select(insertion_col)
                .unique()
            )
            
            df = df.with_columns(
                pl.when(pl.col(insertion_col).is_in(fraud_insertions[insertion_col]))
                .then(pl.lit(1))
                .otherwise(pl.col(target))
                .cast(pl.Int8)
                .alias(target)
            )
            
            logger.info(f"Label propagation: {fraud_insertions.height} fraud insertions")
        
        # Split data
        test_start = train_end + timedelta(days=gap_days)
        train_df = df.filter(pl.col(time_col) <= train_end)
        test_df = df.filter((pl.col(time_col) > test_start) & (pl.col(time_col) <= test_end))
        
        logger.info(f"Train/test split: train={train_df.height}, test={test_df.height}")
        
        return train_df, test_df
    
    def get_node_features_tensor(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Get node features as numpy array for GNN."""
        if feature_cols is None:
            feature_cols = FEATURE_SCHEMA.get_gnn_input_features()
        
        available = [c for c in feature_cols if c in df.columns]
        
        if not available:
            logger.warning("No GNN input features available")
            return np.zeros((df.height, 1)), ["dummy"]
        
        X = df.select(available).to_numpy()
        
        # Impute NaN with column medians
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = np.isnan(col)
            if mask.any():
                median = np.nanmedian(col)
                X[mask, j] = median if not np.isnan(median) else 0.0
        
        return X.astype(np.float32), available
