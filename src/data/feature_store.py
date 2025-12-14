"""
Feature Store interface.
Handles loading raw data, coalescing features, and serving model-ready data.
"""

import logging
from typing import Optional, Union, List
from pathlib import Path
import polars as pl
import polars.selectors as cs

from src.data.schema import FEATURE_SCHEMA, ModelVariant

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Central access point for feature data.
    """
    
    def __init__(self, raw_data_path: Union[str, Path]):
        self.raw_data_path = Path(raw_data_path)
        self._df: Optional[pl.LazyFrame] = None
        
    def load_data(self) -> pl.LazyFrame:
        """Load and preprocess raw data."""
        if self._df is None:
            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Data not found at {self.raw_data_path}")
            
            logger.info(f"Loading and preprocessing data from {self.raw_data_path}")
            df = pl.scan_parquet(self.raw_data_path)
            
            # Target derivation
            if FEATURE_SCHEMA.raw_target_source in df.collect_schema().names():
                df = df.with_columns(
                    pl.col(FEATURE_SCHEMA.raw_target_source).is_not_null().alias(FEATURE_SCHEMA.target).cast(pl.Int8)
                )
            else:
                 logger.warning(f"Raw target source '{FEATURE_SCHEMA.raw_target_source}' not found. 'is_fraud' might be missing.")

            # Feature coalescence
            available_cols = df.collect_schema().names()
            
            def get_cols_expr(names: List[str]):
                valid = [pl.col(n).cast(pl.Float64, strict=False) for n in names if n in available_cols]
                return valid if valid else pl.lit(None, dtype=pl.Float64)
            
            prices = get_cols_expr(list(FEATURE_SCHEMA.raw_price_cols))
            price_expr = pl.coalesce(prices) if isinstance(prices, list) else prices
                 
            areas = get_cols_expr(list(FEATURE_SCHEMA.raw_area_cols))
            area_expr = pl.coalesce(areas) if isinstance(areas, list) else areas
            
            def safe_col(name, alias, dtype=pl.Float64):
                if name in available_cols:
                    return pl.col(name).cast(dtype, strict=False).alias(alias)
                return pl.lit(None, dtype=dtype).alias(alias)

            df = df.with_columns([
                price_expr.alias("feature_price"),
                area_expr.alias("feature_area"),
                safe_col("listing.characteristics.numberOfRooms", "feature_rooms"),
                safe_col("listing.characteristics.numberOfBathrooms", "feature_bathrooms"),
                safe_col("listing.characteristics.yearBuilt", "feature_year_built"),
                safe_col("listing.characteristics.numberOfFloors", "feature_floors"),
                (pl.col("auto_approval_criteria.criteria.seonApproved").cast(pl.Boolean, strict=False).fill_null(True).cast(pl.Int8).alias("benchmark_seon_approved")
                 if "auto_approval_criteria.criteria.seonApproved" in available_cols else pl.lit(1).alias("benchmark_seon_approved")),
            ])
            
            # Temporal features
            time_col = "submission_at"
            if time_col in available_cols:
                df = df.with_columns([
                    pl.col(time_col).dt.hour().alias("posting_hour"),
                    pl.col(time_col).dt.weekday().alias("posting_weekday")
                ])
            
            self._df = df
            
        return self._df
    
    def get_features_for_variant(
        self, 
        df: pl.LazyFrame, 
        variant: ModelVariant
    ) -> pl.LazyFrame:
        """Select features for variant."""
        feature_cols = FEATURE_SCHEMA.get_features_for_variant(variant)
        
        cols_to_select = list(FEATURE_SCHEMA.id_columns) + [FEATURE_SCHEMA.target] + feature_cols
        cols_to_select += list(FEATURE_SCHEMA.benchmark_features)
        cols_to_select = sorted(list(set(cols_to_select)))
        
        for key in FEATURE_SCHEMA.graph_identity_columns:
            if key not in cols_to_select:
                cols_to_select.append(key)
        
        schema_keys = df.collect_schema().names()
        final_cols = [c for c in cols_to_select if c in schema_keys]
        
        missing = set(cols_to_select) - set(final_cols)
        if missing:
            known_missing = {c for c in missing if "gnn_emb" in c or "degree" in c or "neighbor" in c}
            real_missing = missing - known_missing
            if real_missing:
                logger.warning(f"Missing columns in dataframe: {real_missing}")
        
        return df.select(final_cols)
