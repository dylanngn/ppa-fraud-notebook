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
        """
        Load AND PREPROCESS raw data.
        1. Load Parquet
        2. Coalesce Price/Area
        3. Derive Target
        4. Derive Temporal Features
        """
        if self._df is None:
            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"Data not found at {self.raw_data_path}")
            
            logger.info(f"Loading and preprocessing data from {self.raw_data_path}")
            
            # Start Lazy
            df = pl.scan_parquet(self.raw_data_path)
            
            # --- PREPROCESSING ---
            
            # 1. Target Derivation (fraud_flag timestamp -> boolean)
            # If fraud_flag is present.
            if FEATURE_SCHEMA.raw_target_source in df.collect_schema().names():
                df = df.with_columns(
                    pl.col(FEATURE_SCHEMA.raw_target_source).is_not_null().alias(FEATURE_SCHEMA.target).cast(pl.Int8)
                )
            else:
                 logger.warning(f"Raw target source '{FEATURE_SCHEMA.raw_target_source}' not found. 'is_fraud' might be missing.")

            # 2. Feature Coalescence
            # Helper: Get column if exists, else lit(None) (preserve NaN)
            available_cols = df.collect_schema().names()
            
            def get_cols_expr(names: List[str]):
                valid = [pl.col(n).cast(pl.Float64, strict=False) for n in names if n in available_cols]
                if not valid:
                    return pl.lit(None, dtype=pl.Float64)
                return valid
            
            # Price
            prices = get_cols_expr(list(FEATURE_SCHEMA.raw_price_cols))
            if isinstance(prices, list):
                 price_expr = pl.coalesce(prices) # No fill_null(0.0)
            else:
                 price_expr = prices
                 
            # Area
            areas = get_cols_expr(list(FEATURE_SCHEMA.raw_area_cols))
            if isinstance(areas, list):
                 area_expr = pl.coalesce(areas) # No fill_null(0.0)
            else:
                 area_expr = areas
            
            # Rooms/Baths/Year/Floors/Seon
            
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
                
                # Benchmark: Seon Approved
                # If True -> Legit (0). If False -> Fraud (1)? 
                # Needs strict casting. It might be boolean or string.
                (pl.col("auto_approval_criteria.criteria.seonApproved").cast(pl.Boolean, strict=False).fill_null(True).cast(pl.Int8).alias("benchmark_seon_approved")
                 if "auto_approval_criteria.criteria.seonApproved" in available_cols else pl.lit(1).alias("benchmark_seon_approved")), # Default to Approved (1)
            ])
            
            # 3. Temporal Features
            # submission_at -> posting_hour
            time_col = "submission_at"
            if time_col in available_cols:
                # Ensure datetime
                # Note: scan_parquet implies types, usually inferred correctly.
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
        """
        Select features for variant.
        """
        feature_cols = FEATURE_SCHEMA.get_features_for_variant(variant)
        
        # Always include IDs and Target
        cols_to_select = list(FEATURE_SCHEMA.id_columns) + [FEATURE_SCHEMA.target] + feature_cols
        
        # Include Benchmarks for Evaluation
        cols_to_select += list(FEATURE_SCHEMA.benchmark_features)
        
        # Add graph identities if needed? FeatureSchema graph_identity_columns
        # They might be needed for graph building, not necessarily "features" for XGBoost
        # But HybridPipeline might need them.
        # Let's verify overlap.
        
        # Dedup
        cols_to_select = sorted(list(set(cols_to_select)))
        
        # Check availability
        # Note: We rely on Load Data to have created derived cols.
        # But for raw columns (like graph hash), we need to ensure they are picked up.
        
        # Add keys for graph building
        for key in FEATURE_SCHEMA.graph_identity_columns:
            if key not in cols_to_select:
                cols_to_select.append(key)
                
        # Filter strictly what's available to avoid crash, but warn
        schema_keys = df.collect_schema().names()
        final_cols = [c for c in cols_to_select if c in schema_keys]
        
        missing = set(cols_to_select) - set(final_cols)
        if missing:
            # Clean up known post-computed columns
            # e.g. gnn_emb, handcrafted
            known_missing = [c for c in missing if "gnn_emb" in c or "degree" in c or "neighbor" in c]
            real_missing = missing - set(known_missing)
            if real_missing:
                logger.warning(f"Missing columns in dataframe: {real_missing}")
        
        return df.select(final_cols)
