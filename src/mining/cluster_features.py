"""
Cluster-Based Features for Fraud Detection.

Based on data analysis finding: SMALL CLUSTERS (2-4 connections) have 
highest fraud rates (15-22%), while large clusters (10+) have low fraud.

This is the correct graph signal - fraud rings form small, tight clusters.
"""

import polars as pl
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusterFeatureConfig:
    """Configuration for cluster features."""
    
    identity_columns: Tuple[str, ...] = (
        "ip_hash",
        "session/device_hash", 
        "LISTING_LISTER_EMAIL_hash",
        "user_id_hash",
    )
    
    # Cluster size buckets based on analysis
    small_cluster_min: int = 2
    small_cluster_max: int = 4
    power_user_threshold: int = 10
    
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT"
    label_column: str = "is_fraud"


class ClusterFeatureExtractor:
    """
    Extracts cluster-based features using the correct fraud signal.
    
    Key insight: Small clusters (2-4 connections) = fraud rings
    """
    
    def __init__(self, config: ClusterFeatureConfig = None):
        self.config = config or ClusterFeatureConfig()
        self._historical_counts: Dict[str, pl.DataFrame] = {}
    
    def fit(self, df: pl.DataFrame) -> "ClusterFeatureExtractor":
        """Compute cluster statistics from training data."""
        logger.info(f"Building cluster statistics from {len(df)} records")
        
        for col in self.config.identity_columns:
            if col not in df.columns:
                continue
            
            # Count occurrences per identity
            counts = (
                df
                .filter(pl.col(col).is_not_null())
                .group_by(col)
                .agg(pl.len().alias("cluster_size"))
            )
            
            self._historical_counts[col] = counts
            logger.info(f"  {col}: {len(counts)} unique identities")
        
        return self
    
    def transform(self, df: pl.DataFrame, is_training: bool = False) -> pl.DataFrame:
        """Extract cluster features."""
        logger.info(f"Extracting cluster features for {len(df)} records")
        
        result = df
        
        for col in self.config.identity_columns:
            if col not in df.columns:
                continue
            
            col_clean = col.replace("/", "_").replace("_hash", "")
            
            # Get cluster sizes
            if is_training:
                counts = (
                    df
                    .filter(pl.col(col).is_not_null())
                    .group_by(col)
                    .agg(pl.len().alias("_size"))
                )
            else:
                counts = self._historical_counts.get(col)
                if counts is None:
                    continue
                counts = counts.rename({"cluster_size": "_size"})
            
            # Join cluster sizes
            result = result.join(counts, on=col, how="left")
            
            # Create binary features
            result = result.with_columns([
                # Is this a small suspicious cluster (2-4)?
                ((pl.col("_size") >= self.config.small_cluster_min) & 
                 (pl.col("_size") <= self.config.small_cluster_max))
                .fill_null(False)
                .alias(f"cf_{col_clean}_is_small_cluster"),
                
                # Is this a power user (10+)?
                (pl.col("_size") >= self.config.power_user_threshold)
                .fill_null(False)
                .alias(f"cf_{col_clean}_is_power_user"),
                
                # Cluster size (capped)
                pl.col("_size")
                .fill_null(1)
                .clip(1, 100)
                .alias(f"cf_{col_clean}_cluster_size"),
            ])
            
            result = result.drop("_size")
        
        # Add aggregate features
        small_cluster_cols = [c for c in result.columns if c.endswith("_is_small_cluster")]
        power_user_cols = [c for c in result.columns if c.endswith("_is_power_user")]
        
        if small_cluster_cols:
            result = result.with_columns([
                # Count of small cluster signals
                pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in small_cluster_cols])
                .alias("cf_small_cluster_count"),
                
                # Any small cluster signal
                pl.any_horizontal([pl.col(c) for c in small_cluster_cols])
                .alias("cf_any_small_cluster"),
            ])
        
        if power_user_cols:
            result = result.with_columns([
                pl.sum_horizontal([pl.col(c).cast(pl.Int8) for c in power_user_cols])
                .alias("cf_power_user_count"),
            ])
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of cluster feature names."""
        features = []
        for col in self.config.identity_columns:
            col_clean = col.replace("/", "_").replace("_hash", "")
            features.extend([
                f"cf_{col_clean}_is_small_cluster",
                f"cf_{col_clean}_is_power_user",
                f"cf_{col_clean}_cluster_size",
            ])
        features.extend([
            "cf_small_cluster_count",
            "cf_any_small_cluster",
            "cf_power_user_count",
        ])
        return features
