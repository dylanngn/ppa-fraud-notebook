"""
Data extraction module for SEON + Snowflake events pipeline.
Loads data from CSV files with memory-efficient batch processing.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_events_batched(
    path: str,
    batch_size: int = 500_000,
    id_pattern: str = r"#(.+?)##",
) -> pl.DataFrame:
    """
    Load events CSV in batches for memory efficiency.
    
    Args:
        path: Path to events CSV file
        batch_size: Number of rows per batch
        id_pattern: Regex pattern to extract INSERTION_ID from OBJECTREFERENCE
        
    Returns:
        DataFrame with all events and extracted INSERTION_ID
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    
    logger.info(f"Loading events from {path} (batch_size={batch_size:,})...")
    
    reader = pl.read_csv_batched(
        path,
        separator=",",
        batch_size=batch_size,
        ignore_errors=True,
        infer_schema_length=10000,
    )
    
    dfs = []
    batch_idx = 0
    total_rows = 0
    
    while True:
        batches = reader.next_batches(1)
        if batches is None or len(batches) == 0:
            break
        
        batch = batches[0]
        
        # Extract INSERTION_ID from OBJECTREFERENCE
        if "OBJECTREFERENCE" in batch.columns:
            batch = batch.with_columns(
                pl.col("OBJECTREFERENCE")
                .str.extract(id_pattern, 1)
                .alias("INSERTION_ID")
            )
        
        dfs.append(batch)
        total_rows += len(batch)
        batch_idx += 1
        
        if batch_idx % 10 == 0:
            logger.info(f"  Processed {batch_idx} batches ({total_rows:,} rows)...")
    
    if not dfs:
        raise ValueError("No data found in events file")
    
    logger.info(f"  Concatenating {len(dfs)} batches...")
    events_df = pl.concat(dfs)
    
    logger.info(f"  Loaded {len(events_df):,} events")
    return events_df


def load_seon(path: str) -> pl.DataFrame:
    """
    Load SEON transactions CSV.
    
    Args:
        path: Path to SEON CSV file
        
    Returns:
        DataFrame with SEON transactions
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SEON file not found: {path}")
    
    logger.info(f"Loading SEON transactions from {path}...")
    
    seon_df = pl.read_csv(
        path,
        separator=";",
        ignore_errors=True,
        truncate_ragged_lines=True,
        infer_schema_length=10000,
    )
    
    logger.info(f"  Loaded {len(seon_df):,} SEON transactions")
    logger.info(f"  Unique insertion IDs: {seon_df.select('transaction_id').n_unique():,}")
    
    return seon_df


def extract_data(cfg: DictConfig) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Main extraction function - loads both events and SEON data.
    
    Args:
        cfg: Hydra config with data.sources paths
        
    Returns:
        Tuple of (events_df, seon_df)
    """
    # Load events
    events_df = load_events_batched(
        path=cfg.sources.events_csv,
        batch_size=cfg.processing.batch_size,
        id_pattern=cfg.columns.events.id_pattern,
    )
    
    # Load SEON
    seon_df = load_seon(path=cfg.sources.seon_csv)
    
    # Log correlation stats
    event_ids = set(events_df.select("INSERTION_ID").drop_nulls().unique().to_series().to_list())
    seon_ids = set(seon_df.select("transaction_id").drop_nulls().unique().to_series().to_list())
    
    common_ids = event_ids & seon_ids
    events_only = event_ids - seon_ids
    seon_only = seon_ids - event_ids
    
    logger.info("Correlation stats:")
    logger.info(f"  Events with SEON match: {len(common_ids):,}")
    logger.info(f"  Events without SEON: {len(events_only):,}")
    logger.info(f"  SEON without events: {len(seon_only):,}")
    
    return events_df, seon_df


def get_seon_feature_columns(cfg: DictConfig) -> list:
    """
    Get list of SEON columns to use as features.
    
    Args:
        cfg: Hydra config with seon_feature_groups
        
    Returns:
        List of column names
    """
    feature_cols = []
    
    for group_name, columns in cfg.seon_feature_groups.items():
        feature_cols.extend(columns)
    
    return feature_cols
