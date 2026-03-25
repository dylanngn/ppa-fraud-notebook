"""
Data extraction module for SEON + Snowflake events pipeline.
"""
import logging
from pathlib import Path
from typing import Tuple

import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def scan_events_lazy(
    path: str,
    id_source_col: str = "OBJECTREFERENCE",
    id_pattern: str = r"#(.+?)##",
    time_col: str = "DATAPIPELINE_EVENT_SENT_AT",
    status_col: str = "STATUS",
    exclude_statuses: list = None,
    platform_col: str = None,
    allowed_platforms: list = None,
) -> pl.LazyFrame:
    """
    Create a LazyFrame for streaming events processing.

    Args:
        path: Path to events CSV file
        id_source_col: Column containing INSERTION_ID (to be extracted via regex)
        id_pattern: Regex pattern to extract INSERTION_ID
        time_col: Timestamp column name
        status_col: Status column name
        exclude_statuses: Status values to drop (e.g. DRAFT, REPUBLISHING)
        platform_col: Column identifying the platform (e.g. targetplatform)
        allowed_platforms: Lowercase platform values to keep (e.g. homegate, immoscout24)

    Returns:
        LazyFrame for streaming processing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Events file not found: {path}")
    
    logger.info(f"Scanning events from {path}...")
    
    events_lf = (
        pl.scan_csv(
        path,
        separator=",",
        ignore_errors=True,
        infer_schema_length=10000,
    )
        .with_columns([
            pl.col(id_source_col).str.extract(id_pattern, 1).alias("INSERTION_ID"),
            pl.col(time_col)
            .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f %z", strict=False)
            .alias("event_dt"),
        ])
        .filter(
            pl.col("INSERTION_ID").is_not_null()
            & pl.col("event_dt").is_not_null()
            & pl.col(status_col).is_not_null()
            & ~pl.col(status_col).is_in(exclude_statuses or ["DRAFT", "REPUBLISHING"])
        )
    )

    # Filter to allowed platforms if configured
    if platform_col and allowed_platforms:
        events_lf = events_lf.filter(
            pl.col(platform_col).str.to_lowercase().is_in(allowed_platforms)
        )
    # Keep only the first event per (INSERTION_ID, STATUS) — one row per transition
    events_lf = (
        events_lf
        .sort(["INSERTION_ID", status_col, time_col])
        .with_columns(
            pl.int_range(pl.len()).over(["INSERTION_ID", status_col]).alias("_rn")
        )
        .filter(pl.col("_rn") == 0)
        .drop("_rn")
    )

    return events_lf


def load_seon(
    path: str,
    id_col: str = "transaction_id",
    time_col: str = "date",
) -> pl.DataFrame:
    """
    Load SEON transactions CSV.
    
    Args:
        path: Path to SEON CSV file
        id_col: Column containing INSERTION_ID equivalent
        time_col: Timestamp column name
        
    Returns:
        DataFrame with SEON transactions prepared for joining
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

    seon_df = (
        seon_df
        .rename({id_col: "INSERTION_ID"})
        .with_columns([
            pl.col("INSERTION_ID").cast(pl.Utf8),
            pl.col(time_col)
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z", strict=False)
            .alias("seon_dt"),
        ])
        .filter(pl.col("INSERTION_ID").is_not_null() & pl.col("seon_dt").is_not_null())
        .sort(["INSERTION_ID", "seon_dt"])
    )
    
    logger.info(f"  Loaded {len(seon_df):,} SEON transactions")
    logger.info(f"  Unique insertion IDs: {seon_df.select('INSERTION_ID').n_unique():,}")
    
    return seon_df


def extract_data(cfg: DictConfig) -> Tuple[pl.LazyFrame, pl.DataFrame]:
    """
    Main extraction function.
    
    Args:
        cfg: Hydra config with data.sources paths
        
    Returns:
        Tuple of (events_lazy, seon_df)
    """
    import hydra.utils

    events_path = hydra.utils.to_absolute_path(cfg.sources.events_csv)
    seon_path = hydra.utils.to_absolute_path(cfg.sources.seon_csv)

    seon_df = load_seon(
        path=seon_path,
        id_col=cfg.columns.seon.id_col,
        time_col=cfg.columns.seon.time_col,
    )

    events_lf = scan_events_lazy(
        path=events_path,
        id_source_col=cfg.columns.events.id_source,
        id_pattern=cfg.columns.events.id_pattern,
        time_col=cfg.columns.events.time_col,
        status_col=cfg.columns.events.status_col,
        exclude_statuses=list(cfg.columns.events.get("exclude_statuses", ["DRAFT", "REPUBLISHING"])),
        platform_col=cfg.columns.events.get("platform_col", None),
        allowed_platforms=list(cfg.columns.events.get("allowed_platforms", [])) or None,
    )
    
    return events_lf, seon_df
