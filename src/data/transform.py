"""
Data transformation module for SEON + Snowflake events pipeline.
Handles deduplication, correlation, merging, and anonymization.
"""
import logging
from typing import List, Optional, Tuple

import polars as pl
from omegaconf import DictConfig

from src.data.extract import get_seon_feature_columns
from src.utils.anonymize import anonymize_merged_data

logger = logging.getLogger(__name__)


def deduplicate_events(
    events_df: pl.DataFrame,
    time_col: str = "TIME",
    status_col: str = "STATUS",
    fraud_col: str = "FLAGGEDFORFRAUD",
    id_col: str = "INSERTION_ID",
) -> pl.DataFrame:
    """
    Deduplicate events: keep first/last per status + all fraud-flagged events.
    
    Args:
        events_df: Events DataFrame
        time_col: Column with event timestamp
        status_col: Column with status
        fraud_col: Column with fraud flag
        id_col: Column with insertion ID
        
    Returns:
        Deduplicated DataFrame
    """
    logger.info("Deduplicating events...")
    initial_count = len(events_df)
    
    columns = events_df.columns
    events_df = events_df.sort([id_col, time_col])
    
    # 1. Keep all fraud-flagged events
    fraud_events = events_df.filter(pl.col(fraud_col).is_not_null())
    logger.info(f"  Fraud-flagged events: {len(fraud_events):,}")
    
    # 2. Get first event per (INSERTION_ID, STATUS)
    first_per_status = (
        events_df
        .group_by([id_col, status_col])
        .first()
        .select(columns)
    )
    
    # 3. Get last event per (INSERTION_ID, STATUS)
    last_per_status = (
        events_df
        .group_by([id_col, status_col])
        .last()
        .select(columns)
    )
    
    # 4. Combine and deduplicate
    combined = pl.concat([fraud_events, first_per_status, last_per_status])
    deduped = combined.unique(subset=[id_col, time_col])
    deduped = deduped.sort([id_col, time_col])
    
    logger.info(f"  Deduplicated: {initial_count:,} → {len(deduped):,} events")
    logger.info(f"  Reduction: {(1 - len(deduped)/initial_count)*100:.1f}%")
    
    return deduped


def remove_post_terminal_events(
    events_df: pl.DataFrame,
    time_col: str = "TIME",
    status_col: str = "STATUS",
    fraud_col: str = "FLAGGEDFORFRAUD",
    id_col: str = "INSERTION_ID",
) -> pl.DataFrame:
    """
    Remove events after first fraud flag or DELETED status.
    
    Args:
        events_df: Events DataFrame
        time_col: Column with event timestamp
        status_col: Column with status
        fraud_col: Column with fraud flag
        id_col: Column with insertion ID
        
    Returns:
        Filtered DataFrame
    """
    logger.info("Removing post-terminal events...")
    initial_count = len(events_df)
    
    # Find first fraud flag time per insertion
    fraud_events = events_df.filter(pl.col(fraud_col).is_not_null())
    first_fraud = (
        fraud_events
        .group_by(id_col)
        .agg(pl.col(time_col).min().alias("first_fraud_time"))
    )
    
    # Find first DELETED time per insertion
    deleted_events = events_df.filter(pl.col(status_col) == "DELETED")
    first_deleted = (
        deleted_events
        .group_by(id_col)
        .agg(pl.col(time_col).min().alias("first_deleted_time"))
    )
    
    # Join terminal times
    events_with_terminals = (
        events_df
        .join(first_fraud, on=id_col, how="left")
        .join(first_deleted, on=id_col, how="left")
    )
    
    # Keep events before or at terminal time
    filtered = events_with_terminals.filter(
        # No terminal events for this insertion
        ((pl.col("first_fraud_time").is_null()) & (pl.col("first_deleted_time").is_null())) |
        # Before or at fraud flag
        ((pl.col("first_fraud_time").is_not_null()) & (pl.col(time_col) <= pl.col("first_fraud_time"))) |
        # Before or at DELETED (if no fraud flag)
        ((pl.col("first_fraud_time").is_null()) & (pl.col("first_deleted_time").is_not_null()) & 
         (pl.col(time_col) <= pl.col("first_deleted_time")))
    )
    
    # Drop helper columns
    filtered = filtered.drop(["first_fraud_time", "first_deleted_time"])
    
    logger.info(f"  Removed: {initial_count - len(filtered):,} post-terminal events")
    
    return filtered


def correlate_events_to_seon(
    events_df: pl.DataFrame,
    seon_df: pl.DataFrame,
    events_time_col: str = "TIME",
    events_id_col: str = "INSERTION_ID",
    seon_time_col: str = "date",
    seon_id_col: str = "transaction_id",
) -> pl.DataFrame:
    """
    Correlate each event to its corresponding SEON transaction by time period.
    
    For each event, find the SEON TX where: seon.date <= event.TIME < next_seon.date
    
    Args:
        events_df: Events DataFrame with INSERTION_ID
        seon_df: SEON DataFrame with transaction_id
        events_time_col: Event timestamp column
        events_id_col: Event insertion ID column
        seon_time_col: SEON timestamp column
        seon_id_col: SEON insertion ID column
        
    Returns:
        Events DataFrame with seon_tx_id column for correlation
    """
    logger.info("Correlating events to SEON transactions...")
    
    # Rename SEON columns for clarity
    seon_times = (
        seon_df
        .select([seon_id_col, seon_time_col, "id"])
        .rename({
            seon_id_col: "INSERTION_ID",
            seon_time_col: "seon_date",
            "id": "seon_tx_id"
        })
        .sort(["INSERTION_ID", "seon_date"])
    )
    
    # Add next SEON date for each transaction (to define time windows)
    seon_times = seon_times.with_columns(
        pl.col("seon_date")
        .shift(-1)
        .over("INSERTION_ID")
        .alias("next_seon_date")
    )
    
    # Parse event times for comparison (handle timezone in data)
    events_df = events_df.with_columns(
        pl.col(events_time_col)
        .str.replace(" Z", "+00:00")  # Normalize timezone format
        .str.to_datetime(format=None, time_zone="UTC", strict=False)
        .alias("event_dt")
    )
    
    # Parse SEON times (ISO format with timezone)
    seon_times = seon_times.with_columns([
        pl.col("seon_date")
        .str.to_datetime(format=None, time_zone="UTC", strict=False)
        .alias("seon_dt"),
        pl.col("next_seon_date")
        .str.to_datetime(format=None, time_zone="UTC", strict=False)
        .alias("next_seon_dt"),
    ])
    
    # Join events with SEON times on INSERTION_ID
    merged = events_df.join(
        seon_times.select(["INSERTION_ID", "seon_tx_id", "seon_dt", "next_seon_dt"]),
        on="INSERTION_ID",
        how="left"
    )
    
    # Get first SEON tx per insertion for pre-SEON events
    first_seon_per_ins = (
        seon_times
        .filter(pl.col("seon_dt").is_not_null())
        .sort("seon_dt")
        .group_by("INSERTION_ID")
        .first()
        .select(["INSERTION_ID", "seon_tx_id", "seon_dt"])
        .rename({"seon_tx_id": "first_seon_tx_id", "seon_dt": "first_seon_dt"})
    )
    
    # Join first SEON info to merged data
    merged = merged.join(first_seon_per_ins, on="INSERTION_ID", how="left")
    
    # Assign events to correct SEON transaction:
    # 1. Events BEFORE first SEON → assign to first SEON tx
    # 2. Events AFTER first SEON → assign based on time window (seon_dt <= event < next_seon_dt)
    
    correlated = merged.with_columns(
        pl.when(
            # Event is before first SEON - assign to first SEON tx
            (pl.col("first_seon_dt").is_not_null()) &
            (pl.col("event_dt") < pl.col("first_seon_dt"))
        )
        .then(pl.col("first_seon_tx_id"))
        .when(
            # Event is within a SEON time window
            (pl.col("seon_dt").is_not_null()) &
            (pl.col("event_dt") >= pl.col("seon_dt")) &
            ((pl.col("next_seon_dt").is_null()) | (pl.col("event_dt") < pl.col("next_seon_dt")))
        )
        .then(pl.col("seon_tx_id"))
        .otherwise(None)
        .alias("final_seon_tx_id")
    )
    
    # Filter to only events with a valid SEON assignment
    correlated = correlated.filter(pl.col("final_seon_tx_id").is_not_null())
    
    # Replace seon_tx_id with the correct assignment
    correlated = correlated.drop("seon_tx_id").rename({"final_seon_tx_id": "seon_tx_id"})
    
    # Drop helper columns
    cols_to_drop = ["event_dt", "seon_dt", "next_seon_dt", "first_seon_tx_id", "first_seon_dt"]
    correlated = correlated.drop([c for c in cols_to_drop if c in correlated.columns])
    
    # Deduplicate: keep one row per (INSERTION_ID, TIME) - take the first SEON assignment
    result = correlated.unique(subset=["INSERTION_ID", events_time_col], keep="first")
    
    logger.info(f"  Correlated {len(result):,} events to SEON transactions")
    
    # Log events without SEON match
    uncorrelated = len(events_df) - len(result)
    if uncorrelated > 0:
        logger.warning(f"  Events without SEON correlation: {uncorrelated:,}")
    
    return result


def merge_seon_features(
    events_df: pl.DataFrame,
    seon_df: pl.DataFrame,
    feature_cols: List[str],
    seon_tx_id_col: str = "seon_tx_id",
    state_col: str = "state",
    benchmark_alias: str = "seon_state",
) -> pl.DataFrame:
    """
    Merge SEON features into events DataFrame.
    
    Args:
        events_df: Events DataFrame with seon_tx_id
        seon_df: SEON DataFrame
        feature_cols: List of SEON columns to include as features
        seon_tx_id_col: Column linking to SEON transaction
        state_col: SEON state column (benchmark only)
        benchmark_alias: Alias for benchmark column
        
    Returns:
        Events DataFrame with SEON features merged
    """
    logger.info("Merging SEON features...")
    
    # Select SEON columns to merge
    seon_id_col = "id"
    cols_to_select = [seon_id_col]
    
    # Add state as benchmark
    if state_col in seon_df.columns:
        cols_to_select.append(state_col)
    
    # Add feature columns (only those that exist)
    available_cols = set(seon_df.columns)
    for col in feature_cols:
        if col in available_cols:
            cols_to_select.append(col)
        else:
            logger.debug(f"  SEON column not found: {col}")
    
    logger.info(f"  Merging {len(cols_to_select) - 1} SEON columns")
    
    # Select and rename for join
    seon_subset = (
        seon_df
        .select(cols_to_select)
        .rename({seon_id_col: seon_tx_id_col})
    )
    
    # Rename state to benchmark alias
    if state_col in seon_subset.columns:
        seon_subset = seon_subset.rename({state_col: benchmark_alias})
    
    # Merge
    merged = events_df.join(seon_subset, on=seon_tx_id_col, how="left")
    
    logger.info(f"  Merged result: {len(merged):,} rows, {len(merged.columns)} columns")
    
    return merged


def derive_label(
    df: pl.DataFrame,
    fraud_col: str = "FLAGGEDFORFRAUD",
    label_col: str = "is_fraud",
) -> pl.DataFrame:
    """
    Derive binary fraud label from fraud flag column.
    
    Args:
        df: DataFrame with fraud flag column
        fraud_col: Column with fraud flag (not null = fraud)
        label_col: Name for derived label column
        
    Returns:
        DataFrame with label column
    """
    logger.info("Deriving fraud label...")
    
    df = df.with_columns(
        pl.col(fraud_col).is_not_null().cast(pl.Int8).alias(label_col)
    )
    
    fraud_count = df.filter(pl.col(label_col) == 1).height
    total_count = len(df)
    
    logger.info(f"  Fraud events: {fraud_count:,} / {total_count:,} ({fraud_count/total_count*100:.2f}%)")
    
    return df


def transform_data(
    events_df: pl.DataFrame,
    seon_df: pl.DataFrame,
    cfg: DictConfig,
) -> pl.DataFrame:
    """
    Main transformation pipeline.
    
    Args:
        events_df: Raw events DataFrame
        seon_df: Raw SEON DataFrame
        cfg: Hydra config
        
    Returns:
        Transformed and merged DataFrame
    """
    logger.info("Starting transformation pipeline...")
    
    # 1. Deduplicate events
    if cfg.processing.deduplicate_events:
        events_df = deduplicate_events(
            events_df,
            time_col=cfg.columns.events.time_col,
            status_col=cfg.columns.events.status_col,
            fraud_col=cfg.columns.events.fraud_flag_col,
            id_col="INSERTION_ID",
        )
    
    # 2. Remove post-terminal events
    if cfg.processing.remove_post_terminal_events:
        events_df = remove_post_terminal_events(
            events_df,
            time_col=cfg.columns.events.time_col,
            status_col=cfg.columns.events.status_col,
            fraud_col=cfg.columns.events.fraud_flag_col,
            id_col="INSERTION_ID",
        )
    
    # 3. Correlate events to SEON transactions
    events_df = correlate_events_to_seon(
        events_df,
        seon_df,
        events_time_col=cfg.columns.events.time_col,
        events_id_col="INSERTION_ID",
        seon_time_col=cfg.columns.seon.time_col,
        seon_id_col=cfg.columns.seon.id_col,
    )
    
    # 4. Merge SEON features
    feature_cols = get_seon_feature_columns(cfg)
    events_df = merge_seon_features(
        events_df,
        seon_df,
        feature_cols=feature_cols,
        state_col=cfg.columns.seon.state_col,
        benchmark_alias=cfg.benchmark_col,
    )
    
    # 5. Derive fraud label
    events_df = derive_label(
        events_df,
        fraud_col=cfg.columns.events.fraud_flag_col,
        label_col="is_fraud",
    )
    
    # 6. Anonymize PII
    events_df = anonymize_merged_data(events_df)
    
    logger.info(f"Transformation complete: {len(events_df):,} rows, {len(events_df.columns)} columns")
    
    return events_df


def save_merged_events_to_csv(
    df: pl.DataFrame,
    output_path: str = "data/merged_events.csv",
) -> None:
    """
    Save merged events DataFrame to CSV.
    
    Args:
        df: Merged events DataFrame
        output_path: Path to output CSV file
    """
    from pathlib import Path
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    logger.info(f"Saved merged events to {output_path} ({len(df):,} rows)")
