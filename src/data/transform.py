"""
Data transformation module for SEON + Snowflake events pipeline.
"""
import logging
from typing import Dict, Optional, Tuple, Union

import polars as pl
from omegaconf import DictConfig

from src.utils.anonymize import anonymize_merged_data

logger = logging.getLogger(__name__)


def correlate_events_to_seon(
    events_df: pl.DataFrame,
    seon_df: pl.DataFrame,
    seon_unique_col: str = "id",
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """
    Correlate each event to the closest SEON transaction by timestamp.

    Strategy: Find the closest SEON transaction AFTER the event.
    If none exists, fall back to the closest SEON BEFORE the event.

    Args:
        events_df: Events DataFrame with INSERTION_ID and event_dt
        seon_df: SEON DataFrame with INSERTION_ID and seon_dt
        seon_unique_col: Unique SEON transaction identifier column

    Returns:
        Tuple of (correlated DataFrame, stats dict)
    """
    logger.info("Correlating events to SEON transactions...")

    if seon_unique_col not in seon_df.columns:
        raise KeyError(f"SEON unique column '{seon_unique_col}' not found")

    total_events = len(events_df)

    if events_df.is_empty():
        return events_df, {"total_events": 0, "matched_events": 0, "unmatched_events": 0}

    seon_for_join = (
        seon_df
        .select(["INSERTION_ID", "seon_dt", seon_unique_col])
        .sort(["INSERTION_ID", "seon_dt"])
    )

    events_sorted = events_df.sort(["INSERTION_ID", "event_dt"])

    # Forward: closest SEON after the event
    fwd_cols = (
        events_sorted
        .join_asof(
            seon_for_join,
            left_on="event_dt",
            right_on="seon_dt",
            by="INSERTION_ID",
            strategy="forward",
        )
        .select([
            pl.all().exclude([seon_unique_col, "seon_dt"]),
            pl.col(seon_unique_col).alias(f"{seon_unique_col}_fwd"),
        ])
    )

    # Backward: closest SEON before the event
    bwd_cols = (
        events_sorted
        .join_asof(
            seon_for_join,
            left_on="event_dt",
            right_on="seon_dt",
            by="INSERTION_ID",
            strategy="backward",
        )
        .select([pl.col(seon_unique_col).alias(f"{seon_unique_col}_bwd")])
    )

    # Combine and pick forward if available, else backward
    combined = pl.concat([fwd_cols, bwd_cols], how="horizontal")
    result = combined.with_columns(
        pl.when(pl.col(f"{seon_unique_col}_fwd").is_not_null())
        .then(pl.col(f"{seon_unique_col}_fwd"))
        .otherwise(pl.col(f"{seon_unique_col}_bwd"))
        .alias(seon_unique_col)
    ).drop([f"{seon_unique_col}_fwd", f"{seon_unique_col}_bwd"])

    matched = result.filter(pl.col(seon_unique_col).is_not_null())

    stats = {
        "total_events": total_events,
        "matched_events": len(matched),
        "unmatched_events": total_events - len(matched),
    }

    if stats["unmatched_events"] > 0:
        logger.warning(
            "Events without SEON match: %s (%.2f%%)",
            f"{stats['unmatched_events']:,}",
            stats['unmatched_events'] / total_events * 100,
        )

    logger.info("  Correlated: %s → %s rows", f"{total_events:,}", f"{len(matched):,}")
    return matched, stats


def merge_seon_features(
    events_df: pl.DataFrame,
    seon_df: pl.DataFrame,
    seon_unique_col: str = "id",
) -> pl.DataFrame:
    """
    Merge all SEON columns into the events DataFrame.

    Args:
        events_df: Events DataFrame with seon unique id column
        seon_df: SEON DataFrame with all features
        seon_unique_col: Column to join on

    Returns:
        Merged DataFrame with all columns from both sources
    """
    logger.info("Merging SEON features...")

    merged = events_df.join(
        seon_df,
        on=seon_unique_col,
        how="inner",
        suffix="_seon",
    )

    logger.info("  Merged: %s columns", f"{len(merged.columns):,}")
    return merged


def derive_label(
    df: pl.DataFrame,
    fraud_col: str = "FLAGGEDFORFRAUD",
    label_col: str = "is_fraud",
) -> pl.DataFrame:
    """
    Derive binary fraud label from fraud flag column.
    """
    logger.info("Deriving fraud label...")

    df = df.with_columns(
        pl.col(fraud_col).is_not_null().cast(pl.Int8).alias(label_col)
    )

    fraud_count = df.filter(pl.col(label_col) == 1).height
    total_count = len(df)
    logger.info(f"  Fraud: {fraud_count:,} / {total_count:,} ({fraud_count/total_count*100:.2f}%)")

    return df


def drop_all_null_columns(df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Optional[float]]]:
    """
    Drop columns that are entirely null.
    """
    if df.is_empty():
        return df, {"dropped": [], "savings_mb": None}

    before_bytes = df.estimated_size()
    null_counts = df.null_count()
    row_counts = null_counts.row(0)
    empty_cols = [col for col, count in zip(null_counts.columns, row_counts) if count == len(df)]

    if not empty_cols:
        return df, {"dropped": [], "savings_mb": 0.0}

    pruned = df.drop(empty_cols)
    after_bytes = pruned.estimated_size()
    savings_mb = (before_bytes - after_bytes) / (1024 * 1024)

    return pruned, {"dropped": empty_cols, "savings_mb": savings_mb}


def transform_data(
    events_lf: Union[pl.LazyFrame, pl.DataFrame],
    seon_df: pl.DataFrame,
    cfg: DictConfig,
    streaming: bool = True,
) -> pl.DataFrame:
    """
    Main transformation pipeline.

    Steps:
    1. Collect events from LazyFrame
    2. Filter to events with matching SEON IDs
    3. Correlate events to SEON transactions by timestamp
    4. Merge all SEON columns
    5. Derive fraud label
    6. Anonymize PII
    7. Drop all-null columns

    Args:
        events_lf: LazyFrame or DataFrame of events
        seon_df: Prepared SEON DataFrame
        cfg: Hydra config
        streaming: Use streaming collection for LazyFrame

    Returns:
        Transformed DataFrame
    """
    logger.info("Starting transformation...")

    seon_unique_col = getattr(cfg.columns.seon, "unique_col", "id")

    # Collect events
    if isinstance(events_lf, pl.LazyFrame):
        logger.info("Collecting events (streaming=%s)...", streaming)
        events_df = events_lf.collect(streaming=streaming)
        logger.info("  Collected: %s events", f"{len(events_df):,}")
    else:
        events_df = events_lf

    if events_df.is_empty():
        logger.warning("No events to process.")
        return events_df

    # Filter to matching SEON IDs
    seon_ids = seon_df.select("INSERTION_ID").unique()
    events_df = events_df.join(seon_ids, on="INSERTION_ID", how="semi")
    logger.info("  After SEON filter: %s events", f"{len(events_df):,}")

    if events_df.is_empty():
        logger.warning("No events match SEON IDs.")
        return events_df

    # Correlate
    events_df, stats = correlate_events_to_seon(events_df, seon_df, seon_unique_col)
    if stats["matched_events"] == 0:
        return events_df

    # Merge SEON features
    events_df = merge_seon_features(events_df, seon_df, seon_unique_col)

    # Derive label
    events_df = derive_label(events_df, fraud_col=cfg.columns.events.fraud_flag_col)

    # Anonymize
    events_df = anonymize_merged_data(events_df)

    # Drop null columns
    events_df, drop_info = drop_all_null_columns(events_df)
    if drop_info["dropped"]:
        logger.info("Dropped %s null columns (saved %.2f MB)", len(drop_info["dropped"]), drop_info["savings_mb"])

    # Clean up helper columns
    for col in ["event_dt", "seon_dt"]:
        if col in events_df.columns:
            events_df = events_df.drop(col)

    logger.info("Transform complete: %s rows, %s columns", f"{len(events_df):,}", f"{len(events_df.columns):,}")
    return events_df
