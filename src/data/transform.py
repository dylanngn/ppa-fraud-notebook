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

    n_matched = result.filter(pl.col(seon_unique_col).is_not_null()).height
    n_unmatched = total_events - n_matched

    stats = {
        "total_events": total_events,
        "matched_events": n_matched,
        "unmatched_events": n_unmatched,
    }

    if n_unmatched > 0:
        logger.warning(
            "Events without SEON match: %s (%.1f%%) — kept with null SEON features",
            f"{n_unmatched:,}",
            n_unmatched / total_events * 100,
        )

    logger.info("  Correlated: %s matched, %s unmatched", f"{n_matched:,}", f"{n_unmatched:,}")
    # Return ALL events — unmatched have null seon_unique_col; merge_seon_features uses left join
    return result, stats


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
    
    # Left join: events without a SEON match keep all event columns, SEON columns are null
    merged = events_df.join(
        seon_df,
        on=seon_unique_col,
        how="left",
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
    1. Collect all events from LazyFrame
    2. Propagate insertion-level fraud flag (FLAGGEDFORFRAUD lives on ARCHIVED/DELETED rows;
       must be propagated now before scoring-event filter removes those rows)
    3. Filter to scoring events only (PENDING_APPROVAL + PENDING_REPUBLISH_APPROVAL) —
       the only moments when SEON is called and the API scores a listing
    4. Asof-join each scoring event to its closest SEON transaction by timestamp;
       events with no SEON match are kept with null features (XGBoost handles nulls)
    5. Merge all SEON columns (left join)
    6. Derive preliminary fraud label from propagated FLAGGEDFORFRAUD
       (temporal_split will re-apply point-in-time correction during training)
    7. Anonymize PII
    8. Drop all-null columns

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
    fraud_flag_col = cfg.columns.events.fraud_flag_col
    scoring_statuses = list(
        cfg.columns.events.get("seon_scoring_statuses", ["PENDING_APPROVAL", "PENDING_REPUBLISH_APPROVAL"])
    )

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

    # Propagate insertion-level fraud timestamp BEFORE filtering to scoring events.
    # FLAGGEDFORFRAUD is only non-null on ARCHIVED/ARCHIVING/DELETED rows; propagating
    # it now ensures each PENDING_APPROVAL row carries the fraud flag for its insertion,
    # so temporal_split's point-in-time logic works correctly after filtering.
    if fraud_flag_col in events_df.columns:
        insertion_fraud = (
            events_df
            .filter(pl.col(fraud_flag_col).is_not_null())
            .group_by("INSERTION_ID")
            .agg(pl.col(fraud_flag_col).first().alias("_fraud_propagated"))
        )
        events_df = (
            events_df
            .drop(fraud_flag_col)
            .join(insertion_fraud, on="INSERTION_ID", how="left")
            .rename({"_fraud_propagated": fraud_flag_col})
        )
        n_fraud_insertions = insertion_fraud.height
        logger.info("  Propagated fraud flag for %s insertions", f"{n_fraud_insertions:,}")

    # Filter to scoring events only — the moments when SEON is called in production
    events_df = events_df.filter(pl.col("STATUS").is_in(scoring_statuses))
    logger.info(
        "  Scoring events (%s): %s",
        " / ".join(scoring_statuses),
        f"{len(events_df):,}",
    )

    if events_df.is_empty():
        logger.warning("No scoring events after status filter.")
        return events_df

    # Correlate each scoring event to its closest SEON transaction by timestamp.
    # Events with no SEON match are retained with null SEON id (see correlate_events_to_seon).
    events_df, _stats = correlate_events_to_seon(events_df, seon_df, seon_unique_col)

    # Merge SEON features (left join — keeps events with no SEON match as null rows)
    events_df = merge_seon_features(events_df, seon_df, seon_unique_col)

    # Derive preliminary label; temporal_split will re-apply PIT correction during training
    events_df = derive_label(events_df, fraud_col=fraud_flag_col)
    
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
