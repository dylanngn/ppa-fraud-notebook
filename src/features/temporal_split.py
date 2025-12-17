"""
Temporal Split Utilities for Fraud Detection.

Implements Point-in-Time (PIT) correctness to prevent label leakage:
- Labels are only applied if the fraud flag was set BEFORE the cutoff time
- This ensures training doesn't use future information
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)


class LabelPropagation(Enum):
    """How fraud labels propagate across events."""
    # Point-in-time: Only use fraud labels known at cutoff (RECOMMENDED)
    POINT_IN_TIME = "point_in_time"
    # Insertion-level: All events for fraud insertion get label=1 (LEAKAGE RISK)
    INSERTION_LEVEL = "insertion_level"
    # Event-level: Only the flagged event gets label=1
    EVENT_LEVEL = "event_level"
    # First-event: Only predict on first event per insertion
    FIRST_EVENT = "first_event"


class OverlapMode(Enum):
    """How to handle insertions in both train and test."""
    ALLOW = "allow"
    TRAIN_PRIORITY = "train_priority"
    TEST_PRIORITY = "test_priority"
    REMOVE_BOTH = "remove_both"


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal splitting."""
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT"
    insertion_column: str = "INSERTION_ID_hash"
    label_column: str = "is_fraud"
    fraud_flag_column: str = "FLAGGEDFORFRAUD"  # Contains fraud timestamp
    gap_days: int = 7
    label_propagation: LabelPropagation = LabelPropagation.POINT_IN_TIME
    overlap_mode: OverlapMode = OverlapMode.TRAIN_PRIORITY
    min_events: int = 1


class TemporalSplitter:
    """
    Temporal train/test splitter with Point-in-Time label correctness.
    
    Point-in-Time Correctness:
    - For training (cutoff=train_end): Only use fraud labels where 
      fraud_timestamp <= train_end
    - For testing (cutoff=test_end): Only use fraud labels where
      fraud_timestamp <= test_end
    
    This prevents label leakage from future fraud flags.
    """
    
    def __init__(self, config: TemporalSplitConfig):
        self.config = config
    
    def split(
        self,
        df: pl.DataFrame,
        train_end: datetime,
        test_end: datetime,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Create temporal train/test split with Point-in-Time labels.
        
        Args:
            df: DataFrame with events and fraud flags
            train_end: End of training period
            test_end: End of test period
            
        Returns:
            (train_df, test_df) with correct Point-in-Time labels
        """
        cfg = self.config
        test_start = train_end + timedelta(days=cfg.gap_days)
        
        # Split by event time
        train_df = df.filter(pl.col(cfg.time_column) <= train_end)
        test_df = df.filter(
            (pl.col(cfg.time_column) > test_start) &
            (pl.col(cfg.time_column) <= test_end)
        )
        
        logger.info(f"Initial split: train={train_df.height}, test={test_df.height}")
        
        # Handle overlapping insertions
        train_df, test_df = self._handle_overlap(train_df, test_df)
        
        # Apply Point-in-Time labels
        train_df = self._apply_pit_labels(train_df, cutoff=train_end)
        test_df = self._apply_pit_labels(test_df, cutoff=test_end)
        
        self._log_split_stats(train_df, test_df)
        
        return train_df, test_df
    
    def _apply_pit_labels(
        self,
        df: pl.DataFrame,
        cutoff: datetime,
    ) -> pl.DataFrame:
        """
        Apply Point-in-Time labels based on when fraud was flagged.
        
        Only assigns is_fraud=1 if:
        1. The fraud flag timestamp is not null AND
        2. The fraud flag timestamp <= cutoff
        
        This ensures we don't use future fraud information.
        """
        cfg = self.config
        
        if cfg.label_propagation == LabelPropagation.EVENT_LEVEL:
            # Simple: just use existing label
            return df
        
        if cfg.label_propagation == LabelPropagation.FIRST_EVENT:
            # Filter to first event per insertion
            return df.sort(cfg.time_column).group_by(cfg.insertion_column).first()
        
        if cfg.fraud_flag_column not in df.columns:
            logger.warning(f"Fraud flag column '{cfg.fraud_flag_column}' not found")
            return df
        
        # Parse fraud timestamp (normalize to UTC)
        df = df.with_columns(
            pl.col(cfg.fraud_flag_column)
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("_fraud_timestamp")
        )
        
        if cfg.label_propagation == LabelPropagation.POINT_IN_TIME:
            # Point-in-Time: Only count fraud if flagged BEFORE cutoff
            df = df.with_columns(
                pl.when(
                    pl.col("_fraud_timestamp").is_not_null() &
                    (pl.col("_fraud_timestamp") <= cutoff)
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(cfg.label_column)
            )
            
            # For insertion-level propagation within PIT:
            # Find insertions flagged before cutoff, propagate to all their events
            fraud_insertions = (
                df.filter(pl.col(cfg.label_column) == 1)
                .select(cfg.insertion_column)
                .unique()
            )
            
            if fraud_insertions.height > 0:
                df = df.with_columns(
                    pl.when(pl.col(cfg.insertion_column).is_in(
                        fraud_insertions[cfg.insertion_column]
                    ))
                    .then(pl.lit(1))
                    .otherwise(pl.col(cfg.label_column))
                    .cast(pl.Int8)
                    .alias(cfg.label_column)
                )
            
            # Log PIT stats
            future_fraud = df.filter(
                pl.col("_fraud_timestamp").is_not_null() &
                (pl.col("_fraud_timestamp") > cutoff)
            ).height
            
            if future_fraud > 0:
                logger.info(
                    f"PIT: {future_fraud} events have fraud flags AFTER cutoff "
                    f"(correctly labeled as non-fraud)"
                )
        
        elif cfg.label_propagation == LabelPropagation.INSERTION_LEVEL:
            # Legacy: Insertion-level without PIT (LEAKAGE RISK!)
            logger.warning(
                "Using INSERTION_LEVEL without Point-in-Time correction. "
                "This may cause label leakage!"
            )
            fraud_insertions = (
                df.filter(pl.col("_fraud_timestamp").is_not_null())
                .select(cfg.insertion_column)
                .unique()
            )
            
            df = df.with_columns(
                pl.when(pl.col(cfg.insertion_column).is_in(
                    fraud_insertions[cfg.insertion_column]
                ))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias(cfg.label_column)
            )
        
        # Clean up temp column
        df = df.drop("_fraud_timestamp")
        
        return df
    
    def _handle_overlap(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Handle insertions that appear in both train and test."""
        cfg = self.config
        
        if cfg.insertion_column not in train_df.columns:
            return train_df, test_df
        
        train_insertions = set(train_df[cfg.insertion_column].unique().to_list())
        test_insertions = set(test_df[cfg.insertion_column].unique().to_list())
        overlap = train_insertions & test_insertions
        
        if not overlap:
            return train_df, test_df
        
        logger.info(f"Found {len(overlap)} insertions overlapping train/test")
        
        if cfg.overlap_mode == OverlapMode.TRAIN_PRIORITY:
            test_df = test_df.filter(~pl.col(cfg.insertion_column).is_in(list(overlap)))
            logger.info(f"Removed overlapping insertions from test (train priority)")
        elif cfg.overlap_mode == OverlapMode.TEST_PRIORITY:
            train_df = train_df.filter(~pl.col(cfg.insertion_column).is_in(list(overlap)))
            logger.info(f"Removed overlapping insertions from train (test priority)")
        elif cfg.overlap_mode == OverlapMode.REMOVE_BOTH:
            train_df = train_df.filter(~pl.col(cfg.insertion_column).is_in(list(overlap)))
            test_df = test_df.filter(~pl.col(cfg.insertion_column).is_in(list(overlap)))
            logger.info(f"Removed overlapping insertions from both sets")
        
        return train_df, test_df
    
    def _log_split_stats(self, train_df: pl.DataFrame, test_df: pl.DataFrame):
        cfg = self.config
        
        train_fraud = train_df.filter(pl.col(cfg.label_column) == 1).height
        test_fraud = test_df.filter(pl.col(cfg.label_column) == 1).height
        
        train_pct = train_fraud / train_df.height * 100 if train_df.height > 0 else 0
        test_pct = test_fraud / test_df.height * 100 if test_df.height > 0 else 0
        
        logger.info(
            f"Final split: train={train_df.height} ({train_fraud} fraud, {train_pct:.2f}%), "
            f"test={test_df.height} ({test_fraud} fraud, {test_pct:.2f}%)"
        )


class ExpandingWindowSplitter:
    """
    Expanding window splitter for longitudinal backtesting.
    
    Creates multiple train/test splits where training window expands:
    - Window 1: Train on months 1-2, test on month 3
    - Window 2: Train on months 1-3, test on month 4
    - Window 3: Train on months 1-4, test on month 5
    
    This evaluates model's ability to adapt to concept drift over time.
    Each window uses Point-in-Time labels for temporal correctness.
    """
    
    def __init__(
        self,
        config: TemporalSplitConfig,
        window_size_days: int = 30,
    ):
        self.config = config
        self.window_size_days = window_size_days
        self.base_splitter = TemporalSplitter(config)
    
    def generate_splits(
        self,
        df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        min_train_windows: int = 2,
    ) -> List[Tuple[pl.DataFrame, pl.DataFrame, datetime, datetime]]:
        """
        Generate expanding window splits.
        
        Args:
            df: Full DataFrame
            start_date: First training window start
            end_date: Last possible test window end
            min_train_windows: Minimum training windows before first test
            
        Returns:
            List of (train_df, test_df, train_end, test_end) tuples
        """
        splits = []
        window_delta = timedelta(days=self.window_size_days)
        
        current_train_end = start_date + window_delta * min_train_windows
        
        while current_train_end + window_delta <= end_date:
            test_end = current_train_end + window_delta + timedelta(days=self.config.gap_days)
            
            train_df, test_df = self.base_splitter.split(df, current_train_end, test_end)
            
            if train_df.height > 0 and test_df.height > 0:
                splits.append((train_df, test_df, current_train_end, test_end))
                logger.info(
                    f"Window {len(splits)}: train_end={current_train_end.date()}, "
                    f"test_end={test_end.date()}, "
                    f"train={train_df.height}, test={test_df.height}"
                )
            
            current_train_end += window_delta
        
        logger.info(f"Generated {len(splits)} expanding window splits")
        return splits


def validate_temporal_integrity(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT",
) -> bool:
    """Validate that train and test are temporally separated."""
    train_max = train_df[time_column].max()
    test_min = test_df[time_column].min()
    
    if train_max >= test_min:
        raise ValueError(f"Temporal overlap: train_max={train_max}, test_min={test_min}")
    
    logger.info(f"Temporal integrity validated: train ends {train_max}, test starts {test_min}")
    return True


def compute_pit_label_stats(
    df: pl.DataFrame,
    cutoff: datetime,
    fraud_flag_column: str = "FLAGGEDFORFRAUD",
    insertion_column: str = "INSERTION_ID_hash",
) -> dict:
    """
    Compute statistics about Point-in-Time label assignment.
    
    Useful for understanding how many labels would be different
    with vs without PIT correction.
    """
    # Parse fraud timestamp (normalize to UTC)
    df = df.with_columns(
        pl.col(fraud_flag_column)
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
        .dt.replace_time_zone("UTC")
        .alias("_fraud_ts")
    )
    
    # Count insertions with fraud flag
    total_fraud_insertions = (
        df.filter(pl.col("_fraud_ts").is_not_null())
        .select(insertion_column)
        .n_unique()
    )
    
    # Count insertions with fraud flag BEFORE cutoff
    pit_fraud_insertions = (
        df.filter(
            pl.col("_fraud_ts").is_not_null() &
            (pl.col("_fraud_ts") <= cutoff)
        )
        .select(insertion_column)
        .n_unique()
    )
    
    # Count insertions with fraud flag AFTER cutoff (would leak)
    future_fraud_insertions = (
        df.filter(
            pl.col("_fraud_ts").is_not_null() &
            (pl.col("_fraud_ts") > cutoff)
        )
        .select(insertion_column)
        .n_unique()
    )
    
    return {
        "total_fraud_insertions": total_fraud_insertions,
        "pit_fraud_insertions": pit_fraud_insertions,
        "future_fraud_insertions": future_fraud_insertions,
        "leakage_prevented": future_fraud_insertions,
    }
