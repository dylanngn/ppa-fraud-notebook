"""
Temporal split utilities for time-series cross-validation.

Handles:
- Accumulating window splits for XGBoost training
- Train/test splits for GNN training
- Temporal filtering to prevent data leakage
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Tuple, Optional, Dict, Any
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class AccumulatingWindowSplitter:
    """
    Generate accumulating window splits for time-series cross-validation.
    
    Each window uses all data from the start up to train_end for training,
    and a fixed test period after train_end for testing.
    
    Usage:
        splitter = AccumulatingWindowSplitter(
            df=df,
            initial_window_days=180,
            step_days=14,
            test_days=14
        )
        
        for window_idx, train_df, test_df, window_info in splitter.split():
            # Train model on train_df, evaluate on test_df
            model.fit(train_df, labels_train)
            metrics = evaluate(model, test_df, labels_test)
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        initial_window_days: int = 180,
        step_days: int = 14,
        test_days: int = 14,
        time_column: str = "submission_at",
        min_train_samples: int = 1000,
        min_test_samples: int = 50,
        max_windows: Optional[int] = None
    ):
        """
        Initialize accumulating window splitter.
        
        Args:
            df: DataFrame with time column and target
            initial_window_days: Days in initial training window
            step_days: Days to step forward for each window
            test_days: Days in each test window
            time_column: Name of timestamp column
            min_train_samples: Minimum samples required for training
            min_test_samples: Minimum samples required for testing
            max_windows: Optional maximum number of windows (for debugging)
        """
        self.df = df.sort(time_column)
        self.initial_window_days = initial_window_days
        self.step_days = step_days
        self.test_days = test_days
        self.time_column = time_column
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.max_windows = max_windows
        
        # Extract date range
        self.start_date = df[time_column].min()
        self.end_date = df[time_column].max()
        
        logger.info(f"AccumulatingWindowSplitter initialized")
        logger.info(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"  Initial window: {initial_window_days} days")
        logger.info(f"  Step: {step_days} days, Test: {test_days} days")
    
    def split(self) -> Iterator[Tuple[int, pl.DataFrame, pl.DataFrame, Dict[str, Any]]]:
        """
        Generate train/test splits using accumulating window strategy.
        
        Yields:
            Tuple of (window_idx, train_df, test_df, window_info)
            
            window_info contains:
                - train_start: datetime
                - train_end: datetime
                - test_start: datetime (same as train_end)
                - test_end: datetime
                - train_size: int
                - test_size: int
        """
        current_date = self.start_date + timedelta(days=self.initial_window_days)
        test_size = timedelta(days=self.test_days)
        step_size = timedelta(days=self.step_days)
        
        window_idx = 0
        
        while current_date + test_size <= self.end_date:
            train_end = current_date
            test_end = current_date + test_size
            
            # ACCUMULATING WINDOW: Use ALL data from start to train_end
            train_df = self.df.filter(pl.col(self.time_column) < train_end)
            test_df = self.df.filter(
                (pl.col(self.time_column) >= train_end) & 
                (pl.col(self.time_column) < test_end)
            )
            
            # Skip if insufficient samples
            if len(test_df) < self.min_test_samples or len(train_df) < self.min_train_samples:
                current_date += step_size
                if self.max_windows is not None and window_idx >= self.max_windows:
                    break
                continue
            
            # Skip if no fraud in test set
            if "is_fraud" in test_df.columns and test_df["is_fraud"].sum() == 0:
                current_date += step_size
                if self.max_windows is not None and window_idx >= self.max_windows:
                    break
                continue
            
            window_info = {
                "train_start": self.start_date,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "train_size": len(train_df),
                "test_size": len(test_df),
            }
            
            logger.info(
                f"Window {window_idx}: train up to {train_end.date()}, "
                f"test {train_end.date()} → {test_end.date()} "
                f"(train={len(train_df)}, test={len(test_df)})"
            )
            
            yield window_idx, train_df, test_df, window_info
            
            window_idx += 1
            current_date += step_size
            
            if self.max_windows is not None and window_idx >= self.max_windows:
                break
        
        logger.info(f"Generated {window_idx} windows")


class TemporalTrainTestSplitter:
    """
    Simple temporal train/test split for GNN training.
    
    Splits data at a percentile of the valid timestamp range.
    Filters out very old/invalid timestamps using a threshold.
    
    Usage:
        splitter = TemporalTrainTestSplitter(
            timestamps=timestamps,
            split_percent=0.8,
            start_threshold_percentile=10
        )
        
        train_mask, test_mask = splitter.split()
    """
    
    def __init__(
        self,
        timestamps: np.ndarray,
        split_percent: float = 0.8,
        start_threshold_percentile: float = 10,
        start_threshold_timestamp: Optional[float] = None
    ):
        """
        Initialize temporal train/test splitter.
        
        Args:
            timestamps: Array of timestamps (nanoseconds)
            split_percent: Percentile for train/test split (0.0 to 1.0)
            start_threshold_percentile: Percentile to use as start threshold
                (filters out very old/invalid data). Ignored if start_threshold_timestamp is provided.
            start_threshold_timestamp: Optional explicit start threshold timestamp (nanoseconds).
                If provided, overrides start_threshold_percentile.
        """
        self.timestamps = timestamps
        self.split_percent = split_percent
        
        # Determine start threshold
        if start_threshold_timestamp is not None:
            self.start_threshold = start_threshold_timestamp
            logger.info(f"Using explicit start threshold: {datetime.fromtimestamp(start_threshold_timestamp / 1e9)}")
        else:
            # Use percentile-based threshold
            valid_timestamps = timestamps[timestamps > 0]
            if len(valid_timestamps) > 0:
                self.start_threshold = np.percentile(valid_timestamps, start_threshold_percentile)
                logger.info(f"Using {start_threshold_percentile}th percentile as start threshold: {datetime.fromtimestamp(self.start_threshold / 1e9)}")
            else:
                self.start_threshold = 0
                logger.warning("No valid timestamps found, using 0 as start threshold")
        
        # Get valid timestamps (after threshold)
        self.valid_mask = timestamps >= self.start_threshold
        self.valid_timestamps = timestamps[self.valid_mask]
        
        if len(self.valid_timestamps) == 0:
            logger.warning("No timestamps after threshold, using all timestamps")
            self.valid_timestamps = timestamps
            self.valid_mask = np.ones(len(timestamps), dtype=bool)
        
        # Calculate split time
        self.split_time = np.percentile(self.valid_timestamps, split_percent * 100)
        
        logger.info(f"TemporalTrainTestSplitter initialized")
        logger.info(f"  Split time: {datetime.fromtimestamp(self.split_time / 1e9)}")
        logger.info(f"  Valid samples: {len(self.valid_timestamps)} / {len(timestamps)}")
    
    def split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test masks.
        
        Returns:
            Tuple of (train_mask, test_mask) boolean arrays
        """
        train_mask = (self.timestamps <= self.split_time) & (self.timestamps >= self.start_threshold)
        test_mask = (self.timestamps > self.split_time) & (self.timestamps >= self.start_threshold)
        
        logger.info(f"Split: train={train_mask.sum()}, test={test_mask.sum()}")
        
        return train_mask, test_mask
    
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the split."""
        train_mask, test_mask = self.split()
        
        return {
            "split_time": self.split_time,
            "split_time_readable": datetime.fromtimestamp(self.split_time / 1e9),
            "start_threshold": self.start_threshold,
            "start_threshold_readable": datetime.fromtimestamp(self.start_threshold / 1e9),
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "split_percent": self.split_percent,
        }
