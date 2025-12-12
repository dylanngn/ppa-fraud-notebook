"""
Temporal splitting logic for accumulated training.
Handles the critical task of preventing data leakage.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Iterator
import polars as pl

from src.data.schema import DataSplit, TemporalBoundary, FEATURE_SCHEMA


@dataclass
class AccumulatedTrainingConfig:
    """Configuration for accumulated training strategy."""
    
    # Data boundaries
    data_start_date: str           # First date in dataset
    data_end_date: str             # Last date in dataset
    
    # Accumulation strategy
    initial_train_months: int = 12  # Initial training window
    prediction_window_days: int = 7 # Predict 1 week ahead
    accumulation_frequency: str = "weekly"  # How often to retrain
    
    # Validation strategy
    val_window_days: int = 14       # Validation window size
    gap_days: int = 7               # Gap to prevent leakage
    
    # Retraining settings
    retrain_from_scratch: bool = False  # If False, use warm start
    max_history_months: Optional[int] = None  # Sliding window limit (None = expanding)


class TemporalSplitter:
    """
    Creates temporal splits for accumulated training.
    
    Key principle: At any point in time t, the model should only have
    access to data from time < t for training.
    """
    
    def __init__(self, config: AccumulatedTrainingConfig):
        self.config = config
        self.data_start = datetime.fromisoformat(config.data_start_date)
        self.data_end = datetime.fromisoformat(config.data_end_date)
    
    def generate_accumulated_splits(self) -> Iterator[DataSplit]:
        """
        Generate all splits for accumulated training simulation.
        
        Yields splits in chronological order, each representing
        a retraining point with expanding/sliding training data.
        """
        
        # Initial training end
        train_end = self.data_start + timedelta(days=self.config.initial_train_months * 30)
        
        while True:
            # Calculate split boundaries
            
            # Val starts after Gap
            val_start = train_end + timedelta(days=self.config.gap_days)
            val_end = val_start + timedelta(days=self.config.val_window_days)
            
            # Test starts after Gap (from Val end)
            test_start = val_end + timedelta(days=self.config.gap_days)
            test_end = test_start + timedelta(days=self.config.prediction_window_days)
            
            # Check if we've exceeded data bounds
            if test_end > self.data_end:
                break
            
            # Compute training start (for sliding window)
            if self.config.max_history_months:
                train_start_limit = train_end - timedelta(days=self.config.max_history_months * 30)
                train_start = max(self.data_start, train_start_limit)
            else:
                train_start = self.data_start
            
            # Create split
            accumulation_id = f"week_{train_end.strftime('%Y_%m_%d')}"
            
            split = DataSplit(
                train=TemporalBoundary(
                    cutoff_date=train_end.isoformat(),
                    split_name="train",
                    accumulation_id=accumulation_id,
                ),
                val=TemporalBoundary(
                    cutoff_date=val_end.isoformat(),
                    split_name="val",
                    accumulation_id=accumulation_id,
                ),
                test=TemporalBoundary(
                    cutoff_date=test_end.isoformat(),
                    split_name="test",
                    accumulation_id=accumulation_id,
                ),
                gap_days=self.config.gap_days,
            )
            
            split.validate()
            yield split
            
            # Move to next accumulation point
            if self.config.accumulation_frequency == "weekly":
                train_end += timedelta(days=7)
            elif self.config.accumulation_frequency == "monthly":
                train_end += timedelta(days=30)
            else:
                train_end += timedelta(days=7)
    
    def apply_split_to_dataframe(
        self,
        df: pl.LazyFrame,
        split: DataSplit,
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        
        train_end = datetime.fromisoformat(split.train.cutoff_date)
        val_start = train_end + timedelta(days=split.gap_days)
        val_end = datetime.fromisoformat(split.val.cutoff_date)
        test_start = val_end + timedelta(days=split.gap_days)
        test_end = datetime.fromisoformat(split.test.cutoff_date)
        
        # Use submission_at
        time_col = "submission_at"
        
        train_df = df.filter(pl.col(time_col) <= train_end)
        
        val_df = df.filter(
            (pl.col(time_col) > val_start) & 
            (pl.col(time_col) <= val_end)
        )
        
        test_df = df.filter(
            (pl.col(time_col) > test_start) & 
            (pl.col(time_col) <= test_end)
        )
        
        return train_df, val_df, test_df
    
    def create_split_masks(
        self,
        df: pl.DataFrame,
        split: DataSplit,
    ) -> Tuple[pl.Series, pl.Series, pl.Series]:
        
        train_end = datetime.fromisoformat(split.train.cutoff_date)
        val_start = train_end + timedelta(days=split.gap_days)
        val_end = datetime.fromisoformat(split.val.cutoff_date)
        test_start = val_end + timedelta(days=split.gap_days)
        test_end = datetime.fromisoformat(split.test.cutoff_date)
        
        timestamps = df["submission_at"]
        
        train_mask = timestamps <= train_end
        val_mask = (timestamps > val_start) & (timestamps <= val_end)
        test_mask = (timestamps > test_start) & (timestamps <= test_end)
        
        return train_mask, val_mask, test_mask
