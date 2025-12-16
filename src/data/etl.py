"""
ETL Pipeline for SEON + Snowflake Events data.

New pipeline: Extract from CSVs → Transform (correlate + merge) → Load to parquet

Usage:
    python -m src.data.etl
    python -m src.data.etl data.processing.batch_size=1000000
"""
import logging
import os
from pathlib import Path

import hydra
import hydra.utils
from omegaconf import DictConfig

from src.utils.hydra_utils import load_env

# Load .env file for anonymization salts
load_env()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main ETL pipeline.
    
    Steps:
    1. Extract: Load events and SEON CSVs
    2. Transform: Deduplicate, correlate, merge SEON features
    3. Load: Save to parquet
    """
    logger.info("=" * 60)
    logger.info("Starting ETL Pipeline: SEON + Snowflake Events")
    logger.info("=" * 60)
    
    # Import here to avoid circular imports with Hydra
    from src.data.extract import extract_data
    from src.data.transform import transform_data
    from src.data.load import load_data
    
    # Resolve paths
    output_path = hydra.utils.to_absolute_path(cfg.data.paths.output)
    
    # Check if output exists
    if os.path.exists(output_path):
        logger.info(f"Output exists: {output_path}")
        logger.info("Delete the file to re-run the pipeline, or use a different output path.")
        return
    
    # Create temp directory
    temp_dir = hydra.utils.to_absolute_path(cfg.data.paths.temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. EXTRACT
    logger.info("")
    logger.info("=" * 40)
    logger.info("STEP 1: EXTRACT")
    logger.info("=" * 40)
    
    events_df, seon_df = extract_data(cfg.data)
    
    logger.info(f"  Events: {len(events_df):,} rows, {len(events_df.columns)} columns")
    logger.info(f"  SEON: {len(seon_df):,} rows, {len(seon_df.columns)} columns")
    
    # 2. TRANSFORM
    logger.info("")
    logger.info("=" * 40)
    logger.info("STEP 2: TRANSFORM")
    logger.info("=" * 40)
    
    merged_df = transform_data(events_df, seon_df, cfg.data)
    
    # 3. LOAD
    logger.info("")
    logger.info("=" * 40)
    logger.info("STEP 3: LOAD")
    logger.info("=" * 40)
    
    load_data(merged_df, output_path)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("ETL Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Rows: {len(merged_df):,}")
    logger.info(f"  Columns: {len(merged_df.columns)}")
    
    if "is_fraud" in merged_df.columns:
        fraud_count = merged_df.filter(merged_df["is_fraud"] == 1).height
        fraud_rate = fraud_count / len(merged_df) * 100
        logger.info(f"  Fraud rate: {fraud_rate:.2f}%")


if __name__ == "__main__":
    main()
