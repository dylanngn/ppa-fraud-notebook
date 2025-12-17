"""
ETL Pipeline for SEON + Snowflake Events data.

Usage:
    python -m src.data.etl
"""
import logging
import os

import hydra
import hydra.utils
from omegaconf import DictConfig

from src.utils.hydra_utils import load_env

load_env()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main ETL pipeline.

    Steps:
    1. Extract: Scan events lazily, load SEON fully
    2. Transform: Correlate, merge, anonymize
    3. Load: Save to parquet
    """
    from src.data.extract import extract_data
    from src.data.transform import transform_data
    from src.data.load import load_data

    output_path = hydra.utils.to_absolute_path(cfg.data.paths.output)

    if os.path.exists(output_path):
        logger.info(f"Output exists: {output_path}")
        logger.info("Delete the file to re-run the pipeline.")
        return

    # Extract
    logger.info("=" * 40)
    logger.info("EXTRACT")
    logger.info("=" * 40)

    events_lf, seon_df = extract_data(cfg.data)
    logger.info(f"  SEON: {len(seon_df):,} rows, {len(seon_df.columns)} columns")

    # Transform
    logger.info("=" * 40)
    logger.info("TRANSFORM")
    logger.info("=" * 40)

    streaming = getattr(cfg.data.processing, "streaming", True)
    merged_df = transform_data(events_lf, seon_df, cfg.data, streaming=streaming)

    if merged_df.is_empty():
        logger.warning("No data to save.")
        return

    # Load
    logger.info("=" * 40)
    logger.info("LOAD")
    logger.info("=" * 40)

    load_data(merged_df, output_path)

    # Summary
    logger.info("=" * 40)
    logger.info("ETL Complete")
    logger.info("=" * 40)
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Rows: {len(merged_df):,}")
    logger.info(f"  Columns: {len(merged_df.columns)}")

    if "is_fraud" in merged_df.columns:
        fraud_count = merged_df.filter(merged_df["is_fraud"] == 1).height
        fraud_rate = fraud_count / len(merged_df) * 100 if len(merged_df) > 0 else 0
        logger.info(f"  Fraud rate: {fraud_rate:.2f}%")


if __name__ == "__main__":
    main()
