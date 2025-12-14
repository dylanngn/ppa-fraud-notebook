import os
import logging
from datetime import datetime, timedelta

import hydra
import hydra.utils
from omegaconf import DictConfig

from src.utils.hydra_utils import load_env

# Load environment variables before Hydra config resolution
load_env()

from src.data.extract import build_chunk_query, fetch_chunk
from src.data.transform import process_chunk_data
from src.data.load import save_chunk, assemble_chunks

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"
DATE_FORMAT_COMPACT = "%Y%m%d"
CHUNK_PREFIX = "chunk_"
CHUNK_SUFFIX = ".parquet"

def get_last_processed_date(temp_dir: str, start_dt: datetime) -> datetime:
   
    if not os.path.exists(temp_dir):
        logger.debug(f"Temp directory does not exist: {temp_dir}")
        return start_dt
        
    existing_chunks = sorted([
        f for f in os.listdir(temp_dir) 
        if f.startswith(CHUNK_PREFIX) and f.endswith(CHUNK_SUFFIX)
    ])
    
    if not existing_chunks:
        logger.debug("No existing chunks found")
        return start_dt
    
    latest_chunk = existing_chunks[-1]
    logger.info(f"Found {len(existing_chunks)} existing chunk(s), latest: {latest_chunk}")
    
    # Try to parse date from filename (format: chunk_YYYYMMDD_YYYYMMDD.parquet)
    try:
        parts = latest_chunk.replace(CHUNK_PREFIX, "").replace(CHUNK_SUFFIX, "").split("_")
        if len(parts) >= 2:
            chunk_end_str = parts[1]
            last_date = datetime.strptime(chunk_end_str, DATE_FORMAT_COMPACT)
            logger.info(f"Resuming from last processed date: {last_date.strftime(DATE_FORMAT)}")
            return last_date
        else:
            logger.warning(f"Chunk filename has unexpected format: {latest_chunk}")
            return start_dt
    except ValueError as e:
        logger.warning(f"Failed to parse date from chunk filename '{latest_chunk}': {e}")
        return start_dt
    except IndexError as e:
        logger.warning(f"Failed to extract date parts from chunk filename '{latest_chunk}': {e}")
    return start_dt

@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main ETL function.
    """
    logger.info("Starting ETL pipeline...")
    
    if not cfg.data.db_uri:
        raise ValueError("DB_URI not set in config or environment variables")
    
    temp_dir = hydra.utils.to_absolute_path(cfg.data.paths.temp_chunks)
    output_path = hydra.utils.to_absolute_path(cfg.data.paths.raw_insertions)
    
    if os.path.exists(output_path) and not cfg.data.extract.force_refresh:
        logger.info(f"{output_path} exists. Skipping fetch.")
        return

    os.makedirs(temp_dir, exist_ok=True)
    
    start_dt = datetime.strptime(cfg.data.extract.start_date, DATE_FORMAT)
    end_dt = datetime.strptime(cfg.data.extract.end_date, DATE_FORMAT)
    days_per_chunk = cfg.data.extract.days_per_chunk
    current_start = get_last_processed_date(temp_dir, start_dt)
    
    chunk_idx = 0
    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=days_per_chunk), end_dt)
        
        chunk_start_str = current_start.strftime(DATE_FORMAT)
        chunk_end_str = current_end.strftime(DATE_FORMAT)
        chunk_filename = f"{CHUNK_PREFIX}{current_start.strftime(DATE_FORMAT_COMPACT)}_{current_end.strftime(DATE_FORMAT_COMPACT)}{CHUNK_SUFFIX}"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        if os.path.exists(chunk_path):
            logger.info(f"Skipping chunk {chunk_idx} ({chunk_start_str} to {chunk_end_str}) - already exists")
            current_start = current_end
            chunk_idx += 1
            continue
            
        logger.info(f"Processing chunk {chunk_idx} ({chunk_start_str} to {chunk_end_str})...")
        
        # 1. Extract
        query = build_chunk_query(cfg.data.db_schema, chunk_start_str, chunk_end_str)
        df_chunk = fetch_chunk(cfg.data.db_uri, query)
        
        if df_chunk is not None:
            # 2. Transform
            df_processed = process_chunk_data(df_chunk)
            
            # 3. Load
            save_chunk(df_processed, chunk_path)
            logger.info(f"Saved chunk {chunk_idx} with {len(df_processed)} rows")
        else:
            logger.info(f"No data for chunk {chunk_idx}")
            
        current_start = current_end
        chunk_idx += 1
    
    logger.info("Assembling final dataset...")
    try:
        df_final = assemble_chunks(temp_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.write_parquet(output_path, compression="zstd")
        logger.info(f"ETL Complete. Data saved to '{output_path}' ({len(df_final)} rows)")
        
    except ValueError as e:
        logger.error(f"Assembly failed: {e}")

if __name__ == "__main__":
    main()
