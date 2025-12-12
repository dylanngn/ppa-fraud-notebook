import os
import polars as pl
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def save_chunk(df: pl.DataFrame, path: str):
    """Save a processed chunk to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.write_parquet(path, compression="zstd")
    logger.info(f"Saved chunk to {path}")

def resolve_schema_conflicts(all_schemas: Dict[str, set]) -> Dict[str, pl.DataType]:
    """Resolve type conflicts across chunks and return unified schema."""
    final_schema = {}
    
    for col, dtypes in all_schemas.items():
        if len(dtypes) > 1:
            # Type conflict - check if it's numeric
            has_int = any(dt in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
            ] for dt in dtypes)
            has_float = any(dt in [pl.Float32, pl.Float64] for dt in dtypes)
            
            if has_int and has_float:
                final_schema[col] = pl.Float64
                logger.info(f"Schema conflict for {col}: {dtypes} -> using Float64")
            else:
                # Pick the first one (simple resolution)
                final_schema[col] = list(dtypes)[0]
        else:
            final_schema[col] = list(dtypes)[0]
    
    return final_schema

def assemble_chunks(chunk_dir: str) -> pl.DataFrame:
    """Assemble all chunks into a single DataFrame with schema resolution."""
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".parquet")])
    if not chunk_files:
        raise ValueError("No chunk files found!")
    
    logger.info(f"Found {len(chunk_files)} chunk files to assemble...")
    
    # Scan all chunks to detect schema conflicts
    logger.info("Scanning schemas to detect type conflicts...")
    lazy_chunks = [pl.scan_parquet(os.path.join(chunk_dir, f)) for f in chunk_files]
    
    # Collect all unique columns and their types
    all_schemas = {}
    for lf in lazy_chunks:
        schema = lf.collect_schema()
        for col, dtype in schema.items():
            if col not in all_schemas:
                all_schemas[col] = set()
            all_schemas[col].add(dtype)
    
    # Resolve schema conflicts
    final_schema = resolve_schema_conflicts(all_schemas)
    all_columns = sorted(all_schemas.keys())
    logger.info(f"Unified schema: {len(final_schema)} columns")
    
    # Read all chunks and ensure they match the unified schema
    dfs = []
    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        try:
            df_chunk = pl.read_parquet(chunk_path)
            
            # Build expressions to ensure all columns exist with correct types
            exprs = []
            for col in all_columns:
                if col in df_chunk.columns:
                    if df_chunk[col].dtype != final_schema[col]:
                        exprs.append(pl.col(col).cast(final_schema[col]))
                    else:
                        exprs.append(pl.col(col))
                else:
                    exprs.append(pl.lit(None).cast(final_schema[col]).alias(col))
            
            df_chunk = df_chunk.select(exprs)
            dfs.append(df_chunk)
        except Exception as e:
            logger.warning(f"Failed to read {chunk_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid chunk files found!")
    
    logger.info(f"Concatenating {len(dfs)} chunks...")
    return pl.concat(dfs)
