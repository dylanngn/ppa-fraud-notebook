import polars as pl
import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from src.utils.anonymize import anonymize_listings_pii

load_dotenv()

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("DB_URI environment variable not set")

DB_SCHEMA = os.getenv("DB_SCHEMA")
if not DB_SCHEMA:
    raise ValueError("DB_SCHEMA environment variable not set")

def fetch_raw_insertions(
    start_date: str = "2023-01-01",
    end_date: str = "2025-11-02",
    days_per_chunk: int = 7,
    force_refresh: bool = False
):
    """
    Fetches insertions from DB in date-range chunks, flattens JSON fields, anonymizes, and saves to artifacts/raw_insertions.parquet.
    Process: Fetch -> Flatten -> Anonymize -> Store
    
    Uses date-based chunking instead of fixed-size chunks for better tracking and incremental updates.
    Each chunk covers a date range (e.g., 7 days). Can resume from last processed date.
    
    Args:
        start_date: Start date for fetching (YYYY-MM-DD)
        end_date: End date for fetching (YYYY-MM-DD)
        days_per_chunk: Number of days per chunk (default: 7)
        force_refresh: If True, re-fetch even if raw_insertions.parquet exists
    """
    from datetime import datetime, timedelta
    
    if os.path.exists("artifacts/raw_insertions.parquet") and not force_refresh:
        print("raw_insertions.parquet exists. Skipping fetch.")
        print("To re-fetch, set force_refresh=True or delete artifacts/raw_insertions.parquet")
        return

    print("Extracting Insertions (Date-Range Chunked)...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Chunk size: {days_per_chunk} days")
    
    temp_dir = "artifacts/temp_raw_insertions"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Find last processed date by checking existing chunks
    last_processed_date = start_dt
    existing_chunks = sorted([f for f in os.listdir(temp_dir) if f.startswith("chunk_") and f.endswith(".parquet")])
    
    if existing_chunks:
        # Extract date range from latest chunk filename or metadata
        # Format: chunk_YYYYMMDD_YYYYMMDD.parquet
        latest_chunk = existing_chunks[-1]
        try:
            # Try to parse date from filename
            parts = latest_chunk.replace("chunk_", "").replace(".parquet", "").split("_")
            if len(parts) >= 2:
                chunk_end_str = parts[1]  # Second date is the end date
                last_processed_date = datetime.strptime(chunk_end_str, "%Y%m%d")
                print(f"Found existing chunks. Resuming from {last_processed_date.strftime('%Y-%m-%d')}...")
        except (ValueError, IndexError):
            # If filename parsing fails, check the actual data
            try:
                latest_df = pl.read_parquet(os.path.join(temp_dir, latest_chunk))
                if "submission_at" in latest_df.columns:
                    max_date = latest_df.select(pl.col("submission_at").max()).item()
                    if max_date:
                        last_processed_date = max_date if isinstance(max_date, datetime) else datetime.fromisoformat(str(max_date))
                        print(f"Found existing chunks. Resuming from {last_processed_date.strftime('%Y-%m-%d')}...")
            except:
                print("Could not determine last processed date. Starting from beginning.")
                last_processed_date = start_dt
    
    # Generate date ranges
    current_start = last_processed_date
    chunk_idx = len(existing_chunks)
    
    while current_start < end_dt:
        # Calculate chunk end date
        current_end = min(current_start + timedelta(days=days_per_chunk), end_dt)
        
        chunk_start_str = current_start.strftime("%Y-%m-%d")
        chunk_end_str = current_end.strftime("%Y-%m-%d")
        chunk_filename = f"chunk_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.parquet"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        # Skip if chunk already exists
        if os.path.exists(chunk_path):
            print(f"Skipping chunk {chunk_idx} ({chunk_start_str} to {chunk_end_str}) - already exists")
            current_start = current_end
            chunk_idx += 1
            continue
        
        print(f"Fetching chunk {chunk_idx} ({chunk_start_str} to {chunk_end_str})...")
        
        query_chunk = f"""
    SELECT 
        i.object_reference,
        i.user_id,
        i.user_ip_address,
        i.listing::text as listing_json,
        i.fraud_flag,
        i.auto_approval_criteria::text as auto_approval_criteria_json,
        i.first_published_date,
        i.customer_segment,
        i.selected_bundle::text as selected_bundle_json,
        i.created_at as listing_created_at,
        i.platform as listing_platform,
        sh.transition_timestamp as submission_at,
        u.owner_id,
        u.platform as user_platform,
        u.created_at as account_created_at,
        u.contact_emails
    FROM {DB_SCHEMA}.insertions i
    JOIN (
        SELECT insertion_id, min(transition_timestamp) as transition_timestamp
        FROM {DB_SCHEMA}.status_history
        WHERE status_from = 'DRAFT' AND status_to = 'PENDING_APPROVAL'
        GROUP BY insertion_id
    ) sh ON i.id = sh.insertion_id
    LEFT JOIN {DB_SCHEMA}.users u ON i.user_id = u.id
    WHERE sh.transition_timestamp >= '{chunk_start_str}' 
    AND sh.transition_timestamp < '{chunk_end_str}'
    AND i.platform <> 're.smg'
    AND meta -> 'migratedFromPersonId' is null
    ORDER BY submission_at
    """
        
        try:
            df_chunk = pl.read_database_uri(query_chunk, DB_URI, engine="connectorx")
            
            if len(df_chunk) == 0:
                print(f"  No data in date range {chunk_start_str} to {chunk_end_str}")
                current_start = current_end
                chunk_idx += 1
                continue
            
            print(f"  Fetched {len(df_chunk)} rows")
            
            # Step 1: Flatten JSON fields
            print(f"  Flattening JSON fields...")
            df_chunk_flat = flatten_chunk(df_chunk)
            print(f"  After flattening: {len(df_chunk_flat.columns)} columns")
            
            # Step 2: Anonymize flattened data
            print(f"  Anonymizing...")
            df_chunk_anon = anonymize_listings_pii(df_chunk_flat)
            print(f"  After anonymization: {len(df_chunk_anon.columns)} columns")
                
            df_chunk_anon.write_parquet(chunk_path, compression="zstd")
            print(f"  Saved {chunk_path} ({len(df_chunk_anon)} rows, {len(df_chunk_anon.columns)} columns)")
            print(f"  Date range: {chunk_start_str} to {chunk_end_str}")
            
            current_start = current_end
            chunk_idx += 1
            
        except Exception as e:
            print(f"Error fetching chunk {chunk_idx} ({chunk_start_str} to {chunk_end_str}): {e}")
            print("Stopping. You can resume later - it will continue from the last successful chunk.")
            raise e

    print("\nAssembling flattened & anonymized chunks...")
    # Read all chunks and concatenate with schema flexibility
    # Different chunks may have different types for the same column (e.g., Int64 vs Float64)
    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".parquet")])
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print(f"Found {len(chunk_files)} chunk files to assemble...")
    
    # First pass: scan all chunks to detect schema conflicts
    print("Scanning schemas to detect type conflicts...")
    lazy_chunks = [pl.scan_parquet(os.path.join(temp_dir, f)) for f in chunk_files]
    
    # Collect all unique columns and their types across all chunks
    all_schemas = {}
    for lf in lazy_chunks:
        schema = lf.collect_schema()  # Use collect_schema() to avoid performance warning
        for col, dtype in schema.items():
            if col not in all_schemas:
                all_schemas[col] = set()
            all_schemas[col].add(dtype)
    
    # Determine final schema: resolve type conflicts and collect all columns
    final_schema = {}
    all_columns = sorted(all_schemas.keys())
    
    for col in all_columns:
        dtypes = all_schemas[col]
        if len(dtypes) > 1:
            # Type conflict - check if it's numeric
            has_int = any(dt in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64] for dt in dtypes)
            has_float = any(dt in [pl.Float32, pl.Float64] for dt in dtypes)
            if has_int and has_float:
                # Use Float64 to handle both
                final_schema[col] = pl.Float64
                print(f"  Schema conflict for {col}: {dtypes} -> using Float64")
            else:
                # Use the first type (or most common)
                final_schema[col] = list(dtypes)[0]
        else:
            final_schema[col] = list(dtypes)[0]
    
    print(f"Unified schema: {len(final_schema)} columns")
    
    # Read all chunks and ensure they match the unified schema
    dfs = []
    for chunk_file in chunk_files:
        chunk_path = os.path.join(temp_dir, chunk_file)
        try:
            df_chunk = pl.read_parquet(chunk_path)
            
            # Build expressions to ensure all columns exist with correct types
            exprs = []
            for col in all_columns:
                if col in df_chunk.columns:
                    # Column exists - cast to final type if needed
                    if df_chunk[col].dtype != final_schema[col]:
                        exprs.append(pl.col(col).cast(final_schema[col]))
                    else:
                        exprs.append(pl.col(col))
                else:
                    # Column missing - add as null with correct type
                    exprs.append(pl.lit(None).cast(final_schema[col]).alias(col))
            
            df_chunk = df_chunk.select(exprs)
            dfs.append(df_chunk)
        except Exception as e:
            print(f"Warning: Failed to read {chunk_file}: {e}")
            continue
    
    if not dfs:
        print("No valid chunk files found!")
        return
    
    # Concatenate all chunks
    print(f"Concatenating {len(dfs)} chunks...")
    df_insertions = pl.concat(dfs)
    
    os.makedirs("artifacts", exist_ok=True)
    print(f"Saving {len(df_insertions)} flattened & anonymized insertions to artifacts/raw_insertions.parquet...")
    df_insertions.write_parquet("artifacts/raw_insertions.parquet", compression="zstd")
    
    # Save metadata about the date range
    if len(df_insertions) > 0 and "submission_at" in df_insertions.columns:
        min_date = df_insertions.select(pl.col("submission_at").min()).item()
        max_date = df_insertions.select(pl.col("submission_at").max()).item()
        print(f"\nData date range: {min_date} to {max_date}")
        print(f"Total rows: {len(df_insertions):,}") 

def extract_data():
    """
    Orchestrates the ETL process.
    """
    fetch_raw_insertions()
    
    print("Loading raw data from Parquet artifacts...")
    df_insertions = pl.read_parquet("artifacts/raw_insertions.parquet")
    
    return df_insertions

def flatten_dict_recursive(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary.
    
    Args:
        d: Dictionary or value to flatten
        parent_key: Parent key path
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary with dot-notation keys
    """
    items = []
    
    if d is None:
        return {}
    
    if isinstance(d, dict):
        for k, v in d.items():
            # Create the new key with prefix
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and v:
                # Recursively flatten nested dicts (only if dict is non-empty)
                nested_items = flatten_dict_recursive(v, new_key, sep=sep)
                items.extend(nested_items.items())
            elif isinstance(v, dict) and not v:
                # Empty dict - still add it as None or skip
                items.append((new_key, None))
            elif isinstance(v, list):
                # For arrays, convert to JSON string to preserve structure
                # Can be parsed later if needed
                items.append((new_key, json.dumps(v) if v else None))
            else:
                # Primitive value (str, int, float, bool, None)
                items.append((new_key, v))
    else:
        # Not a dict, return as-is with parent key
        key = parent_key if parent_key else "value"
        items.append((key, d))
    
    return dict(items)


def flatten_chunk(df_chunk: pl.DataFrame) -> pl.DataFrame:
    """
    Flattens JSON fields in a chunk and selects/aliases fields for downstream processing.
    
    This function:
    1. Flattens listing_json, selected_bundle_json, and auto_approval_criteria_json
    2. Combines with base columns
    3. Selects and aliases fields for downstream processing (same as process_listings output)
    
    Args:
        df_chunk: DataFrame chunk with JSON string columns
        
    Returns:
        DataFrame with flattened and aliased fields (ready for anonymization)
    """
    n_rows = len(df_chunk)
    
    listing_jsons = df_chunk["listing_json"].to_list()
    bundle_jsons = df_chunk["selected_bundle_json"].to_list() if "selected_bundle_json" in df_chunk.columns else [None] * n_rows
    criteria_jsons = df_chunk["auto_approval_criteria_json"].to_list() if "auto_approval_criteria_json" in df_chunk.columns else [None] * n_rows
    
    # Flatten JSON fields
    listing_flattened = flatten_json_batch(listing_jsons, prefix="listing")
    bundle_flattened = flatten_json_batch(bundle_jsons, prefix="bundle")
    criteria_flattened = flatten_json_batch(criteria_jsons, prefix="auto_approval_criteria")
    
    # Debug: Check if flattening is working
    if n_rows > 0:
        # Test flattening on first row manually to see what's happening
        if listing_jsons and listing_jsons[0]:
            try:
                test_json_str = listing_jsons[0]
                test_json = json.loads(test_json_str) if isinstance(test_json_str, str) else test_json_str
                print(f"  Debug: Parsed JSON type: {type(test_json)}, is dict: {isinstance(test_json, dict)}")
                if isinstance(test_json, dict):
                    print(f"  Debug: Top-level keys: {list(test_json.keys())[:5]}")
                test_flattened = flatten_dict_recursive(test_json, parent_key="listing")
                print(f"  Debug: Manual test - flattened keys: {len(test_flattened)}, sample keys: {list(test_flattened.keys())[:10]}")
            except Exception as e:
                import traceback
                print(f"  Debug: Manual test failed: {e}")
                print(f"  Debug: Traceback: {traceback.format_exc()}")
        
        # Check what flatten_json_batch actually returned
        if listing_flattened:
            print(f"  Debug: listing_flattened length: {len(listing_flattened)}")
            print(f"  Debug: First item type: {type(listing_flattened[0])}, is dict: {isinstance(listing_flattened[0], dict)}")
            if isinstance(listing_flattened[0], dict):
                print(f"  Debug: First item keys: {list(listing_flattened[0].keys())[:5] if listing_flattened[0] else 'EMPTY'}")
        
        sample_listing_keys = len(listing_flattened[0]) if listing_flattened and listing_flattened[0] else 0
        sample_bundle_keys = len(bundle_flattened[0]) if bundle_flattened and bundle_flattened[0] else 0
        sample_criteria_keys = len(criteria_flattened[0]) if criteria_flattened and criteria_flattened[0] else 0
        print(f"  Debug: Sample flattened keys - listing: {sample_listing_keys}, bundle: {sample_bundle_keys}, criteria: {sample_criteria_keys}")
        
        # Check if any flattened dicts have keys
        non_empty_listing = sum(1 for d in listing_flattened if d)
        print(f"  Debug: Non-empty listing dicts: {non_empty_listing}/{n_rows}")
    
    # Combine efficiently using list comprehension
    base_cols = [c for c in df_chunk.columns if c not in ["listing_json", "selected_bundle_json", "auto_approval_criteria_json"]]
    base_data = df_chunk.select(base_cols).to_dicts()
    
    # Merge flattened JSON data with base data
    combined_data = [
        {**base_data[i], **listing_flattened[i], **bundle_flattened[i], **criteria_flattened[i]}
        for i in range(n_rows)
    ]
    
    # Collect all possible keys to ensure all columns are created
    # This is important because Polars only creates columns that exist in early rows
    all_keys = set()
    for row in combined_data:
        all_keys.update(row.keys())
    
    print(f"  Debug: Total unique keys collected: {len(all_keys)} (base: {len(base_cols)}, flattened: {len(all_keys) - len(base_cols)})")
    
    # Get schema from base DataFrame to preserve types for existing columns
    base_df = df_chunk.select(base_cols)
    base_schema = {col: dtype for col, dtype in zip(base_df.columns, base_df.dtypes)}
    
    # Ensure all rows have all keys
    # For base columns, use the original value (they should always exist)
    # For new flattened columns, use None if missing
    combined_data_complete = []
    for i, row in enumerate(combined_data):
        complete_row = {}
        for key in all_keys:
            if key in base_cols:
                # Base columns should always exist, use original value
                complete_row[key] = row.get(key, base_data[i].get(key))
            else:
                # New flattened columns, use None if missing
                complete_row[key] = row.get(key, None)
        combined_data_complete.append(complete_row)
    
    # Create DataFrame from combined data
    # Use infer_schema_length=None to scan all rows for proper type inference
    # This is important when columns have mixed None + actual values
    df_flat = pl.DataFrame(combined_data_complete, infer_schema_length=None)
    
    # Cast base columns to their original types to ensure consistency
    # This prevents issues where None values cause type inference to fail
    for col in base_cols:
        if col in df_flat.columns and col in base_schema:
            original_dtype = base_schema[col]
            current_dtype = df_flat[col].dtype
            # Cast if types don't match (e.g., inferred as string but should be datetime)
            if current_dtype != original_dtype:
                try:
                    df_flat = df_flat.with_columns(
                        pl.col(col).cast(original_dtype, strict=False)
                    )
                except Exception as e:
                    # If casting fails, try to handle it gracefully
                    print(f"Warning: Could not cast {col} from {current_dtype} to {original_dtype}: {e}")
                    pass
    
    # Return flattened DataFrame with dot-notation field names (no aliasing)
    # The anonymization function will work with these original field names
    return df_flat


def flatten_json_batch(json_strings: list, prefix: str = "") -> list:
    """
    Efficiently flattens a batch of JSON strings.
    
    This processes JSON in batches rather than row-by-row.
    Uses list comprehension for better performance than explicit loops.
    
    Args:
        json_strings: List of JSON strings to flatten
        prefix: Prefix for flattened keys
        
    Returns:
        List of flattened dictionaries
    """
    def process_one(json_str):
        """Process a single JSON string."""
        if json_str is None:
            return {}
        
        try:
            # Parse JSON
            if isinstance(json_str, str):
                json_obj = json.loads(json_str)
            else:
                json_obj = json_str
            
            # Flatten recursively (use parent_key parameter, not prefix)
            flattened = flatten_dict_recursive(json_obj, parent_key=prefix)
            # Debug: Check if flattening actually worked
            if not flattened and json_obj:
                print(f"  Warning: Flattening returned empty dict for non-empty JSON. Type: {type(json_obj)}, Is dict: {isinstance(json_obj, dict)}")
            return flattened
        except (json.JSONDecodeError, TypeError) as e:
            # Log the error for debugging
            print(f"  Warning: Failed to flatten JSON: {type(e).__name__}: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return {}
        except Exception as e:
            # Catch any other errors
            print(f"  Warning: Unexpected error flattening JSON: {type(e).__name__}: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return {}
    
    # Use list comprehension for better performance
    return [process_one(js) for js in json_strings]



def main():
    """
    Main ETL function: fetches, flattens, and anonymizes data, then creates graph artifacts.
    """
    os.makedirs("artifacts", exist_ok=True)
    
    df_insertions = extract_data()
    print(f"Extracted {len(df_insertions)} insertions (flattened and anonymized).")
    
    # Create graph artifacts (nodes and edges) from flattened data
    from src.data.create_graph_artifacts import create_nodes_and_edges
    create_nodes_and_edges(df_insertions)
    
    print("ETL Complete. Data and graph artifacts saved to 'artifacts/' directory.")

if __name__ == "__main__":
    main()
