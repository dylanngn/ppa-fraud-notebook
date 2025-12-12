import json
import polars as pl
from typing import Any, Dict
import logging
from src.utils.anonymize import anonymize_listings_pii

logger = logging.getLogger(__name__)

# JSON field names to flatten
JSON_FIELDS = ["listing_json", "selected_bundle_json", "auto_approval_criteria_json"]

def flatten_dict_recursive(d: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    if d is None:
        return {}
    
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and v:
                nested_items = flatten_dict_recursive(v, new_key, sep=sep)
                items.extend(nested_items.items())
            elif isinstance(v, dict) and not v:
                items.append((new_key, None))
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v) if v else None))
            else:
                items.append((new_key, v))
    else:
        key = parent_key if parent_key else "value"
        items.append((key, d))
    
    return dict(items)

def flatten_json_batch(json_strings: list, prefix: str = "") -> list:
    def process_one(json_str):
        if json_str is None:
            return {}
            
        try:
            json_obj = json.loads(json_str) if isinstance(json_str, str) else json_str
            return flatten_dict_recursive(json_obj, parent_key=prefix)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON format: {e}")
            return {}
        except TypeError as e:
            logger.warning(f"Type error while flattening JSON: {e}")
            return {}
        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Error accessing JSON structure: {e}")
            return {}
        except Exception as e:
            # Catch any other unexpected errors to prevent batch failure
            logger.error(f"Unexpected error flattening JSON: {type(e).__name__}: {e}")
            return {}
    
    return [process_one(js) for js in json_strings]

def flatten_chunk(df_chunk: pl.DataFrame) -> pl.DataFrame:
    n_rows = len(df_chunk)
    
    listing_jsons = df_chunk["listing_json"].to_list()
    bundle_jsons = (
        df_chunk["selected_bundle_json"].to_list() 
        if "selected_bundle_json" in df_chunk.columns 
        else [None] * n_rows
    )
    criteria_jsons = (
        df_chunk["auto_approval_criteria_json"].to_list() 
        if "auto_approval_criteria_json" in df_chunk.columns 
        else [None] * n_rows
    )
    
    listing_flattened = flatten_json_batch(listing_jsons, prefix="listing")
    bundle_flattened = flatten_json_batch(bundle_jsons, prefix="bundle")
    criteria_flattened = flatten_json_batch(criteria_jsons, prefix="auto_approval_criteria")
    
    base_cols = [c for c in df_chunk.columns if c not in JSON_FIELDS]
    base_data = df_chunk.select(base_cols).to_dicts()
    
    combined_data = [
        {**base_data[i], **listing_flattened[i], **bundle_flattened[i], **criteria_flattened[i]}
        for i in range(n_rows)
    ]
    
    all_keys = set()
    for row in combined_data:
        all_keys.update(row.keys())
    
    base_df = df_chunk.select(base_cols)
    base_schema = {col: dtype for col, dtype in zip(base_df.columns, base_df.dtypes)}
    
    combined_data_complete = []
    for i, row in enumerate(combined_data):
        complete_row = {}
        for key in all_keys:
            if key in base_cols:
                complete_row[key] = row.get(key, base_data[i].get(key))
            else:
                complete_row[key] = row.get(key, None)
        combined_data_complete.append(complete_row)
    
    df_flat = pl.DataFrame(combined_data_complete, infer_schema_length=None)
    
    for col in base_cols:
        if col in df_flat.columns and col in base_schema:
            original_dtype = base_schema[col]
            current_dtype = df_flat[col].dtype
            if current_dtype != original_dtype:
                try:
                    df_flat = df_flat.with_columns(
                        pl.col(col).cast(original_dtype, strict=False)
                    )
                except (pl.ComputeError, pl.SchemaError) as e:
                    logger.warning(f"Could not cast {col} from {current_dtype} to {original_dtype}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error casting {col}: {type(e).__name__}: {e}")
    
    return df_flat

def process_chunk_data(df: pl.DataFrame) -> pl.DataFrame:

    logger.info("Flattening JSON fields...")
    df_flat = flatten_chunk(df)
    
    logger.info("Anonymizing PII...")
    df_anon = anonymize_listings_pii(df_flat)
    
    return df_anon
