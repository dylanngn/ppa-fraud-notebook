import polars as pl
import numpy as np
from datetime import datetime

def check_timestamps():
    try:
        df = pl.read_parquet("artifacts/nodes_listing.parquet")
        print(f"Total rows: {len(df)}")
        
        if "submission_at" in df.columns:
            ts = df["submission_at"]
            print(f"Null count: {ts.null_count()}")
            
            # Check values
            valid_ts = ts.drop_nulls()
            if len(valid_ts) > 0:
                print(f"Min timestamp: {valid_ts.min()}")
                print(f"Max timestamp: {valid_ts.max()}")
                
                # Check > 2023
                start_2023 = datetime(2023, 1, 1).timestamp() * 1e6 # us? or ns?
                # Polars timestamps are usually us or ns.
                # Let's check dtype
                print(f"Dtype: {ts.dtype}")
                
                # Count > 2023
                # Assuming ns for now if it came from Arrow/Parquet usually
                # But let's just print sample
                print("Sample timestamps:")
                print(valid_ts.head(5))
            else:
                print("All timestamps are null!")
        else:
            print("Column 'submission_at' not found!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_timestamps()
