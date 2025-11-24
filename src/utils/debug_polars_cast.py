import polars as pl
from datetime import datetime

def debug_polars_cast():
    # Create a sample dataframe with a microsecond timestamp
    data = {"ts": [datetime(2023, 1, 1, 12, 0, 0)]}
    df = pl.DataFrame(data)
    df = df.with_columns(pl.col("ts").cast(pl.Datetime("us")))

    print("Original (us):")
    print(df)
    print(df["ts"].dtype)
    print(df["ts"].cast(pl.Int64).head(1))

    # Try casting to ns
    df_ns = df.with_columns(pl.col("ts").cast(pl.Datetime("ns")).alias("ts_ns"))
    print("\nCasted to ns:")
    print(df_ns)
    print(df_ns["ts_ns"].dtype)
    print(df_ns["ts_ns"].cast(pl.Int64).head(1))

    # Check values
    val_us = df["ts"].cast(pl.Int64)[0]
    val_ns = df_ns["ts_ns"].cast(pl.Int64)[0]

    print(f"\nValue US (int64): {val_us}")
    print(f"Value NS (int64): {val_ns}")
    print(f"Ratio: {val_ns / val_us}")

if __name__ == "__main__":
    debug_polars_cast()
