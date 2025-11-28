"""
Data Quality Report Generator

Analyzes the flattened insertions data to generate a comprehensive report on:
- Fields that are mostly null and unusable
- Top fields with highest coverage (most present)
- Phone, email, and address field coverage specifically
- Other useful validations (data types, value distributions, etc.)
"""

import polars as pl
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def format_dataframe_table(df: pl.DataFrame, max_rows: int = None) -> str:
    """
    Format a Polars DataFrame as a readable table string using Polars only.
    
    Args:
        df: Polars DataFrame to format
        max_rows: Maximum number of rows to display (None for all)
    
    Returns:
        Formatted string representation of the DataFrame
    """
    if len(df) == 0:
        return "(empty DataFrame)"
    
    # Limit rows if specified
    df_display = df.head(max_rows) if max_rows else df
    
    # Use Polars' built-in string representation which produces nice table output
    # This avoids pandas dependency entirely
    return str(df_display)


def load_flattened_data(data_path: str = "artifacts/raw_insertions.parquet") -> pl.DataFrame:
    """
    Load the flattened and anonymized insertions data.
    
    The ETL now flattens and anonymizes before saving to raw_insertions.parquet,
    so this file contains the fully flattened data.
    """
    if os.path.exists(data_path):
        print(f"Loading flattened & anonymized data from {data_path}...")
        return pl.read_parquet(data_path)
    else:
        raise FileNotFoundError(
            f"{data_path} not found. Please run ETL first to generate flattened data."
        )


def calculate_null_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate null percentage and counts for all columns using Polars expressions."""
    n_rows = len(df)
    
    if n_rows == 0:
        return pl.DataFrame({
            "column": [],
            "null_count": [],
            "null_pct": [],
            "non_null_count": [],
            "non_null_pct": [],
            "dtype": []
        })
    
    # Use Polars expressions to calculate null stats for all columns at once
    stats = []
    for col in df.columns:
        col_dtype = df[col].dtype
        null_count = df.select(pl.col(col).is_null().sum()).item()
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0.0
        non_null_count = n_rows - null_count
        non_null_pct = 100.0 - null_pct
        
        stats.append({
            "column": col,
            "null_count": null_count,
            "null_pct": null_pct,
            "non_null_count": non_null_count,
            "non_null_pct": non_null_pct,
            "dtype": str(col_dtype)
        })
    
    return pl.DataFrame(stats).sort("null_pct", descending=True)


def identify_unusable_fields(null_stats: pl.DataFrame, threshold: float = 95.0) -> pl.DataFrame:
    """
    Identify fields that are mostly null (above threshold %).
    
    Args:
        null_stats: DataFrame with null statistics
        threshold: Percentage threshold (default 95% null = unusable)
    
    Returns:
        DataFrame of unusable fields
    """
    return null_stats.filter(pl.col("null_pct") >= threshold)


def identify_high_coverage_fields(null_stats: pl.DataFrame, threshold: float = 50.0) -> pl.DataFrame:
    """
    Identify fields with high coverage (below threshold % null).
    
    Args:
        null_stats: DataFrame with null statistics
        threshold: Maximum null percentage (default <50% null = high coverage)
    
    Returns:
        DataFrame of high coverage fields, sorted by coverage
    """
    return (
        null_stats
        .filter(pl.col("null_pct") < threshold)
        .sort("non_null_pct", descending=True)
    )


def analyze_contact_fields(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """
    Analyze phone, email, and address fields specifically.
    
    Works with dot-notation field names (e.g., listing.lister.email, listing.lister.phone).
    Includes both original fields and hash fields created by anonymization.
    
    Returns:
        Dictionary with 'phones', 'emails', 'addresses' keys containing stats
    """
    all_cols = df.columns
    
    # Identify phone fields (including dot-notation like listing.lister.phone)
    # Exclude fields that are clearly not phone fields (e.g., "phoneType" might be a category)
    phone_fields = [
        c for c in all_cols 
        if "phone" in c.lower() and "phonetype" not in c.lower()
    ]
    
    # Identify email fields (including dot-notation like listing.lister.email)
    email_fields = [c for c in all_cols if "email" in c.lower()]
    
    # Identify address fields (including dot-notation like listing.address.street)
    # Look for address-related terms in the field name
    address_fields = [
        c for c in all_cols if any(
            term in c.lower() for term in [
                "address", "street", "zip", "postal", "city", "country", 
                "locality", "postalcode"
            ]
        )
        # Exclude fields that are clearly not address fields
        and "address_hash" not in c.lower()  # We want the component fields, not just the hash
    ]
    
    def get_field_stats(fields: List[str]) -> pl.DataFrame:
        if not fields:
            return pl.DataFrame({
                "field": [],
                "null_count": [],
                "null_pct": [],
                "non_null_count": [],
                "non_null_pct": [],
                "unique_count": [],
                "dtype": []
            })
        
        n_rows = len(df)
        stats = []
        for field in fields:
            if field not in df.columns:
                continue
            
            null_count = df.select(pl.col(field).is_null().sum()).item()
            null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0
            non_null_count = n_rows - null_count
            unique_count = df.select(pl.col(field).n_unique()).item()
            
            stats.append({
                "field": field,
                "null_count": null_count,
                "null_pct": null_pct,
                "non_null_count": non_null_count,
                "non_null_pct": 100 - null_pct,
                "unique_count": unique_count,
                "dtype": str(df[field].dtype)
            })
        
        return (
            pl.DataFrame(stats)
            .sort("non_null_pct", descending=True)
        )
    
    return {
        "phones": get_field_stats(phone_fields),
        "emails": get_field_stats(email_fields),
        "addresses": get_field_stats(address_fields)
    }


def analyze_data_types(df: pl.DataFrame) -> pl.DataFrame:
    """Analyze data type distribution across columns."""
    type_counts = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
    
    return pl.DataFrame({
        "dtype": list(type_counts.keys()),
        "count": list(type_counts.values())
    }).sort("count", descending=True)


def analyze_numeric_fields(df: pl.DataFrame, null_stats: pl.DataFrame) -> pl.DataFrame:
    """Analyze numeric fields with basic statistics."""
    numeric_cols = [
        col for col in df.columns 
        if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                             pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                             pl.Float32, pl.Float64]
    ]
    
    if not numeric_cols:
        return pl.DataFrame()
    
    stats = []
    for col in numeric_cols:
        col_stats = df.select([
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).mean().alias("mean"),
            pl.col(col).median().alias("median"),
            pl.col(col).std().alias("std"),
            pl.col(col).n_unique().alias("unique_count")
        ]).row(0)
        
        null_info = null_stats.filter(pl.col("column") == col).row(0)
        
        stats.append({
            "column": col,
            "null_pct": null_info[1],
            "non_null_pct": null_info[3],
            "min": col_stats[0],
            "max": col_stats[1],
            "mean": col_stats[2],
            "median": col_stats[3],
            "std": col_stats[4],
            "unique_count": col_stats[5],
            "dtype": str(df[col].dtype)
        })
    
    return pl.DataFrame(stats).sort("non_null_pct", descending=True)


def analyze_categorical_fields(df: pl.DataFrame, null_stats: pl.DataFrame, top_n: int = 10) -> Dict[str, pl.DataFrame]:
    """
    Analyze categorical/string fields with value distributions.
    
    Returns top N most common values for each categorical field.
    """
    string_cols = [
        col for col in df.columns 
        if df[col].dtype == pl.Utf8
    ]
    
    categorical_stats = {}
    
    for col in string_cols[:20]:  # Limit to first 20 to avoid too much output
        null_info = null_stats.filter(pl.col("column") == col).row(0)
        null_pct = null_info[1]
        
        # Skip if mostly null
        if null_pct > 90:
            continue
        
        # Get value counts
        # In Polars, value_counts() works on a Series, so we get the column first
        value_counts = (
            df
            .filter(pl.col(col).is_not_null())
            .get_column(col)
            .value_counts()
            .head(top_n)
            .sort("count", descending=True)
        )
        
        if len(value_counts) > 0:
            categorical_stats[col] = value_counts
    
    return categorical_stats


def check_data_consistency(df: pl.DataFrame) -> Dict[str, any]:
    """
    Check for data consistency issues.
    
    Works with dot-notation field names (e.g., listing.id, listing.meta.createdAt).
    
    Returns dictionary of consistency checks.
    """
    checks = {}
    
    # Check for duplicate object_reference
    if "object_reference" in df.columns:
        total = len(df)
        unique = df.select("object_reference").n_unique()
        checks["duplicate_object_reference"] = {
            "total": total,
            "unique": unique,
            "duplicates": total - unique,
            "issue": total != unique
        }
    
    # Check date consistency (works with both base columns and dot-notation)
    # Look for date/datetime fields
    date_cols = [
        c for c in df.columns 
        if ("date" in c.lower() or "at" in c.lower()) 
        and c not in ["object_reference"]  # Exclude non-date fields
    ]
    date_checks = {}
    for col in date_cols[:10]:  # Limit to first 10 date columns
        if col in df.columns:
            try:
                # Check if it's actually a date/datetime column
                dtype = str(df[col].dtype)
                if "date" in dtype.lower() or "datetime" in dtype.lower():
                    min_date = df.select(pl.col(col).min()).item()
                    max_date = df.select(pl.col(col).max()).item()
                    date_checks[col] = {
                        "min": str(min_date) if min_date else None,
                        "max": str(max_date) if max_date else None
                    }
            except:
                pass
    checks["date_ranges"] = date_checks
    
    # Check for flattened JSON structure consistency
    # Count how many rows have flattened listing fields
    listing_id_cols = [c for c in df.columns if c.startswith("listing.id") or c == "listing.id"]
    if listing_id_cols:
        listing_id_col = listing_id_cols[0]
        total = len(df)
        has_listing_id = df.filter(pl.col(listing_id_col).is_not_null()).height
        checks["flattened_listing_coverage"] = {
            "total": total,
            "with_listing_id": has_listing_id,
            "coverage_pct": (has_listing_id / total * 100) if total > 0 else 0,
            "issue": has_listing_id < total * 0.9  # Flag if <90% have listing data
        }
    
    # Check for empty strings vs nulls in key fields
    if "object_reference" in df.columns:
        empty_refs = df.filter(pl.col("object_reference").is_null() | (pl.col("object_reference") == "")).height
        checks["empty_object_reference"] = {
            "count": empty_refs,
            "issue": empty_refs > 0
        }
    
    return checks


def generate_report(
    data_path: str = "artifacts/raw_insertions.parquet",
    output_path: str = "artifacts/data_quality_report.txt",
    null_threshold_unusable: float = 95.0,
    null_threshold_high_coverage: float = 50.0
) -> None:
    """
    Generate comprehensive data quality report.
    
    Args:
        data_path: Path to flattened insertions parquet file
        output_path: Path to save the report
        null_threshold_unusable: Fields with >this % null are considered unusable
        null_threshold_high_coverage: Fields with <this % null are considered high coverage
    """
    print("=" * 80)
    print("DATA QUALITY REPORT GENERATOR")
    print("=" * 80)
    
    # Load data
    df = load_flattened_data(data_path)
    print(f"\nLoaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Calculate null statistics
    print("\nCalculating null statistics...")
    null_stats = calculate_null_stats(df)
    
    # Identify unusable fields
    unusable = identify_unusable_fields(null_stats, threshold=null_threshold_unusable)
    
    # Identify high coverage fields
    high_coverage = identify_high_coverage_fields(null_stats, threshold=null_threshold_high_coverage)
    
    # Analyze contact fields
    print("Analyzing contact fields (phones, emails, addresses)...")
    contact_analysis = analyze_contact_fields(df)
    
    # Analyze data types
    type_analysis = analyze_data_types(df)
    
    # Analyze numeric fields
    print("Analyzing numeric fields...")
    numeric_analysis = analyze_numeric_fields(df, null_stats)
    
    # Analyze categorical fields (limited to avoid too much output)
    print("Analyzing categorical fields...")
    categorical_analysis = analyze_categorical_fields(df, null_stats, top_n=5)
    
    # Check data consistency
    print("Checking data consistency...")
    consistency_checks = check_data_consistency(df)
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA QUALITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nDataset: {data_path}")
    report_lines.append(f"Total Rows: {len(df):,}")
    report_lines.append(f"Total Columns: {len(df.columns)}")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary statistics
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"\nUnusable Fields (>={null_threshold_unusable}% null): {len(unusable)}")
    report_lines.append(f"High Coverage Fields (<{null_threshold_high_coverage}% null): {len(high_coverage)}")
    report_lines.append(f"Total Fields Analyzed: {len(null_stats)}")
    
    # Unusable fields
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"UNUSABLE FIELDS (>{null_threshold_unusable}% NULL)")
    report_lines.append("=" * 80)
    if len(unusable) > 0:
        report_lines.append(f"\nFound {len(unusable)} fields that are mostly null:")
        # Format using Polars
        report_lines.append(format_dataframe_table(unusable.head(50)))
        if len(unusable) > 50:
            report_lines.append(f"\n... and {len(unusable) - 50} more fields")
    else:
        report_lines.append("\nNo fields found above the unusable threshold.")
    
    # High coverage fields
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"HIGH COVERAGE FIELDS (<{null_threshold_high_coverage}% NULL)")
    report_lines.append("=" * 80)
    if len(high_coverage) > 0:
        report_lines.append(f"\nTop {min(30, len(high_coverage))} fields with highest coverage:")
        # Format using Polars
        report_lines.append(format_dataframe_table(high_coverage.head(30)))
    else:
        report_lines.append("\nNo high coverage fields found.")
    
    # Phone fields analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("PHONE FIELDS ANALYSIS")
    report_lines.append("=" * 80)
    if len(contact_analysis["phones"]) > 0:
        report_lines.append(f"\nFound {len(contact_analysis['phones'])} phone fields:")
        # Format using Polars
        report_lines.append(format_dataframe_table(contact_analysis["phones"]))
    else:
        report_lines.append("\nNo phone fields found.")
    
    # Email fields analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("EMAIL FIELDS ANALYSIS")
    report_lines.append("=" * 80)
    if len(contact_analysis["emails"]) > 0:
        report_lines.append(f"\nFound {len(contact_analysis['emails'])} email fields:")
        # Format using Polars
        report_lines.append(format_dataframe_table(contact_analysis["emails"]))
    else:
        report_lines.append("\nNo email fields found.")
    
    # Address fields analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ADDRESS FIELDS ANALYSIS")
    report_lines.append("=" * 80)
    if len(contact_analysis["addresses"]) > 0:
        report_lines.append(f"\nFound {len(contact_analysis['addresses'])} address fields:")
        # Format using Polars
        report_lines.append(format_dataframe_table(contact_analysis["addresses"]))
    else:
        report_lines.append("\nNo address fields found.")
    
    # Data type distribution
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATA TYPE DISTRIBUTION")
    report_lines.append("=" * 80)
    # Format using Polars
    report_lines.append(format_dataframe_table(type_analysis))
    
    # Numeric fields summary
    if len(numeric_analysis) > 0:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("NUMERIC FIELDS SUMMARY (Top 20 by Coverage)")
        report_lines.append("=" * 80)
        # Format using Polars
        report_lines.append(format_dataframe_table(numeric_analysis.head(20)))
    
    # Categorical fields summary (top values)
    if categorical_analysis:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("CATEGORICAL FIELDS - TOP VALUES")
        report_lines.append("=" * 80)
        for col, value_counts in list(categorical_analysis.items())[:10]:
            report_lines.append(f"\n{col}:")
            # Format using Polars
            report_lines.append(format_dataframe_table(value_counts))
    
    # Data consistency checks
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATA CONSISTENCY CHECKS")
    report_lines.append("=" * 80)
    for check_name, check_result in consistency_checks.items():
        report_lines.append(f"\n{check_name}:")
        if isinstance(check_result, dict):
            for key, value in check_result.items():
                report_lines.append(f"  {key}: {value}")
        else:
            report_lines.append(f"  {check_result}")
    
    # Write report
    report_text = "\n".join(report_lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    
    print(f"\n{'=' * 80}")
    print(f"Report saved to: {output_path}")
    print(f"{'=' * 80}")
    
    # Print summary to console
    print("\nSUMMARY:")
    print(f"  - Unusable fields (>={null_threshold_unusable}% null): {len(unusable)}")
    print(f"  - High coverage fields (<{null_threshold_high_coverage}% null): {len(high_coverage)}")
    print(f"  - Phone fields: {len(contact_analysis['phones'])}")
    print(f"  - Email fields: {len(contact_analysis['emails'])}")
    print(f"  - Address fields: {len(contact_analysis['addresses'])}")
    
    if len(contact_analysis['phones']) > 0:
        top_phone = contact_analysis['phones'].head(1)
        print(f"  - Top phone field: {top_phone['field'][0]} ({top_phone['non_null_pct'][0]:.1f}% coverage)")
    
    if len(contact_analysis['emails']) > 0:
        top_email = contact_analysis['emails'].head(1)
        print(f"  - Top email field: {top_email['field'][0]} ({top_email['non_null_pct'][0]:.1f}% coverage)")
    
    if len(contact_analysis['addresses']) > 0:
        top_addr = contact_analysis['addresses'].head(1)
        print(f"  - Top address field: {top_addr['field'][0]} ({top_addr['non_null_pct'][0]:.1f}% coverage)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate data quality report")
    parser.add_argument(
        "--data-path",
        type=str,
        default="artifacts/raw_insertions.parquet",
        help="Path to flattened & anonymized insertions parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/data_quality_report.txt",
        help="Path to save the report"
    )
    parser.add_argument(
        "--null-threshold-unusable",
        type=float,
        default=95.0,
        help="Fields with >= this % null are considered unusable"
    )
    parser.add_argument(
        "--null-threshold-high-coverage",
        type=float,
        default=50.0,
        help="Fields with < this % null are considered high coverage"
    )
    
    args = parser.parse_args()
    
    generate_report(
        data_path=args.data_path,
        output_path=args.output,
        null_threshold_unusable=args.null_threshold_unusable,
        null_threshold_high_coverage=args.null_threshold_high_coverage
    )

