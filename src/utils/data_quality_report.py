"""
Data Quality Report Generator

Analyzes the flattened insertions data to generate CSV reports for:
- Fields that are mostly null and unusable
- Top fields with highest coverage (most present)
- Phone, email, and address field coverage specifically
- Other useful validations (data types, value distributions, etc.)

All outputs are sorted by positive metrics (coverage, non-null percentage, etc.)
"""

import polars as pl
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import typer


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
    """Calculate null percentage and counts for all columns using Polars expressions.
    
    Returns DataFrame sorted by non_null_pct (highest coverage first).
    """
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
    
    return pl.DataFrame(stats).sort("non_null_pct", descending=True)


def identify_unusable_fields(null_stats: pl.DataFrame, threshold: float = 95.0) -> pl.DataFrame:
    """
    Identify fields that are mostly null (above threshold %).
    
    Args:
        null_stats: DataFrame with null statistics
        threshold: Percentage threshold (default 95% null = unusable)
    
    Returns:
        DataFrame of unusable fields, sorted by non_null_pct ascending (worst coverage first)
    """
    return (
        null_stats
        .filter(pl.col("null_pct") >= threshold)
        .sort("non_null_pct", descending=False)  # Worst coverage first
    )


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


def check_data_consistency(df: pl.DataFrame) -> pl.DataFrame:
    """
    Check for data consistency issues.
    
    Works with dot-notation field names (e.g., listing.id, listing.meta.createdAt).
    
    Returns DataFrame of consistency checks, sorted by positive metrics (coverage, etc.)
    """
    checks = []
    
    # Check for duplicate object_reference
    if "object_reference" in df.columns:
        total = len(df)
        unique = df.select("object_reference").n_unique()
        checks.append({
            "check_name": "duplicate_object_reference",
            "metric": "unique_pct",
            "value": (unique / total * 100) if total > 0 else 0,
            "total": total,
            "unique": unique,
            "duplicates": total - unique,
            "issue": total != unique
        })
    
    # Check for flattened JSON structure consistency
    # Count how many rows have flattened listing fields
    listing_id_cols = [c for c in df.columns if c.startswith("listing.id") or c == "listing.id"]
    if listing_id_cols:
        listing_id_col = listing_id_cols[0]
        total = len(df)
        has_listing_id = df.filter(pl.col(listing_id_col).is_not_null()).height
        coverage_pct = (has_listing_id / total * 100) if total > 0 else 0
        checks.append({
            "check_name": "flattened_listing_coverage",
            "metric": "coverage_pct",
            "value": coverage_pct,
            "total": total,
            "with_listing_id": has_listing_id,
            "issue": has_listing_id < total * 0.9  # Flag if <90% have listing data
        })
    
    # Check for empty strings vs nulls in key fields
    if "object_reference" in df.columns:
        total = len(df)
        empty_refs = df.filter(pl.col("object_reference").is_null() | (pl.col("object_reference") == "")).height
        valid_pct = ((total - empty_refs) / total * 100) if total > 0 else 0
        checks.append({
            "check_name": "empty_object_reference",
            "metric": "valid_pct",
            "value": valid_pct,
            "total": total,
            "empty_count": empty_refs,
            "valid_count": total - empty_refs,
            "issue": empty_refs > 0
        })
    
    # Check date consistency (works with both base columns and dot-notation)
    # Look for date/datetime fields
    date_cols = [
        c for c in df.columns 
        if ("date" in c.lower() or "at" in c.lower()) 
        and c not in ["object_reference"]  # Exclude non-date fields
    ]
    for col in date_cols[:20]:  # Limit to first 20 date columns
        if col in df.columns:
            try:
                # Check if it's actually a date/datetime column
                dtype = str(df[col].dtype)
                if "date" in dtype.lower() or "datetime" in dtype.lower():
                    min_date = df.select(pl.col(col).min()).item()
                    max_date = df.select(pl.col(col).max()).item()
                    if min_date and max_date:
                        checks.append({
                            "check_name": f"date_range_{col}",
                            "metric": "date_range",
                            "value": None,  # No single numeric metric
                            "min_date": str(min_date),
                            "max_date": str(max_date),
                            "issue": False
                        })
            except:
                pass
    
    if not checks:
        return pl.DataFrame({
            "check_name": [],
            "metric": [],
            "value": [],
            "issue": []
        })
    
    result_df = pl.DataFrame(checks)
    # Sort by value (positive metric) descending, with None values last
    return result_df.sort(
        by="value",
        descending=True,
        nulls_last=True
    )


def generate_text_report(
    df: pl.DataFrame,
    null_stats: pl.DataFrame,
    unusable: pl.DataFrame,
    high_coverage: pl.DataFrame,
    contact_analysis: Dict[str, pl.DataFrame],
    type_analysis: pl.DataFrame,
    numeric_analysis: pl.DataFrame,
    categorical_analysis: Dict[str, pl.DataFrame],
    consistency_checks: pl.DataFrame,
    data_path: str,
    output_path: Path,
    null_threshold_unusable: float,
    null_threshold_high_coverage: float
) -> None:
    """
    Generate comprehensive text report with all details.
    
    Args:
        df: Original DataFrame
        null_stats: Null statistics for all fields
        unusable: Unusable fields DataFrame
        high_coverage: High coverage fields DataFrame (ALL fields, not limited)
        contact_analysis: Dictionary with phone, email, address analyses
        type_analysis: Data type distribution
        numeric_analysis: Numeric field statistics
        categorical_analysis: Categorical field value distributions
        consistency_checks: Consistency check results
        data_path: Path to source data
        output_path: Path to save text report
        null_threshold_unusable: Threshold for unusable fields
        null_threshold_high_coverage: Threshold for high coverage fields
    """
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
        report_lines.append(format_dataframe_table(unusable))
    else:
        report_lines.append("\nNo fields found above the unusable threshold.")
    
    # High coverage fields - LIST ALL, NOT JUST TOP ONES
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"HIGH COVERAGE FIELDS (<{null_threshold_high_coverage}% NULL)")
    report_lines.append("=" * 80)
    if len(high_coverage) > 0:
        report_lines.append(f"\nAll {len(high_coverage)} fields with high coverage (sorted by coverage, highest first):")
        report_lines.append(format_dataframe_table(high_coverage))  # No limit - show all
    else:
        report_lines.append("\nNo high coverage fields found.")
    
    # Phone fields analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("PHONE FIELDS ANALYSIS")
    report_lines.append("=" * 80)
    if len(contact_analysis["phones"]) > 0:
        report_lines.append(f"\nFound {len(contact_analysis['phones'])} phone fields:")
        report_lines.append(format_dataframe_table(contact_analysis["phones"]))
    else:
        report_lines.append("\nNo phone fields found.")
    
    # Email fields analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("EMAIL FIELDS ANALYSIS")
    report_lines.append("=" * 80)
    if len(contact_analysis["emails"]) > 0:
        report_lines.append(f"\nFound {len(contact_analysis['emails'])} email fields:")
        report_lines.append(format_dataframe_table(contact_analysis["emails"]))
    else:
        report_lines.append("\nNo email fields found.")
    
    # Address fields analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("ADDRESS FIELDS ANALYSIS")
    report_lines.append("=" * 80)
    if len(contact_analysis["addresses"]) > 0:
        report_lines.append(f"\nFound {len(contact_analysis['addresses'])} address fields:")
        report_lines.append(format_dataframe_table(contact_analysis["addresses"]))
    else:
        report_lines.append("\nNo address fields found.")
    
    # Data type distribution
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATA TYPE DISTRIBUTION")
    report_lines.append("=" * 80)
    report_lines.append(format_dataframe_table(type_analysis))
    
    # Numeric fields summary
    if len(numeric_analysis) > 0:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("NUMERIC FIELDS SUMMARY (Sorted by Coverage)")
        report_lines.append("=" * 80)
        report_lines.append(format_dataframe_table(numeric_analysis))  # Show all numeric fields
    
    # Categorical fields summary (top values)
    if categorical_analysis:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("CATEGORICAL FIELDS - TOP VALUES")
        report_lines.append("=" * 80)
        for col, value_counts in categorical_analysis.items():
            report_lines.append(f"\n{col}:")
            report_lines.append(format_dataframe_table(value_counts))
    
    # Data consistency checks
    report_lines.append("\n" + "=" * 80)
    report_lines.append("DATA CONSISTENCY CHECKS")
    report_lines.append("=" * 80)
    if len(consistency_checks) > 0:
        report_lines.append(format_dataframe_table(consistency_checks))
    else:
        report_lines.append("\nNo consistency checks performed.")
    
    # Write report
    report_text = "\n".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)


def generate_report(
    data_path: str = "artifacts/raw_insertions.parquet",
    output_dir: str = "artifacts/data_quality_reports",
    null_threshold_unusable: float = 95.0,
    null_threshold_high_coverage: float = 50.0
) -> None:
    """
    Generate comprehensive data quality reports as CSV files.
    
    Args:
        data_path: Path to flattened insertions parquet file
        output_dir: Directory to save CSV reports
        null_threshold_unusable: Fields with >=this % null are considered unusable
        null_threshold_high_coverage: Fields with <this % null are considered high coverage
    """
    print("=" * 80)
    print("DATA QUALITY REPORT GENERATOR")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_flattened_data(data_path)
    print(f"\nLoaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Calculate null statistics
    print("\nCalculating null statistics...")
    null_stats = calculate_null_stats(df)
    
    # Save all null statistics (sorted by coverage)
    null_stats_path = output_path / "01_all_fields_null_stats.csv"
    null_stats.write_csv(null_stats_path)
    print(f"  ✓ Saved: {null_stats_path} ({len(null_stats)} fields)")
    
    # Identify unusable fields
    unusable = identify_unusable_fields(null_stats, threshold=null_threshold_unusable)
    if len(unusable) > 0:
        unusable_path = output_path / "02_unusable_fields.csv"
        unusable.write_csv(unusable_path)
        print(f"  ✓ Saved: {unusable_path} ({len(unusable)} fields)")
    else:
        print(f"  - No unusable fields found (threshold: {null_threshold_unusable}% null)")
    
    # Identify high coverage fields
    high_coverage = identify_high_coverage_fields(null_stats, threshold=null_threshold_high_coverage)
    if len(high_coverage) > 0:
        high_coverage_path = output_path / "03_high_coverage_fields.csv"
        high_coverage.write_csv(high_coverage_path)
        print(f"  ✓ Saved: {high_coverage_path} ({len(high_coverage)} fields)")
    else:
        print(f"  - No high coverage fields found (threshold: <{null_threshold_high_coverage}% null)")
    
    # Analyze contact fields
    print("\nAnalyzing contact fields (phones, emails, addresses)...")
    contact_analysis = analyze_contact_fields(df)
    
    if len(contact_analysis["phones"]) > 0:
        phones_path = output_path / "04_phone_fields.csv"
        contact_analysis["phones"].write_csv(phones_path)
        print(f"  ✓ Saved: {phones_path} ({len(contact_analysis['phones'])} fields)")
    
    if len(contact_analysis["emails"]) > 0:
        emails_path = output_path / "05_email_fields.csv"
        contact_analysis["emails"].write_csv(emails_path)
        print(f"  ✓ Saved: {emails_path} ({len(contact_analysis['emails'])} fields)")
    
    if len(contact_analysis["addresses"]) > 0:
        addresses_path = output_path / "06_address_fields.csv"
        contact_analysis["addresses"].write_csv(addresses_path)
        print(f"  ✓ Saved: {addresses_path} ({len(contact_analysis['addresses'])} fields)")
    
    # Analyze data types
    print("\nAnalyzing data types...")
    type_analysis = analyze_data_types(df)
    type_path = output_path / "07_data_type_distribution.csv"
    type_analysis.write_csv(type_path)
    print(f"  ✓ Saved: {type_path} ({len(type_analysis)} types)")
    
    # Analyze numeric fields
    print("\nAnalyzing numeric fields...")
    numeric_analysis = analyze_numeric_fields(df, null_stats)
    if len(numeric_analysis) > 0:
        numeric_path = output_path / "08_numeric_fields.csv"
        numeric_analysis.write_csv(numeric_path)
        print(f"  ✓ Saved: {numeric_path} ({len(numeric_analysis)} fields)")
    else:
        print(f"  - No numeric fields found")
    
    # Analyze categorical fields
    print("\nAnalyzing categorical fields...")
    categorical_analysis = analyze_categorical_fields(df, null_stats, top_n=10)
    if categorical_analysis:
        categorical_dir = output_path / "09_categorical_fields"
        categorical_dir.mkdir(exist_ok=True)
        for col, value_counts in categorical_analysis.items():
            # Sanitize column name for filename
            safe_col_name = col.replace("/", "_").replace("\\", "_").replace(".", "_")
            cat_path = categorical_dir / f"{safe_col_name}_top_values.csv"
            value_counts.write_csv(cat_path)
        print(f"  ✓ Saved: {len(categorical_analysis)} categorical field analyses to {categorical_dir}/")
    else:
        print(f"  - No categorical fields analyzed")
    
    # Check data consistency
    print("\nChecking data consistency...")
    consistency_checks = check_data_consistency(df)
    if len(consistency_checks) > 0:
        consistency_path = output_path / "10_data_consistency_checks.csv"
        consistency_checks.write_csv(consistency_path)
        print(f"  ✓ Saved: {consistency_path} ({len(consistency_checks)} checks)")
    else:
        print(f"  - No consistency checks performed")
    
    # Generate comprehensive text report
    print("\nGenerating text report...")
    text_report_path = output_path / "data_quality_report.txt"
    generate_text_report(
        df=df,
        null_stats=null_stats,
        unusable=unusable,
        high_coverage=high_coverage,
        contact_analysis=contact_analysis,
        type_analysis=type_analysis,
        numeric_analysis=numeric_analysis,
        categorical_analysis=categorical_analysis,
        consistency_checks=consistency_checks,
        data_path=data_path,
        output_path=text_report_path,
        null_threshold_unusable=null_threshold_unusable,
        null_threshold_high_coverage=null_threshold_high_coverage
    )
    print(f"  ✓ Saved: {text_report_path}")
    
    # Print summary to console
    print(f"\n{'=' * 80}")
    print("SUMMARY:")
    print(f"  - Total fields analyzed: {len(null_stats)}")
    print(f"  - Unusable fields (>={null_threshold_unusable}% null): {len(unusable)}")
    print(f"  - High coverage fields (<{null_threshold_high_coverage}% null): {len(high_coverage)}")
    print(f"  - Phone fields: {len(contact_analysis['phones'])}")
    print(f"  - Email fields: {len(contact_analysis['emails'])}")
    print(f"  - Address fields: {len(contact_analysis['addresses'])}")
    print(f"  - Numeric fields: {len(numeric_analysis)}")
    print(f"  - Categorical fields analyzed: {len(categorical_analysis)}")
    print(f"  - Consistency checks: {len(consistency_checks)}")
    
    if len(contact_analysis['phones']) > 0:
        top_phone = contact_analysis['phones'].head(1)
        print(f"\n  Top phone field: {top_phone['field'][0]} ({top_phone['non_null_pct'][0]:.1f}% coverage)")
    
    if len(contact_analysis['emails']) > 0:
        top_email = contact_analysis['emails'].head(1)
        print(f"  Top email field: {top_email['field'][0]} ({top_email['non_null_pct'][0]:.1f}% coverage)")
    
    if len(contact_analysis['addresses']) > 0:
        top_addr = contact_analysis['addresses'].head(1)
        print(f"  Top address field: {top_addr['field'][0]} ({top_addr['non_null_pct'][0]:.1f}% coverage)")
    
    print(f"\n{'=' * 80}")
    print(f"All reports saved to: {output_path}")
    print(f"{'=' * 80}")


def main(
    data_path: str = typer.Option("artifacts/raw_insertions.parquet", help="Path to flattened & anonymized insertions parquet file"),
    output_dir: str = typer.Option("artifacts/data_quality_reports", help="Directory to save CSV reports"),
    null_threshold_unusable: float = typer.Option(95.0, help="Fields with >= this % null are considered unusable"),
    null_threshold_high_coverage: float = typer.Option(50.0, help="Fields with < this % null are considered high coverage")
):
    """
    Generate comprehensive data quality reports as CSV files.
    """
    generate_report(
        data_path=data_path,
        output_dir=output_dir,
        null_threshold_unusable=null_threshold_unusable,
        null_threshold_high_coverage=null_threshold_high_coverage
    )


if __name__ == "__main__":
    import typer
    typer.run(main)

