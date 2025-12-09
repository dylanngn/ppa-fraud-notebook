"""
Data Schema Definitions using Pandera

Defines expected schema for ETL outputs and feature DataFrames.
Used for validation in data quality reports and production serving.

Note: This is for TABULAR data schema, not graph schema.
For graph structure, see src/data/graph/schema.py
"""
import pandera.polars as pa
from pandera.polars import Column
import polars as pl
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RAW INSERTIONS SCHEMA (ETL Output)
# =============================================================================

class RawInsertionsSchema(pa.DataFrameModel):
    """
    Schema for raw_insertions.parquet - the ETL output.
    
    This validates the core fields required for model training.
    Optional fields are allowed (coerce handles missing columns gracefully).
    """
    
    # === Required Identifiers ===
    object_reference: str = pa.Field(nullable=False, unique=True, description="Listing business ID")
    
    # === Required Timestamps ===
    submission_at: pl.Datetime = pa.Field(nullable=False, description="Listing submission timestamp")
    
    # === Required Target ===
    is_fraud: bool = pa.Field(nullable=False, description="Fraud label (target variable)")
    
    # === Core Features (High Coverage) ===
    account_created_at: Optional[pl.Datetime] = pa.Field(nullable=True, description="Account creation timestamp")
    
    class Config:
        strict = False
        coerce = True
        name = "RawInsertionsSchema"


# =============================================================================
# FEATURE SCHEMA (Training Input)
# =============================================================================

def create_feature_schema(include_graph: bool = True):
    """
    Create a dynamic feature schema based on enabled feature groups.
    
    Args:
        include_graph: Include graph-derived features
        
    Returns:
        Pandera DataFrameSchema for validation
    """
    from src.models.config.constants import GRAPH_FEATURES, CATEGORICAL_FEATURES
    
    # Build column definitions dynamically
    columns = {
        # Always required
        "insertion_id": Column(str, nullable=False),
        "is_fraud": Column(bool, nullable=False),
    }
    
    # Categorical features
    for feat in CATEGORICAL_FEATURES:
        columns[feat] = Column(str, nullable=True)
    
    # Graph features
    if include_graph:
        for feat in GRAPH_FEATURES:
            columns[feat] = Column(float, nullable=True)
    
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_raw_insertions(df: pl.DataFrame, raise_on_error: bool = False) -> dict:
    """
    Validate raw insertions DataFrame against schema.
    
    Args:
        df: Polars DataFrame to validate
        raise_on_error: If True, raise exception on validation failure
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }
    }
    
    # Check required columns
    required_cols = ["object_reference", "submission_at", "is_fraud"]
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        result["valid"] = False
        result["errors"].append(f"Missing required columns: {missing_required}")
        if raise_on_error:
            raise ValueError(f"Schema validation failed: Missing required columns {missing_required}")
        return result
    
    # Check for duplicates in object_reference
    unique_count = df.select("object_reference").n_unique()
    if unique_count != len(df):
        duplicates = len(df) - unique_count
        result["warnings"].append(f"Found {duplicates} duplicate object_reference values")
    
    # Check for nulls in required columns
    for col in required_cols:
        null_count = df.select(pl.col(col).is_null().sum()).item()
        if null_count > 0:
            result["valid"] = False
            result["errors"].append(f"Column '{col}' has {null_count} null values (should be 0)")
    
    # Check fraud rate is reasonable
    fraud_rate = df.select(pl.col("is_fraud").mean()).item()
    result["stats"]["fraud_rate"] = fraud_rate
    if fraud_rate < 0.001:
        result["warnings"].append(f"Fraud rate is very low ({fraud_rate:.4%})")
    elif fraud_rate > 0.5:
        result["warnings"].append(f"Fraud rate is very high ({fraud_rate:.4%})")
    
    # Check date range
    if "submission_at" in df.columns:
        min_date = df.select(pl.col("submission_at").min()).item()
        max_date = df.select(pl.col("submission_at").max()).item()
        result["stats"]["date_range"] = {"min": str(min_date), "max": str(max_date)}
    
    if raise_on_error and not result["valid"]:
        raise ValueError(f"Schema validation failed: {result['errors']}")
    
    logger.info(f"Schema validation: {'PASSED' if result['valid'] else 'FAILED'}")
    for warning in result["warnings"]:
        logger.warning(warning)
    
    return result


def validate_features(df: pl.DataFrame, expected_features: list[str]) -> dict:
    """
    Validate that a feature DataFrame contains expected features.
    
    Args:
        df: Polars DataFrame with features
        expected_features: List of expected feature column names
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "missing_features": [],
        "extra_columns": [],
        "feature_count": 0,
    }
    
    available_cols = set(df.columns)
    expected_set = set(expected_features)
    
    result["missing_features"] = list(expected_set - available_cols)
    result["extra_columns"] = list(available_cols - expected_set - {"insertion_id", "is_fraud", "submission_at"})
    result["feature_count"] = len(expected_set & available_cols)
    
    if result["missing_features"]:
        result["valid"] = False
        logger.warning(f"Missing features: {result['missing_features']}")
    
    return result
