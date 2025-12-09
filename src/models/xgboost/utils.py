"""
XGBoost training utilities.

SIMPLIFIED (2025-12-09 after Exp 9):
- Removed complex FEATURE_MANIFEST (auto mode doesn't need it)
- Kept utility functions for categorical features, constraints
"""
import subprocess
import logging
import pandas as pd
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE VALIDATION (Simplified for auto mode)
# =============================================================================

def validate_features(df: pd.DataFrame, strict: bool = False) -> Tuple[List[str], List[str]]:
    """
    Validate DataFrame columns for XGBoost compatibility.
    
    In auto mode, we accept all numeric/categorical columns.
    
    Args:
        df: DataFrame to validate
        strict: If True, raise error on issues (not used in auto mode)
        
    Returns:
        Tuple of (valid_features, invalid_columns)
    """
    valid_features = []
    invalid_columns = []
    
    # ID/metadata columns to exclude
    exclude_cols = {
        "insertion_id", "object_reference", "is_fraud", 
        "submission_at", "fraud_flag", "owner_id", "user_id"
    }
    
    for col in df.columns:
        if col in exclude_cols:
            invalid_columns.append(col)
            continue
            
        dtype = df[col].dtype
        
        # Accept numeric, boolean, categorical, and object (will be encoded)
        is_valid = (
            pd.api.types.is_numeric_dtype(dtype) or
            pd.api.types.is_bool_dtype(dtype) or
            pd.api.types.is_categorical_dtype(dtype) or
            pd.api.types.is_object_dtype(dtype) or
            pd.api.types.is_string_dtype(dtype)
        )
        
        if is_valid:
            valid_features.append(col)
        else:
            invalid_columns.append(col)
            logger.debug(f"Excluding column '{col}' with unsupported dtype: {dtype}")
    
    logger.info(f"Validated features: {len(valid_features)} valid, {len(invalid_columns)} excluded")
    return valid_features, invalid_columns


def get_optimal_tree_method() -> str:
    """
    Detect optimal tree method based on available hardware.
    
    Note: XGBoost GPU support (gpu_hist) only works with NVIDIA CUDA GPUs.
    Apple Silicon (M1/M2/M3/M4) GPUs are NOT supported by XGBoost.
    
    Returns:
        Tree method string: "gpu_hist" if NVIDIA GPU available, "hist" otherwise
    """
    # Try to detect NVIDIA GPU availability (XGBoost only supports CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu_hist"
    except ImportError:
        pass
    
    # Check nvidia-smi (if available)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return "gpu_hist"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return "hist"


def get_categorical_features(features: List[str]) -> List[str]:
    """Identify categorical features that should use XGBoost's native categorical support."""
    from src.models.config.constants import CATEGORICAL_FEATURES
    
    categorical_features = []
    for feat in CATEGORICAL_FEATURES:
        if feat in features:
            categorical_features.append(feat)
    
    return categorical_features


def build_monotonic_constraints(features: List[str]) -> Dict[str, int]:
    """
    Build monotonic constraints dictionary based on domain knowledge.
    
    Returns:
        Dictionary mapping feature name to constraint (-1, 0, or 1)
    """
    constraints = {}
    
    # Account age: fraud decreases as account age increases
    if "account_age_days" in features:
        constraints["account_age_days"] = -1
    
    return constraints


def build_feature_interaction_constraints(features: List[str]) -> List[List[str]]:
    """
    Build feature interaction constraints based on domain knowledge.
    
    Returns:
        List of feature name groups that should interact
    """
    constraints = []
    
    def has_feat(feat_name: str) -> bool:
        return feat_name in features
    
    # GROUP 1: Account Age + Graph Features
    group1 = []
    if has_feat("account_age_days"):
        group1.append("account_age_days")
    
    graph_features_to_interact = [
        "shared_contact_email_count",
        "shared_contact_phone_count",
        "listing_component_size",
        "max_shared_contact_email",
        "max_shared_contact_phone",
    ]
    for feat in graph_features_to_interact:
        if has_feat(feat):
            group1.append(feat)
    
    if len(group1) > 1:
        constraints.append(group1)
    
    # GROUP 2: Payment + Bundle Features (high fraud lift per Exp 10C)
    group2 = []
    payment_bundle_features = ["payment_type", "bundle_tier", "offer_type"]
    for feat in payment_bundle_features:
        if has_feat(feat):
            group2.append(feat)
    
    if len(group2) > 1:
        constraints.append(group2)
    
    return constraints if constraints else []
