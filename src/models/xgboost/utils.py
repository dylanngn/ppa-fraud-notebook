"""
XGBoost training utilities.
"""
import subprocess
import logging
import pandas as pd
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE MANIFEST
# Explicitly defines all valid features for the model.
# Each feature specifies: (dtype_category, required)
#   dtype_category: 'numeric', 'categorical', 'boolean'
#   required: whether the feature must be present
# =============================================================================

# =============================================================================
# FEATURE MANIFEST - Auto-generated from FEATURE_GROUPS in constants.py
# This ensures manifest stays in sync with the authoritative feature definitions.
# DO NOT edit manually - update constants.py instead.
# =============================================================================

def _build_manifest_from_constants() -> Dict[str, Tuple[str, bool]]:
    """Build feature manifest from FEATURE_GROUPS in constants.py."""
    from src.models.config.constants import (
        TIER1_CORE_FEATURES,
        ALL_BOOLEAN_FEATURES,
        TIER2_GRAPH_FEATURES,
        TIME_WEIGHTED_FEATURE_COLUMNS,
        TEXT_FEATURE_COLUMNS,
        CANDIDATE_FEATURES,
    )
    
    manifest = {}
    
    # Core numerical features (required: account_age_days)
    for feat in TIER1_CORE_FEATURES:
        required = feat == "account_age_days"
        if feat in ["payment_type", "bundle_tier", "offer_type"]:
            manifest[feat] = ("categorical", required)
        else:
            manifest[feat] = ("numeric", required)
    
    # Boolean features
    for feat in ALL_BOOLEAN_FEATURES:
        manifest[feat] = ("boolean", False)
    
    # Graph features (all numeric)
    for feat in TIER2_GRAPH_FEATURES:
        manifest[feat] = ("numeric", False)
    
    # Time-weighted features (all numeric)
    for feat in TIME_WEIGHTED_FEATURE_COLUMNS:
        manifest[feat] = ("numeric", False)
    
    # Text features (all numeric)
    for feat in TEXT_FEATURE_COLUMNS:
        manifest[feat] = ("numeric", False)
    
    # Candidate features (all categorical) - Feature Discovery Pipeline
    for feat in CANDIDATE_FEATURES:
        manifest[feat] = ("categorical", False)
    
    return manifest

FEATURE_MANIFEST: Dict[str, Tuple[str, bool]] = _build_manifest_from_constants()


def get_manifest_features() -> Set[str]:
    """Return the set of all features defined in the manifest."""
    return set(FEATURE_MANIFEST.keys())


def validate_features(df: pd.DataFrame, strict: bool = False) -> Tuple[List[str], List[str]]:
    """
    Validate DataFrame columns against the feature manifest.
    
    Args:
        df: DataFrame to validate
        strict: If True, raise error on missing required features
        
    Returns:
        Tuple of (valid_features, invalid_columns)
        
    Raises:
        ValueError: If strict=True and required features are missing or have wrong types
    """
    valid_features = []
    invalid_columns = []
    missing_required = []
    wrong_type = []
    
    manifest_features = get_manifest_features()
    
    for col in df.columns:
        if col not in manifest_features:
            invalid_columns.append(col)
            continue
            
        expected_type, required = FEATURE_MANIFEST[col]
        dtype = df[col].dtype
        
        # Validate dtype matches expected category
        is_valid_type = False
        if expected_type == "numeric":
            is_valid_type = pd.api.types.is_numeric_dtype(dtype)
        elif expected_type == "boolean":
            is_valid_type = pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype)
        elif expected_type == "categorical":
            is_valid_type = (
                pd.api.types.is_categorical_dtype(dtype) or 
                pd.api.types.is_object_dtype(dtype) or
                pd.api.types.is_string_dtype(dtype)
            )
        
        if is_valid_type:
            valid_features.append(col)
        else:
            wrong_type.append(f"{col} (expected {expected_type}, got {dtype})")
            invalid_columns.append(col)
    
    # Check for missing required features
    for feat, (_, required) in FEATURE_MANIFEST.items():
        if required and feat not in df.columns:
            missing_required.append(feat)
    
    if strict:
        errors = []
        if missing_required:
            errors.append(f"Missing required features: {missing_required}")
        if wrong_type:
            errors.append(f"Features with wrong types: {wrong_type}")
        if errors:
            raise ValueError("\n".join(errors))
    else:
        if wrong_type:
            logger.warning(f"Features with unexpected types (excluded): {wrong_type}")
        if invalid_columns:
            logger.debug(f"Non-feature columns in DataFrame (excluded): {[c for c in invalid_columns if c not in [w.split()[0] for w in wrong_type]]}")
    
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
    # Method 1: Check for CUDA via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu_hist"
    except ImportError:
        pass
    
    # Method 2: Check nvidia-smi (if available)
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
    categorical_features = []
    potential_categoricals = [
        # Core categorical features
        "payment_type", "bundle_tier", "offer_type",
        # Feature Discovery Pipeline candidates (2025-12-03)
        "rent_interval", "platforms", "billing_language",
    ]
    
    for feat in potential_categoricals:
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
        List of feature name groups
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
    
    # GROUP 2: Payment + Bundle Features
    group2 = []
    payment_bundle_features = ["payment_type", "bundle_tier"]
    for feat in payment_bundle_features:
        if has_feat(feat):
            group2.append(feat)
    
    if len(group2) > 1:
        constraints.append(group2)
    
    # GROUP 3: Time-Weighted + Graph Features
    group3 = []
    time_weighted_features = [
        "email_velocity_7d",
        "phone_velocity_7d",
        "email_is_burst",
        "phone_is_burst",
        "combined_velocity_7d",
    ]
    for feat in time_weighted_features:
        if has_feat(feat):
            group3.append(feat)
    
    graph_velocity_features = ["shared_contact_email_count", "shared_contact_phone_count"]
    for feat in graph_velocity_features:
        if has_feat(feat) and feat not in group3:
            group3.append(feat)
    
    if len(group3) > 1:
        constraints.append(group3)
    
    # GROUP 4: Interaction Features
    group4 = []
    interaction_features = [
        "new_account_high_email_reuse",
        "new_account_high_phone_reuse",
        "new_account_high_reuse_any",
        "new_account_invoice_payment",
        "account_age_risk_score",
    ]
    for feat in interaction_features:
        if has_feat(feat):
            group4.append(feat)
    
    if has_feat("account_age_days") and "account_age_days" not in [feat for group in constraints for feat in group]:
        group4.append("account_age_days")
    
    if len(group4) > 1:
        constraints.append(group4)
    
    return constraints if constraints else []
