"""
ARCHIVED: Experiment 9 - Feature Selection Methodology Comparison
==================================================================

STATUS: HISTORICAL (completed 2025-12-09)
REASON: Uses deprecated constants (TIER1_*, TIER2_*, PRODUCTION_PROFILE_FEATURES)
        that were removed after the experiment validated auto-selection.

KEY FINDING: Auto-selection (all 278 ETL fields) achieves 0.784 AUC-PR,
             matching manually-curated features (0.776 AUC-PR).
             
OUTCOME: Pipeline simplified to use auto mode (see conf/features/auto.yaml).
         This script is preserved for reference but will not run.

Results archived in:
    - artifacts/feature_selection/method_comparison.csv
    - artifacts/feature_selection/selection_results.json
    - docs/experiment_journal.md (Experiment 9 section)

---
Original docstring below:

Research Gap: Current feature selection is ad-hoc. This experiment
provides statistical justification for feature choices.

Sub-experiments:
    9A: All-fields baseline
        - all_127_fields: Fields with ≥50% coverage
        - all_292_fields: ALL fields with XGBoost native missing handling
    9B: Algorithmic selection comparison (ALL methods from supervisor feedback)
        - RFE (HIGH priority)
        - LASSO (HIGH priority)
        - Information Gain (MEDIUM priority)
        - Chi-Square (MEDIUM priority)
        - Mutual Information (MEDIUM priority)
        - Permutation Importance (MEDIUM priority)
        - Correlation (implicit baseline)
    9C: Graph-only ablation (addresses RQ1 gap)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif, SelectFromModel, chi2, f_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import xgboost as xgb

from src.data.loader import load_data
from src.features.definitions.base import compute_base_features
from src.features.processor import FeatureProcessor
from src.models.config.constants import (
    PRODUCTION_PROFILE_FEATURES,
    TIER2_GRAPH_FEATURES,
    TIER1_CORE_FEATURES,
    TIER1_BOOLEAN_FEATURES,
)
from src.models.utils.common import setup_mlflow
from src.models.xgboost.utils import get_optimal_tree_method, validate_features
from src.utils.hydra_utils import resolve_path
from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
OUTPUT_DIR = ARTIFACTS_DIR / "feature_selection"
NODES_LISTING = ARTIFACTS_DIR / "nodes_listing.parquet"


def load_all_features(min_coverage: float = 0.5) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Load all available features from raw_insertions.parquet (ETL output).
    
    This uses the RAW ETL output, NOT the hand-picked nodes_listing.parquet.
    raw_insertions.parquet contains all ~292 flattened fields from the ETL pipeline.
    
    Args:
        min_coverage: Minimum coverage threshold (0.0 = all fields, 0.5 = ≥50% coverage)
    
    Returns:
        Tuple of (X, feature_names, y)
    """
    logger.info(f"Loading data from raw_insertions.parquet (ETL output) with min_coverage={min_coverage}...")
    
    # Load raw_insertions.parquet (ETL output with ALL fields)
    RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"
    df = pl.read_parquet(RAW_INSERTIONS)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns from raw_insertions.parquet")
    
    # Add fraud label
    df = df.with_columns(
        pl.col("fraud_flag").is_not_null().cast(pl.Int8).alias("is_fraud")
    )
    
    # Identify feature columns (exclude IDs, targets, dates, leakage)
    exclude_patterns = {
        "object_reference", "insertion_id", "owner_id", "user_id",
        "submission_at", "fraud_flag", "is_fraud", "first_published_date",
        "listing_created_at", "account_created_at", "contact_emails_hash",
        "user_ip_address_hash",
        # Exclude potential leakage
        "seonFraudScore",  # This is the baseline we're comparing against
        # Exclude metadata fields
        "listing.meta.created", "listing.meta.createdAt", "listing.meta.updatedAt",
        "listing.meta.fileModified",
    }
    
    feature_cols = []
    for col in df.columns:
        # Skip if in exclude set
        if col in exclude_patterns:
            continue
        # Skip if contains any exclude pattern
        if any(pattern in col for pattern in ["_hash", "legacy.personId", "legacy.ppaPersonId"]):
            continue
        feature_cols.append(col)
    
    logger.info(f"Total feature columns (after exclusions): {len(feature_cols)}")
    
    # Filter to columns with minimum coverage
    selected_cols = []
    coverage_info = {}
    for col in feature_cols:
        coverage = 1 - (df[col].null_count() / len(df))
        coverage_info[col] = coverage
        if coverage >= min_coverage:
            selected_cols.append(col)
    
    logger.info(f"Features with ≥{min_coverage*100:.0f}% coverage: {len(selected_cols)} / {len(feature_cols)}")
    
    # Convert to pandas
    pdf = df.select(selected_cols + ["is_fraud"]).to_pandas()
    
    # Handle categorical columns
    for col in selected_cols:
        if pdf[col].dtype == 'object' or pdf[col].dtype.name == 'category':
            # Use label encoding, treating NaN as a separate category
            pdf[col] = pdf[col].fillna('__MISSING__')
            pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))
    
    # Fill remaining NaN with -999 (XGBoost handles this, sklearn needs explicit value)
    pdf = pdf.fillna(-999)
    
    X = pdf[selected_cols]
    y = pdf["is_fraud"]
    
    return X, selected_cols, y


def load_all_292_fields() -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Load ALL 292 fields from raw_insertions.parquet.
    XGBoost handles missing values natively, so minimal imputation needed.
    
    Returns:
        Tuple of (X, feature_names, y)
    """
    logger.info("Loading ALL 292 fields from raw_insertions.parquet...")
    
    RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"
    df = pl.read_parquet(RAW_INSERTIONS)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Add fraud label
    df = df.with_columns(
        pl.col("fraud_flag").is_not_null().cast(pl.Int8).alias("is_fraud")
    )
    
    # Exclude metadata, IDs, targets, and leakage fields
    exclude_patterns = [
        "object_reference", "owner_id", "user_id", "submission_at",
        "fraud_flag", "is_fraud", "first_published_date", "listing_created_at",
        "account_created_at", "contact_emails_hash", "user_ip_address_hash",
        "seonFraudScore",  # Baseline - not a feature
        "legacy.personId", "legacy.ppaPersonId",  # IDs
    ]
    
    feature_cols = []
    for col in df.columns:
        exclude = False
        for pattern in exclude_patterns:
            if pattern in col:
                exclude = True
                break
        if not exclude and col not in ["is_fraud"]:
            feature_cols.append(col)
    
    logger.info(f"Total feature columns (excluding metadata): {len(feature_cols)}")
    
    # Convert to pandas
    pdf = df.select(feature_cols + ["is_fraud"]).to_pandas()
    
    # Handle categorical columns
    for col in feature_cols:
        if pdf[col].dtype == 'object' or pdf[col].dtype.name == 'category':
            # Use label encoding, treating NaN as a separate category
            pdf[col] = pdf[col].fillna('__MISSING__')
            pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))
    
    # For numerical columns, keep NaN (XGBoost handles natively)
    # But for sklearn methods, we need to fill them
    pdf = pdf.fillna(-999)
    
    X = pdf[feature_cols]
    y = pdf["is_fraud"]
    
    return X, feature_cols, y


def load_production_features(config: DictConfig) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Load features using production config.
    
    Returns:
        Tuple of (X, feature_names, y)
    """
    logger.info("Loading production features...")
    
    df = load_data()
    df = compute_base_features(df, cutoff_date=datetime.now(timezone.utc), config=None)
    
    processor = FeatureProcessor.from_config(config)
    df = processor.process(df, datetime.now(timezone.utc))
    
    pdf = df.to_pandas()
    
    # Get feature columns
    exclude_cols = {"insertion_id", "object_reference", "is_fraud", "submission_at", "fraud_flag"}
    feature_cols = [c for c in pdf.columns if c not in exclude_cols and c in PRODUCTION_PROFILE_FEATURES]
    
    # Handle categorical
    for col in feature_cols:
        if pdf[col].dtype == 'object' or pdf[col].dtype.name == 'category':
            pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))
    
    pdf = pdf.fillna(-999)
    
    X = pdf[feature_cols]
    y = pdf["is_fraud"]
    
    return X, feature_cols, y


def load_graph_only_features() -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Load only graph-derived features (TIER2) by computing them on-the-fly.
    
    Returns:
        Tuple of (X, feature_names, y)
    """
    logger.info("Computing graph-only features...")
    
    # Load raw data
    df = load_data()
    df = compute_base_features(df, cutoff_date=datetime.now(timezone.utc), config=None)
    
    # Create processor to compute graph features
    processor = FeatureProcessor(
        categories=["graph", "advanced_graph"],
        include_groups=["graph_all"]
    )
    df = processor.process(df, datetime.now(timezone.utc))
    
    # Add fraud label if not present
    if "is_fraud" not in df.columns:
        df = df.with_columns(
            pl.col("fraud_flag").is_not_null().cast(pl.Int8).alias("is_fraud")
        )
    
    # Select only graph features that exist (TIER2)
    available_graph_features = [c for c in TIER2_GRAPH_FEATURES if c in df.columns]
    
    if len(available_graph_features) == 0:
        # Try to find any graph-related columns computed by processor
        graph_keywords = ["pagerank", "degree", "cluster", "betweenness", "authority", "hub", 
                         "shared", "component", "contact"]
        exclude_cols = {"insertion_id", "object_reference", "is_fraud", "submission_at", "fraud_flag"}
        available_graph_features = [c for c in df.columns 
                                    if c not in exclude_cols and
                                    any(kw in c.lower() for kw in graph_keywords)]
        logger.info(f"Using detected graph features: {available_graph_features}")
    
    logger.info(f"Available graph features: {len(available_graph_features)}")
    
    if len(available_graph_features) == 0:
        raise ValueError("No graph features could be computed. Check graph artifacts exist.")
    
    pdf = df.select(available_graph_features + ["is_fraud"]).to_pandas()
    
    # Handle categorical columns
    for col in available_graph_features:
        if pdf[col].dtype == 'object' or pdf[col].dtype.name == 'category':
            pdf[col] = LabelEncoder().fit_transform(pdf[col].astype(str))
    
    pdf = pdf.fillna(-999)
    
    X = pdf[available_graph_features]
    y = pdf["is_fraud"]
    
    return X, available_graph_features, y


def correlation_select(X: pd.DataFrame, y: pd.Series, n: int = 50) -> List[str]:
    """Select top N features by absolute correlation with target."""
    correlations = {}
    for col in X.columns:
        corr = np.corrcoef(X[col].values, y.values)[0, 1]
        if not np.isnan(corr):
            correlations[col] = abs(corr)
    
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_features[:n]]


def mutual_info_select(X: pd.DataFrame, y: pd.Series, n: int = 50) -> List[str]:
    """Select top N features by mutual information."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    return mi_df.head(n)['feature'].tolist()


def lasso_select(X: pd.DataFrame, y: pd.Series) -> List[str]:
    """Select features with non-zero LASSO coefficients."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Get features with non-zero coefficients
    selected = [X.columns[i] for i in range(len(X.columns)) if abs(lasso.coef_[i]) > 1e-6]
    
    logger.info(f"LASSO selected {len(selected)} features (alpha={lasso.alpha_:.6f})")
    return selected


def rfe_select(X: pd.DataFrame, y: pd.Series, n: int = 50) -> List[str]:
    """Select top N features using Recursive Feature Elimination with XGBoost."""
    # Use a small XGBoost model for RFE (faster)
    estimator = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        tree_method=get_optimal_tree_method(),
        verbosity=0
    )
    
    rfe = RFE(estimator=estimator, n_features_to_select=n, step=10, verbose=0)
    rfe.fit(X, y)
    
    selected = X.columns[rfe.support_].tolist()
    return selected


def information_gain_select(X: pd.DataFrame, y: pd.Series, n: int = 50) -> List[str]:
    """
    Select top N features by Information Gain (using ANOVA F-value as proxy).
    
    Note: For continuous features, ANOVA F-value approximates information gain.
    For truly discrete features, we'd use mutual_info_classif with discrete_features=True.
    """
    # Use f_classif (ANOVA F-value) which measures the linear dependency
    # This is a common proxy for information gain in sklearn
    f_scores, p_values = f_classif(X, y)
    
    # Handle NaN scores (can happen with constant features)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    
    ig_df = pd.DataFrame({
        'feature': X.columns,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)
    
    selected = ig_df.head(n)['feature'].tolist()
    logger.info(f"Information Gain: Top feature = {selected[0]} (F={ig_df.iloc[0]['f_score']:.2f})")
    return selected


def chi_square_select(X: pd.DataFrame, y: pd.Series, n: int = 50) -> List[str]:
    """
    Select top N features by Chi-Square test.
    
    Note: Chi-square requires non-negative values, so we apply MinMax scaling.
    """
    # Chi-square requires non-negative values
    # Scale to [0, 1] range
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Apply chi-square test
    chi2_scores, p_values = chi2(X_scaled, y)
    
    # Handle NaN scores
    chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
    
    chi2_df = pd.DataFrame({
        'feature': X.columns,
        'chi2_score': chi2_scores,
        'p_value': p_values
    }).sort_values('chi2_score', ascending=False)
    
    selected = chi2_df.head(n)['feature'].tolist()
    logger.info(f"Chi-Square: Top feature = {selected[0]} (χ²={chi2_df.iloc[0]['chi2_score']:.2f})")
    return selected


def permutation_importance_select(X: pd.DataFrame, y: pd.Series, n: int = 50) -> List[str]:
    """
    Select top N features by Permutation Importance.
    
    Uses a trained XGBoost model and measures performance drop when each feature is shuffled.
    This is model-agnostic and captures non-linear relationships.
    """
    logger.info("Computing permutation importance (this may take a while)...")
    
    # Train a quick XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        tree_method=get_optimal_tree_method(),
        verbosity=0
    )
    
    # Use a subset for speed (permutation importance is slow)
    sample_size = min(10000, len(X))
    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y
    
    # Split for train/test
    split_idx = int(len(X_sample) * 0.8)
    X_train, X_test = X_sample.iloc[:split_idx], X_sample.iloc[split_idx:]
    y_train, y_test = y_sample.iloc[:split_idx], y_sample.iloc[split_idx:]
    
    model.fit(X_train, y_train)
    
    # Compute permutation importance
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring='average_precision',
        n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    selected = perm_df.head(n)['feature'].tolist()
    logger.info(f"Permutation Importance: Top feature = {selected[0]} (importance={perm_df.iloc[0]['importance_mean']:.4f})")
    return selected


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    n_windows: int = 10,
    config: Optional[DictConfig] = None
) -> Dict[str, Any]:
    """
    Train XGBoost with time-based CV and evaluate.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_windows: Number of evaluation windows
        config: Optional Hydra config for XGBoost params
    
    Returns:
        Dictionary with mean metrics
    """
    # Sort by index (assumes index is time-ordered)
    # For simplicity, use K-fold style but treat as time-split
    n_samples = len(X)
    window_size = n_samples // (n_windows + 1)
    
    auc_prs = []
    p_at_100s = []
    
    # Default XGBoost params
    xgb_params = {
        'n_estimators': 300,
        'max_depth': 9,
        'learning_rate': 0.148,
        'min_child_weight': 15,
        'subsample': 0.832,
        'colsample_bytree': 0.8,
        'reg_alpha': 9.92,
        'reg_lambda': 9.03,
        'random_state': 42,
        'tree_method': get_optimal_tree_method(),
        'verbosity': 0
    }
    
    if config is not None and hasattr(config, 'model') and hasattr(config.model, 'params'):
        xgb_params.update(dict(config.model.params))
    
    for i in range(n_windows):
        train_end = (i + 1) * window_size
        test_start = train_end
        test_end = min(test_start + window_size, n_samples)
        
        if test_end <= test_start:
            break
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]
        
        if len(X_train) < 100 or len(X_test) < 100:
            continue
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, proba)
        
        auc_prs.append(metrics['auc_pr'])
        p_at_100s.append(metrics.get('p_at_100', 0))
    
    return {
        'mean_auc_pr': np.mean(auc_prs) if auc_prs else 0,
        'std_auc_pr': np.std(auc_prs) if auc_prs else 0,
        'mean_p_at_100': np.mean(p_at_100s) if p_at_100s else 0,
        'n_windows': len(auc_prs),
        'n_features': len(feature_names),
        'feature_names': feature_names
    }


def run_9a_all_fields(config: DictConfig) -> Dict[str, Any]:
    """
    Experiment 9A: All-Fields Baseline
    
    Test performance with:
    - all_127_fields: Fields with ≥50% coverage
    - all_292_fields: ALL fields (XGBoost handles missing natively)
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 9A: ALL-FIELDS BASELINE")
    logger.info("=" * 70)
    
    results = {}
    
    # Test 1: All 127 fields with ≥50% coverage
    logger.info("\n--- 9A.1: All fields with ≥50% coverage ---")
    X_127, features_127, y_127 = load_all_features(min_coverage=0.5)
    
    logger.info(f"Training with {len(features_127)} features (≥50% coverage)...")
    result_127 = train_and_evaluate(X_127, y_127, features_127, n_windows=10, config=config)
    result_127['method'] = 'all_127_fields'
    result_127['description'] = f'All {len(features_127)} fields with ≥50% coverage'
    results['all_127_fields'] = result_127
    
    logger.info(f"Results (127 fields): Mean AUC-PR = {result_127['mean_auc_pr']:.4f} (±{result_127['std_auc_pr']:.4f})")
    
    # Test 2: ALL 292 fields (no coverage filter)
    logger.info("\n--- 9A.2: ALL 292 fields (no coverage filter) ---")
    try:
        X_292, features_292, y_292 = load_all_292_fields()
        
        logger.info(f"Training with {len(features_292)} features (all fields)...")
        result_292 = train_and_evaluate(X_292, y_292, features_292, n_windows=10, config=config)
        result_292['method'] = 'all_292_fields'
        result_292['description'] = f'All {len(features_292)} fields (no coverage filter)'
        results['all_292_fields'] = result_292
        
        logger.info(f"Results (292 fields): Mean AUC-PR = {result_292['mean_auc_pr']:.4f} (±{result_292['std_auc_pr']:.4f})")
        
        # Compare
        delta = result_292['mean_auc_pr'] - result_127['mean_auc_pr']
        logger.info(f"\n📊 292 vs 127 fields: {delta:+.4f} AUC-PR difference")
        if delta > 0.01:
            logger.info("   → Using all fields IMPROVES performance")
        elif delta < -0.01:
            logger.info("   → Using all fields DECREASES performance (overfitting?)")
        else:
            logger.info("   → Minimal difference between coverage thresholds")
    except Exception as e:
        logger.warning(f"Could not load all 292 fields: {e}")
        results['all_292_fields'] = {'error': str(e)}
    
    return results


def run_9b_comparison(config: DictConfig) -> List[Dict[str, Any]]:
    """
    Experiment 9B: Algorithmic Selection Comparison
    
    Compare ALL methods from supervisor feedback:
    - RFE (HIGH priority)
    - LASSO (HIGH priority)  
    - Information Gain (MEDIUM priority)
    - Chi-Square (MEDIUM priority)
    - Mutual Information (MEDIUM priority)
    - Permutation Importance (MEDIUM priority)
    - Correlation (implicit baseline)
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 9B: ALGORITHMIC SELECTION COMPARISON")
    logger.info("All methods from supervisor feedback (Section 2.4 & 4.2)")
    logger.info("=" * 70)
    
    X, all_features, y = load_all_features(min_coverage=0.5)
    
    results = []
    
    # Method 1: Production (manual selection) - baseline
    # NOTE: PRODUCTION_PROFILE_FEATURES use processed names (e.g., "rooms")
    # but raw_insertions.parquet has raw names (e.g., "listing.rooms")
    # We map known equivalents here
    RAW_TO_PROD_MAP = {
        "listing.characteristics.numberOfRooms": "rooms",
        "listing.characteristics.livingSpace": "living_space",
        "listing.localization.primary.geo.latitude": "latitude",
        "listing.localization.primary.geo.longitude": "longitude",
        "listing.offerType": "offer_type",
        "bundle.numberOfDays": "bundle_period",
        "bundle.tier": "bundle_tier",
    }
    
    logger.info("\n--- Method 1: Production-Equivalent Features [BASELINE] ---")
    # Find columns that match production features or their raw equivalents
    prod_features = []
    for col in X.columns:
        # Direct match
        if col in PRODUCTION_PROFILE_FEATURES:
            prod_features.append(col)
        # Mapped match
        elif col in RAW_TO_PROD_MAP and RAW_TO_PROD_MAP[col] in PRODUCTION_PROFILE_FEATURES:
            prod_features.append(col)
    
    if len(prod_features) == 0:
        logger.warning("No production feature equivalents found in raw data. Using top 30 by coverage as baseline.")
        # Use top 30 columns with highest coverage as fallback
        prod_features = list(X.columns[:min(30, len(X.columns))])
    
    X_prod = X[prod_features]
    prod_result = train_and_evaluate(X_prod, y, prod_features, n_windows=10, config=config)
    prod_result['method'] = 'production_equivalent'
    prod_result['description'] = f'Production-equivalent features ({len(prod_features)})'
    prod_result['priority'] = 'BASELINE'
    results.append(prod_result)
    logger.info(f"Production: Mean AUC-PR = {prod_result['mean_auc_pr']:.4f}")
    
    # Method 2: RFE top-50 (HIGH priority)
    logger.info("\n--- Method 2: RFE Top-50 [HIGH PRIORITY] ---")
    rfe_features = rfe_select(X, y, n=50)
    X_rfe = X[rfe_features]
    rfe_result = train_and_evaluate(X_rfe, y, rfe_features, n_windows=10, config=config)
    rfe_result['method'] = 'rfe_top_50'
    rfe_result['description'] = 'Recursive Feature Elimination (50)'
    rfe_result['priority'] = 'HIGH'
    results.append(rfe_result)
    logger.info(f"RFE: Mean AUC-PR = {rfe_result['mean_auc_pr']:.4f}")
    
    # Method 3: LASSO (HIGH priority)
    logger.info("\n--- Method 3: LASSO Selection [HIGH PRIORITY] ---")
    lasso_features = lasso_select(X, y)
    if len(lasso_features) > 0:
        X_lasso = X[lasso_features]
        lasso_result = train_and_evaluate(X_lasso, y, lasso_features, n_windows=10, config=config)
    else:
        lasso_result = {'mean_auc_pr': 0, 'std_auc_pr': 0, 'n_features': 0, 'feature_names': []}
    lasso_result['method'] = 'lasso_selected'
    lasso_result['description'] = f'LASSO non-zero coefficients ({len(lasso_features)})'
    lasso_result['priority'] = 'HIGH'
    results.append(lasso_result)
    logger.info(f"LASSO: Mean AUC-PR = {lasso_result['mean_auc_pr']:.4f}")
    
    # Method 4: Information Gain / F-Score (MEDIUM priority)
    logger.info("\n--- Method 4: Information Gain Top-50 [MEDIUM PRIORITY] ---")
    ig_features = information_gain_select(X, y, n=50)
    X_ig = X[ig_features]
    ig_result = train_and_evaluate(X_ig, y, ig_features, n_windows=10, config=config)
    ig_result['method'] = 'information_gain_top_50'
    ig_result['description'] = 'Top 50 by Information Gain (ANOVA F-score)'
    ig_result['priority'] = 'MEDIUM'
    results.append(ig_result)
    logger.info(f"Information Gain: Mean AUC-PR = {ig_result['mean_auc_pr']:.4f}")
    
    # Method 5: Chi-Square (MEDIUM priority)
    logger.info("\n--- Method 5: Chi-Square Top-50 [MEDIUM PRIORITY] ---")
    chi2_features = chi_square_select(X, y, n=50)
    X_chi2 = X[chi2_features]
    chi2_result = train_and_evaluate(X_chi2, y, chi2_features, n_windows=10, config=config)
    chi2_result['method'] = 'chi_square_top_50'
    chi2_result['description'] = 'Top 50 by Chi-Square test'
    chi2_result['priority'] = 'MEDIUM'
    results.append(chi2_result)
    logger.info(f"Chi-Square: Mean AUC-PR = {chi2_result['mean_auc_pr']:.4f}")
    
    # Method 6: Mutual Information (MEDIUM priority)
    logger.info("\n--- Method 6: Mutual Information Top-50 [MEDIUM PRIORITY] ---")
    mi_features = mutual_info_select(X, y, n=50)
    X_mi = X[mi_features]
    mi_result = train_and_evaluate(X_mi, y, mi_features, n_windows=10, config=config)
    mi_result['method'] = 'mutual_info_top_50'
    mi_result['description'] = 'Top 50 by Mutual Information'
    mi_result['priority'] = 'MEDIUM'
    results.append(mi_result)
    logger.info(f"Mutual Info: Mean AUC-PR = {mi_result['mean_auc_pr']:.4f}")
    
    # Method 7: Permutation Importance (MEDIUM priority)
    logger.info("\n--- Method 7: Permutation Importance Top-50 [MEDIUM PRIORITY] ---")
    perm_features = permutation_importance_select(X, y, n=50)
    X_perm = X[perm_features]
    perm_result = train_and_evaluate(X_perm, y, perm_features, n_windows=10, config=config)
    perm_result['method'] = 'permutation_importance_top_50'
    perm_result['description'] = 'Top 50 by Permutation Importance'
    perm_result['priority'] = 'MEDIUM'
    results.append(perm_result)
    logger.info(f"Permutation Importance: Mean AUC-PR = {perm_result['mean_auc_pr']:.4f}")
    
    # Method 8: Correlation top-50 (implicit baseline)
    logger.info("\n--- Method 8: Correlation Top-50 [IMPLICIT BASELINE] ---")
    corr_features = correlation_select(X, y, n=50)
    X_corr = X[corr_features]
    corr_result = train_and_evaluate(X_corr, y, corr_features, n_windows=10, config=config)
    corr_result['method'] = 'correlation_top_50'
    corr_result['description'] = 'Top 50 by Pearson correlation'
    corr_result['priority'] = 'BASELINE'
    results.append(corr_result)
    logger.info(f"Correlation: Mean AUC-PR = {corr_result['mean_auc_pr']:.4f}")
    
    return results


def run_9c_graph_only(config: DictConfig) -> Dict[str, Any]:
    """
    Experiment 9C: Graph-Only Ablation
    
    Test graph features alone to quantify their standalone value (RQ1 gap).
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 9C: GRAPH-ONLY ABLATION (RQ1 GAP)")
    logger.info("=" * 70)
    
    X, feature_names, y = load_graph_only_features()
    
    logger.info(f"Training with {len(feature_names)} graph-only features...")
    logger.info(f"Features: {feature_names}")
    
    results = train_and_evaluate(X, y, feature_names, n_windows=10, config=config)
    results['method'] = 'graph_only'
    results['description'] = f'Graph features only ({len(feature_names)} TIER2 features)'
    
    logger.info(f"Results: Mean AUC-PR = {results['mean_auc_pr']:.4f} (±{results['std_auc_pr']:.4f})")
    
    # Also run tabular-only for comparison
    logger.info("\n--- Tabular-Only Baseline (for comparison) ---")
    
    # Load from nodes_listing.parquet which has both tabular and graph features
    df_tab = pl.read_parquet(NODES_LISTING)
    
    # Add fraud label if not present
    if "is_fraud" not in df_tab.columns:
        df_tab = df_tab.with_columns(
            pl.col("fraud_flag").is_not_null().cast(pl.Int8).alias("is_fraud")
        )
    
    # Get available tabular (TIER1) features, excluding graph features
    exclude_cols = {"insertion_id", "object_reference", "is_fraud", "submission_at", "fraud_flag"}
    tabular_features = [f for f in TIER1_CORE_FEATURES + TIER1_BOOLEAN_FEATURES 
                        if f in df_tab.columns and f not in TIER2_GRAPH_FEATURES]
    
    if len(tabular_features) == 0:
        # Fallback: use non-graph features
        graph_keywords = ["pagerank", "degree", "cluster", "betweenness", "authority", "hub"]
        tabular_features = [c for c in df_tab.columns 
                           if c not in exclude_cols 
                           and not any(kw in c.lower() for kw in graph_keywords)]
        logger.warning(f"No TIER1 features found, using non-graph features: {tabular_features[:10]}...")
    
    logger.info(f"Tabular features: {len(tabular_features)}")
    
    pdf_tab = df_tab.select(tabular_features + ["is_fraud"]).to_pandas()
    
    for col in tabular_features:
        if pdf_tab[col].dtype == 'object' or pdf_tab[col].dtype.name == 'category':
            pdf_tab[col] = LabelEncoder().fit_transform(pdf_tab[col].astype(str))
    pdf_tab = pdf_tab.fillna(-999)
    
    X_tab = pdf_tab[tabular_features]
    y_tab = pdf_tab["is_fraud"]
    
    tabular_result = train_and_evaluate(X_tab, y_tab, tabular_features, n_windows=10, config=config)
    tabular_result['method'] = 'tabular_only'
    tabular_result['description'] = f'Tabular features only ({len(tabular_features)} TIER1 features)'
    
    logger.info(f"Tabular-only: Mean AUC-PR = {tabular_result['mean_auc_pr']:.4f}")
    
    return {
        'graph_only': results,
        'tabular_only': tabular_result
    }


def plot_comparison(results: List[Dict], output_path: Path) -> None:
    """Create comparison visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = [r['method'] for r in results]
    auc_prs = [r['mean_auc_pr'] for r in results]
    stds = [r.get('std_auc_pr', 0) for r in results]
    n_features = [r['n_features'] for r in results]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, auc_prs, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    
    # Add feature count labels on bars
    for i, (bar, n) in enumerate(zip(bars, n_features)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                f'n={n}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Mean AUC-PR')
    ax.set_xlabel('Feature Selection Method')
    ax.set_title('Experiment 9B: Feature Selection Method Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r.get('description', r['method'])[:25] + '...' if len(r.get('description', r['method'])) > 25 
                        else r.get('description', r['method']) for r in results], 
                       rotation=45, ha='right')
    
    # Add baseline line for production
    prod_auc = next((r['mean_auc_pr'] for r in results if r['method'] == 'production_manual'), None)
    if prod_auc:
        ax.axhline(y=prod_auc, color='red', linestyle='--', label=f'Production baseline ({prod_auc:.3f})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot: {output_path}")
    plt.close()


def save_results(all_results: Dict, output_dir: Path) -> None:
    """Save all experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'results': all_results
    }
    
    # Make serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(output_dir / 'selection_results.json', 'w') as f:
        json.dump(make_serializable(summary), f, indent=2, default=str)
    
    logger.info(f"Saved results to {output_dir / 'selection_results.json'}")
    
    # Save comparison CSV if we have 9B results
    if '9b_comparison' in all_results:
        comparison_df = pd.DataFrame(all_results['9b_comparison'])
        comparison_df = comparison_df[['method', 'n_features', 'mean_auc_pr', 'std_auc_pr', 'description']]
        comparison_df = comparison_df.sort_values('mean_auc_pr', ascending=False)
        comparison_df.to_csv(output_dir / 'method_comparison.csv', index=False)
        logger.info(f"Saved comparison CSV: {output_dir / 'method_comparison.csv'}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Experiment 9: Feature Selection Methodology Comparison
    
    Run with:
        python -m src.experiments.exp9_feature_selection +exp9.method=all_fields
        python -m src.experiments.exp9_feature_selection +exp9.method=comparison
        python -m src.experiments.exp9_feature_selection +exp9.method=graph_only
        python -m src.experiments.exp9_feature_selection +exp9.method=full
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 9: FEATURE SELECTION METHODOLOGY")
    logger.info("=" * 70)
    
    # Get method from config (default to 'full')
    method = OmegaConf.select(cfg, "exp9.method", default="full")
    logger.info(f"Running method: {method}")
    
    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    experiment_name = cfg.get('experiment_name', 'feature-selection-exp9')
    setup_mlflow(experiment_name)
    
    all_results = {}
    
    with mlflow.start_run(
        run_name=f"exp9_{method}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "feature_selection", "method": method}
    ):
        mlflow.log_param("method", method)
        
        start_time = time.time()
        
        if method in ['all_fields', 'full']:
            results_9a = run_9a_all_fields(cfg)
            all_results['9a_all_fields'] = results_9a
            
            # Log metrics for each configuration
            if 'all_127_fields' in results_9a:
                r = results_9a['all_127_fields']
                mlflow.log_metrics({
                    '9a_127_mean_auc_pr': r['mean_auc_pr'],
                    '9a_127_n_features': r['n_features']
                })
            if 'all_292_fields' in results_9a and 'error' not in results_9a['all_292_fields']:
                r = results_9a['all_292_fields']
                mlflow.log_metrics({
                    '9a_292_mean_auc_pr': r['mean_auc_pr'],
                    '9a_292_n_features': r['n_features']
                })
        
        if method in ['comparison', 'full']:
            results_9b = run_9b_comparison(cfg)
            all_results['9b_comparison'] = results_9b
            
            # Log best method
            best = max(results_9b, key=lambda x: x['mean_auc_pr'])
            mlflow.log_metrics({
                '9b_best_auc_pr': best['mean_auc_pr'],
                '9b_best_method': 0  # Log method as tag instead
            })
            mlflow.set_tag('9b_best_method', best['method'])
            
            # Create comparison plot
            plot_comparison(results_9b, OUTPUT_DIR / 'method_comparison.png')
            mlflow.log_artifact(str(OUTPUT_DIR / 'method_comparison.png'))
        
        if method in ['graph_only', 'full']:
            results_9c = run_9c_graph_only(cfg)
            all_results['9c_graph_only'] = results_9c
            mlflow.log_metrics({
                '9c_graph_only_auc_pr': results_9c['graph_only']['mean_auc_pr'],
                '9c_tabular_only_auc_pr': results_9c['tabular_only']['mean_auc_pr'],
            })
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Save all results
        save_results(all_results, OUTPUT_DIR)
        mlflow.log_artifacts(str(OUTPUT_DIR))
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT 9 COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        
        if '9a_all_fields' in all_results:
            logger.info("\n--- 9A: All-Fields Baseline ---")
            results_9a = all_results['9a_all_fields']
            if 'all_127_fields' in results_9a:
                r = results_9a['all_127_fields']
                logger.info(f"  127 fields (≥50% coverage): {r['mean_auc_pr']:.4f} AUC-PR")
            if 'all_292_fields' in results_9a and 'error' not in results_9a['all_292_fields']:
                r = results_9a['all_292_fields']
                logger.info(f"  292 fields (all):            {r['mean_auc_pr']:.4f} AUC-PR")
        
        if '9b_comparison' in all_results:
            logger.info("\n--- 9B: Method Comparison Summary ---")
            logger.info(f"{'Method':<30} {'Priority':<10} {'AUC-PR':>10} {'Features':>10}")
            logger.info("-" * 65)
            for r in sorted(all_results['9b_comparison'], key=lambda x: x['mean_auc_pr'], reverse=True):
                logger.info(f"  {r['method']:<28} {r.get('priority', 'N/A'):<10} {r['mean_auc_pr']:>10.4f} {r['n_features']:>10}")
        
        if '9c_graph_only' in all_results:
            logger.info("\n--- 9C: RQ1 Gap - Graph-Only Ablation ---")
            graph_auc = all_results['9c_graph_only']['graph_only']['mean_auc_pr']
            tab_auc = all_results['9c_graph_only']['tabular_only']['mean_auc_pr']
            logger.info(f"  Graph-only:   {graph_auc:.4f}")
            logger.info(f"  Tabular-only: {tab_auc:.4f}")
            logger.info(f"  Delta:        {graph_auc - tab_auc:+.4f}")
            logger.info(f"  → Graph features {'HAVE' if graph_auc > 0.4 else 'may NOT have'} standalone predictive value")
    
    return all_results


if __name__ == "__main__":
    main()
