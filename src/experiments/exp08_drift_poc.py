"""
Experiment 8: Feature Drift Detection POC using Evidently AI

This POC detects data drift across training windows WITHOUT modifying
the existing data pipeline. It reads directly from raw_insertions.parquet
which contains all 292 columns.

Research Question:
    How can we detect emerging fraud patterns that our handcrafted features don't capture?

Approach:
    1. Compare feature distributions between consecutive training windows
    2. Flag features with significant drift (PSI > 0.1 or p-value < 0.05)
    3. Track Seon fraud score distribution as external baseline
    4. Generate actionable alerts when drift exceeds thresholds

Usage:
    python -m src.experiments.exp8_drift_poc

Output:
    - artifacts/drift_monitoring/drift_report.html
    - artifacts/drift_monitoring/drift_summary.json
    - artifacts/drift_monitoring/drift_timeseries.png
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# Evidently AI imports
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("WARNING: Evidently not installed. Run: pip install evidently")

from src.utils.hydra_utils import resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"
OUTPUT_DIR = ARTIFACTS_DIR / "drift_monitoring"

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Features we train on (from nodes_listing / FeatureProcessor)
TRAINING_FEATURES = [
    # Numerical features
    "price_rent_gross",
    "price_buy",
    "living_space",
    "rooms",
    "year_built",
    "floor",
    "num_floors",
    "latitude",
    "longitude",
    
    # Categorical features
    "offer_type",
    "payment_type",
    "bundle_tier",
    "customer_segment",
    "language",
    "region",
]

# Boolean features (stored in nodes_listing)
BOOLEAN_FEATURES = [
    "has_balcony",
    "has_parking",
    "has_elevator",
    "has_nice_view",
    "has_garage",
    "is_child_friendly",
    "is_quiet",
    "has_washing_machine",
    "are_pets_allowed",
    "is_wheelchair_accessible",
    "is_old",
    "is_new_building",
]

# Additional monitoring features (NOT in nodes_listing but useful for drift)
# These are read directly from raw_insertions.parquet
ADDITIONAL_MONITORING_FEATURES = [
    "auto_approval_criteria.meta.seonFraudScore",  # Seon baseline drift
    "bundle.initialPrice",                          # Pricing patterns
    "auto_approval_criteria.criteria.hasUsedSameEmailBefore",  # Behavioral
]

# Column mapping: raw_insertions -> nodes_listing names
COLUMN_MAPPING = {
    "listing.offerType": "offer_type",
    "listing.prices.rent.gross": "price_rent_gross",
    "listing.prices.buy.price": "price_buy",
    "listing.characteristics.livingSpace": "living_space",
    "listing.characteristics.numberOfRooms": "rooms",
    "listing.characteristics.yearBuilt": "year_built",
    "listing.characteristics.floor": "floor",
    "listing.characteristics.numberOfFloors": "num_floors",
    "listing.address.geoCoordinates.latitude": "latitude",
    "listing.address.geoCoordinates.longitude": "longitude",
    "listing.address.region": "region",
    "listing.lister.billing.payment.paymentType": "payment_type",
    "bundle.tier": "bundle_tier",
    "listing.localization.primary": "language",
    # Boolean mappings
    "listing.characteristics.hasBalcony": "has_balcony",
    "listing.characteristics.hasParking": "has_parking",
    "listing.characteristics.hasElevator": "has_elevator",
    "listing.characteristics.hasNiceView": "has_nice_view",
    "listing.characteristics.hasGarage": "has_garage",
    "listing.characteristics.isChildFriendly": "is_child_friendly",
    "listing.characteristics.isQuiet": "is_quiet",
    "listing.characteristics.hasWashingMachine": "has_washing_machine",
    "listing.characteristics.arePetsAllowed": "are_pets_allowed",
    "listing.characteristics.isWheelchairAccessible": "is_wheelchair_accessible",
    "listing.characteristics.isOldBuilding": "is_old",
    "listing.characteristics.isNewBuilding": "is_new_building",
}

# Drift thresholds
PSI_THRESHOLD = 0.1  # Population Stability Index threshold
DRIFT_SHARE_THRESHOLD = 0.3  # Alert if >30% of features drift


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data() -> pl.DataFrame:
    """Load raw insertions with all columns."""
    logger.info(f"Loading data from {RAW_INSERTIONS}...")
    df = pl.read_parquet(RAW_INSERTIONS)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def prepare_monitoring_data(df: pl.DataFrame) -> pd.DataFrame:
    """
    Prepare data for drift monitoring.
    
    Maps raw column names to standardized names and selects relevant features.
    """
    # Build selection expressions
    select_exprs = [
        pl.col("submission_at"),
        pl.col("object_reference").alias("insertion_id"),
    ]
    
    # Add mapped columns
    for raw_col, clean_name in COLUMN_MAPPING.items():
        if raw_col in df.columns:
            select_exprs.append(pl.col(raw_col).alias(clean_name))
    
    # Add additional monitoring columns (keep original names)
    for col in ADDITIONAL_MONITORING_FEATURES:
        if col in df.columns:
            # Clean the column name for pandas compatibility
            clean_name = col.replace(".", "_")
            select_exprs.append(pl.col(col).alias(clean_name))
    
    # Add customer_segment directly if exists
    if "customer_segment" in df.columns:
        select_exprs.append(pl.col("customer_segment"))
    
    # Add fraud flag for reference
    if "fraud_flag" in df.columns:
        select_exprs.append(pl.col("fraud_flag").is_not_null().alias("is_fraud"))
    
    df_selected = df.select(select_exprs)
    
    # Convert to pandas for Evidently
    pdf = df_selected.to_pandas()
    
    # Convert boolean columns to int for better drift detection
    for col in pdf.columns:
        if pdf[col].dtype == bool:
            pdf[col] = pdf[col].astype(int)
    
    return pdf


def split_by_time_windows(
    df: pd.DataFrame,
    window_size_days: int = 7,
    n_windows: int = 10
) -> List[Tuple[datetime, datetime, pd.DataFrame]]:
    """Split data into time windows for drift comparison."""
    df = df.copy()
    df['submission_at'] = pd.to_datetime(df['submission_at'])
    
    max_date = df['submission_at'].max()
    windows = []
    
    for i in range(n_windows):
        end_date = max_date - timedelta(days=i * window_size_days)
        start_date = end_date - timedelta(days=window_size_days)
        
        window_df = df[
            (df['submission_at'] >= start_date) & 
            (df['submission_at'] < end_date)
        ]
        
        if len(window_df) > 100:  # Minimum samples for meaningful drift detection
            windows.append((start_date, end_date, window_df))
    
    # Reverse to chronological order
    windows.reverse()
    
    logger.info(f"Created {len(windows)} time windows")
    for i, (start, end, w) in enumerate(windows):
        logger.info(f"  Window {i}: {start.date()} to {end.date()} ({len(w):,} samples)")
    
    return windows


# =============================================================================
# DRIFT DETECTION
# =============================================================================

def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Slight change
    PSI >= 0.2: Significant change
    """
    # Handle categorical data
    if reference.dtype == 'object' or str(reference.dtype) == 'category':
        categories = set(reference.dropna().unique()) | set(current.dropna().unique())
        ref_counts = reference.value_counts(normalize=True)
        cur_counts = current.value_counts(normalize=True)
        
        psi = 0.0
        for cat in categories:
            ref_pct = ref_counts.get(cat, 0.0001)
            cur_pct = cur_counts.get(cat, 0.0001)
            # Avoid log(0)
            ref_pct = max(ref_pct, 0.0001)
            cur_pct = max(cur_pct, 0.0001)
            psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        return psi
    
    # Handle numerical data
    reference = reference.dropna()
    current = current.dropna()
    
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    
    # Create bins from reference distribution
    _, bin_edges = np.histogram(reference, bins=bins)
    
    # Compute histograms
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    
    # Normalize
    ref_pct = ref_hist / len(reference)
    cur_pct = cur_hist / len(current)
    
    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
    
    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return psi


def detect_drift_between_windows(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str]
) -> Dict:
    """Detect drift between two windows for specified features."""
    results = {
        "features": {},
        "drifted_features": [],
        "drift_share": 0.0,
    }
    
    valid_features = [f for f in features if f in reference_df.columns and f in current_df.columns]
    
    for feature in valid_features:
        ref_col = reference_df[feature]
        cur_col = current_df[feature]
        
        # Compute PSI
        psi = compute_psi(ref_col, cur_col)
        
        # Determine drift status
        drifted = psi >= PSI_THRESHOLD
        
        results["features"][feature] = {
            "psi": round(psi, 4),
            "drifted": drifted,
            "ref_mean": float(ref_col.mean()) if ref_col.dtype in ['int64', 'float64'] else None,
            "cur_mean": float(cur_col.mean()) if cur_col.dtype in ['int64', 'float64'] else None,
        }
        
        if drifted:
            results["drifted_features"].append(feature)
    
    results["drift_share"] = len(results["drifted_features"]) / len(valid_features) if valid_features else 0
    
    return results


def run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: Path
) -> Optional[Dict]:
    """Run Evidently AI drift report."""
    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently not available, skipping report generation")
        return None
    
    # Select only numeric and categorical columns
    valid_cols = []
    for col in reference_df.columns:
        if col in ['submission_at', 'insertion_id']:
            continue
        if reference_df[col].dtype in ['int64', 'float64', 'bool', 'object']:
            valid_cols.append(col)
    
    ref_clean = reference_df[valid_cols].copy()
    cur_clean = current_df[valid_cols].copy()
    
    # Create report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    
    report.run(reference_data=ref_clean, current_data=cur_clean)
    
    # Save HTML report
    report.save_html(str(output_path))
    logger.info(f"Evidently report saved to {output_path}")
    
    # Extract results
    result = report.as_dict()
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_drift_timeseries(
    windows: List[Tuple[datetime, datetime, pd.DataFrame]],
    reference_window_idx: int = 0
) -> None:
    """Plot drift over time for key features."""
    if len(windows) < 2:
        logger.warning("Not enough windows for drift timeseries")
        return
    
    reference_df = windows[reference_window_idx][2]
    
    # Track drift metrics over time
    dates = []
    drift_shares = []
    seon_score_means = []
    price_means = []
    
    # Key features to track
    features_to_monitor = TRAINING_FEATURES + BOOLEAN_FEATURES
    features_to_monitor = [f for f in features_to_monitor if f in reference_df.columns]
    
    for start, end, window_df in windows[1:]:
        dates.append(end)
        
        # Compute drift
        drift_result = detect_drift_between_windows(reference_df, window_df, features_to_monitor)
        drift_shares.append(drift_result["drift_share"])
        
        # Track Seon score if available
        seon_col = "auto_approval_criteria_meta_seonFraudScore"
        if seon_col in window_df.columns:
            seon_score_means.append(window_df[seon_col].mean())
        else:
            seon_score_means.append(None)
        
        # Track price
        if "price_rent_gross" in window_df.columns:
            price_means.append(window_df["price_rent_gross"].mean())
        else:
            price_means.append(None)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Drift share over time
    ax1 = axes[0]
    ax1.plot(dates, drift_shares, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax1.axhline(y=DRIFT_SHARE_THRESHOLD, color='#e74c3c', linestyle='--', alpha=0.5, 
                label=f'Alert Threshold ({DRIFT_SHARE_THRESHOLD*100:.0f}%)')
    ax1.fill_between(dates, 0, drift_shares, alpha=0.3, color='#e74c3c')
    ax1.set_ylabel('Drift Share')
    ax1.set_title('Feature Drift Over Time (vs Reference Window)', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 2. Seon score drift
    ax2 = axes[1]
    seon_valid = [(d, s) for d, s in zip(dates, seon_score_means) if s is not None]
    if seon_valid:
        seon_dates, seon_vals = zip(*seon_valid)
        ax2.plot(seon_dates, seon_vals, 's-', color='#3498db', linewidth=2, markersize=8)
        ax2.fill_between(seon_dates, 0, seon_vals, alpha=0.3, color='#3498db')
    ax2.set_ylabel('Mean Seon Score')
    ax2.set_title('Seon Fraud Score Distribution', fontweight='bold')
    
    # 3. Price drift
    ax3 = axes[2]
    price_valid = [(d, p) for d, p in zip(dates, price_means) if p is not None]
    if price_valid:
        price_dates, price_vals = zip(*price_valid)
        ax3.plot(price_dates, price_vals, '^-', color='#27ae60', linewidth=2, markersize=8)
        ax3.fill_between(price_dates, 0, price_vals, alpha=0.3, color='#27ae60')
    ax3.set_ylabel('Mean Rent Price')
    ax3.set_title('Price Distribution', fontweight='bold')
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "drift_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Drift timeseries plot saved to {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run drift detection POC."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 8: FEATURE DRIFT DETECTION POC")
    logger.info("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    raw_df = load_raw_data()
    pdf = prepare_monitoring_data(raw_df)
    
    logger.info(f"Prepared {len(pdf):,} records with {len(pdf.columns)} monitoring columns")
    
    # Split into time windows
    windows = split_by_time_windows(pdf, window_size_days=7, n_windows=12)
    
    if len(windows) < 2:
        logger.error("Not enough data for drift detection")
        return
    
    # Use first window as reference
    ref_start, ref_end, reference_df = windows[0]
    logger.info(f"\nReference window: {ref_start.date()} to {ref_end.date()}")
    
    # Use latest window as current
    cur_start, cur_end, current_df = windows[-1]
    logger.info(f"Current window: {cur_start.date()} to {cur_end.date()}")
    
    # Define features to monitor
    all_features = TRAINING_FEATURES + BOOLEAN_FEATURES
    available_features = [f for f in all_features if f in reference_df.columns]
    
    logger.info(f"\nMonitoring {len(available_features)} features for drift")
    
    # Compute drift
    logger.info("\n" + "=" * 70)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("=" * 70)
    
    drift_results = detect_drift_between_windows(reference_df, current_df, available_features)
    
    # Print results
    print(f"\nDrift Share: {drift_results['drift_share']*100:.1f}%")
    print(f"Drifted Features: {len(drift_results['drifted_features'])} / {len(available_features)}")
    
    if drift_results['drifted_features']:
        print("\nDrifted Features (PSI >= 0.1):")
        for feat in drift_results['drifted_features']:
            info = drift_results['features'][feat]
            print(f"  ⚠️  {feat}: PSI = {info['psi']:.4f}")
    
    # Features with no drift
    stable_features = [f for f in available_features if f not in drift_results['drifted_features']]
    print(f"\nStable Features: {len(stable_features)}")
    
    # Alert check
    print("\n" + "=" * 70)
    if drift_results['drift_share'] >= DRIFT_SHARE_THRESHOLD:
        print("🚨 ALERT: Significant drift detected!")
        print(f"   {drift_results['drift_share']*100:.1f}% of features have drifted")
        print("   Recommendation: Investigate and consider retraining")
    else:
        print("✅ No significant drift detected")
        print(f"   Only {drift_results['drift_share']*100:.1f}% of features show drift")
    print("=" * 70)
    
    # Save JSON summary
    summary = {
        "reference_window": {
            "start": ref_start.isoformat(),
            "end": ref_end.isoformat(),
            "samples": len(reference_df),
        },
        "current_window": {
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
            "samples": len(current_df),
        },
        "drift_share": drift_results['drift_share'],
        "drifted_features": drift_results['drifted_features'],
        "feature_psi": {k: v['psi'] for k, v in drift_results['features'].items()},
        "alert_triggered": drift_results['drift_share'] >= DRIFT_SHARE_THRESHOLD,
        "generated_at": datetime.now().isoformat(),
    }
    
    summary_path = OUTPUT_DIR / "drift_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")
    
    # Generate Evidently report
    if EVIDENTLY_AVAILABLE:
        logger.info("\nGenerating Evidently report...")
        evidently_path = OUTPUT_DIR / "drift_report.html"
        run_evidently_report(reference_df, current_df, evidently_path)
    
    # Plot drift timeseries
    logger.info("\nGenerating drift timeseries plot...")
    plot_drift_timeseries(windows)
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 8 POC COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

