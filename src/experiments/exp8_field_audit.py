"""
Experiment 8C: Full Field Audit

Systematically analyze ALL 292 fields from raw_insertions.parquet to:
1. Identify fields with fraud correlation that we're NOT using
2. Rank feature candidates by predictive value
3. Recommend which fields to add to the model

Usage:
    python -m src.experiments.exp8_field_audit

Output:
    - artifacts/drift_monitoring/field_audit_report.json
    - artifacts/drift_monitoring/field_audit_summary.png
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from src.utils.hydra_utils import resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"
NODES_LISTING = ARTIFACTS_DIR / "nodes_listing.parquet"
OUTPUT_DIR = ARTIFACTS_DIR / "drift_monitoring"

# Known mappings from raw to nodes_listing
RAW_TO_LISTING_MAPPING = {
    "listing.characteristics.numberOfRooms": "rooms",
    "listing.characteristics.livingSpace": "living_space",
    "listing.characteristics.yearBuilt": "year_built",
    "listing.characteristics.floor": "floor",
    "listing.characteristics.numberOfFloors": "num_floors",
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
    "listing.characteristics.hasCableTv": "has_cable_tv",
    "listing.characteristics.hasFireplace": "has_fireplace",
    "listing.characteristics.isMinergieGeneral": "is_minergie_general",
    "listing.characteristics.isMinergieCertified": "is_minergie_certified",
    "listing.characteristics.isSmokingAllowed": "is_smoking_allowed",
    "listing.characteristics.hasSwimmingPool": "has_swimming_pool",
    "listing.prices.rent.gross": "price_rent_gross",
    "listing.prices.rent.net": "price_rent_net",
    "listing.prices.buy.price": "price_buy",
    "listing.offerType": "offer_type",
    "listing.lister.billing.payment.paymentType": "payment_type",
    "listing.address.geoCoordinates.latitude": "latitude",
    "listing.address.geoCoordinates.longitude": "longitude",
    "listing.address.region": "region",
    "listing.address.postalCode": "zip_code",
    "listing.localization.primary": "language",
    "bundle.tier": "bundle_tier",
    "bundle.period": "bundle_period",
    "auto_approval_criteria.criteria.seonApproved": "seon_approved",
}

# Fields to EXCLUDE from feature consideration (metadata, IDs, leakage)
EXCLUDE_FIELDS = {
    "object_reference",
    "owner_id", 
    "user_id",
    "submission_at",
    "fraud_flag",
    "is_fraud",  # Target variable - NOT a feature!
    "first_published_date",
    "listing_created_at",
    "account_created_at",
    # Hash fields (identifiers, not features)
    "contact_emails_hash",
    "user_ip_address_hash",
    # Seon score - should not be a feature (it's the baseline)
    "auto_approval_criteria.meta.seonFraudScore",
    # Legacy IDs
    "listing.legacy.personId",
    "listing.legacy.ppaPersonId",
}


def load_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load raw and nodes_listing data."""
    logger.info(f"Loading data...")
    raw = pl.read_parquet(RAW_INSERTIONS)
    listing = pl.read_parquet(NODES_LISTING)
    
    # Add fraud flag
    raw = raw.with_columns(
        pl.col("fraud_flag").is_not_null().alias("is_fraud")
    )
    
    logger.info(f"Raw: {len(raw):,} rows, {len(raw.columns)} columns")
    logger.info(f"Listing: {len(listing):,} rows, {len(listing.columns)} columns")
    
    return raw, listing


def analyze_field(raw: pl.DataFrame, col: str) -> Optional[Dict[str, Any]]:
    """Analyze a single field for fraud correlation and coverage."""
    if col in EXCLUDE_FIELDS:
        return None
    
    # Check coverage
    null_count = raw[col].null_count()
    coverage = 1 - (null_count / len(raw))
    
    if coverage < 0.1:  # Skip fields with <10% coverage
        return None
    
    dtype = raw[col].dtype
    
    result = {
        "field": col,
        "dtype": str(dtype),
        "coverage": coverage,
        "fraud_correlation": None,
        "fraud_rate_diff": None,
        "unique_values": None,
        "is_in_listing": col in RAW_TO_LISTING_MAPPING or any(
            col.endswith(v) for v in RAW_TO_LISTING_MAPPING.values()
        ),
    }
    
    # Numerical fields - compute correlation
    if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Int16, pl.Int8]:
        valid = raw.filter(pl.col(col).is_not_null())
        if len(valid) > 100:
            try:
                corr = valid.select([
                    pl.corr(col, "is_fraud").alias("corr")
                ]).item()
                result["fraud_correlation"] = float(corr) if corr is not None else None
            except Exception:
                pass
        
        # Unique values for categoricals stored as int
        n_unique = raw[col].n_unique()
        result["unique_values"] = n_unique
    
    # Boolean fields - compute fraud rate difference
    elif dtype == pl.Boolean:
        try:
            stats = raw.group_by(col).agg([
                pl.len().alias("count"),
                pl.col("is_fraud").mean().alias("fraud_rate")
            ]).filter(pl.col("count") > 100)
            
            if len(stats) >= 2:
                rates = stats["fraud_rate"].to_list()
                if None not in rates:
                    result["fraud_rate_diff"] = abs(rates[0] - rates[1]) if len(rates) == 2 else None
                    result["fraud_correlation"] = result["fraud_rate_diff"]
        except Exception:
            pass
        result["unique_values"] = 2
    
    # Categorical (string) fields
    elif dtype == pl.Utf8:
        n_unique = raw[col].n_unique()
        result["unique_values"] = n_unique
        
        # Only analyze if reasonable cardinality
        if n_unique <= 50:
            try:
                stats = raw.group_by(col).agg([
                    pl.len().alias("count"),
                    pl.col("is_fraud").mean().alias("fraud_rate")
                ]).filter(pl.col("count") > 50)
                
                if len(stats) >= 2:
                    rates = [r for r in stats["fraud_rate"].to_list() if r is not None]
                    if len(rates) >= 2:
                        result["fraud_rate_diff"] = max(rates) - min(rates)
                        result["fraud_correlation"] = result["fraud_rate_diff"]
            except Exception:
                pass
    
    return result


def categorize_fields(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize fields by their status and potential."""
    categories = {
        "already_used": [],      # In nodes_listing
        "high_value_unused": [], # High fraud correlation, not used
        "medium_value_unused": [], # Medium correlation, not used
        "low_coverage": [],      # Good correlation but low coverage
        "no_signal": [],         # No fraud correlation
    }
    
    for r in results:
        if r is None:
            continue
        
        corr = r.get("fraud_correlation")
        coverage = r.get("coverage", 0)
        is_used = r.get("is_in_listing", False)
        
        # Check if already mapped
        if r["field"] in RAW_TO_LISTING_MAPPING:
            categories["already_used"].append(r)
        elif corr is not None and abs(corr) >= 0.05:
            if coverage >= 0.5:
                if abs(corr) >= 0.1:
                    categories["high_value_unused"].append(r)
                else:
                    categories["medium_value_unused"].append(r)
            else:
                categories["low_coverage"].append(r)
        else:
            categories["no_signal"].append(r)
    
    # Sort by correlation
    for cat in categories:
        categories[cat].sort(
            key=lambda x: abs(x.get("fraud_correlation") or 0), 
            reverse=True
        )
    
    return categories


def generate_recommendation(categories: Dict[str, List[Dict]]) -> Dict:
    """Generate recommendation on whether to add new features."""
    high_value = categories["high_value_unused"]
    medium_value = categories["medium_value_unused"]
    
    recommendation = {
        "should_add_features": False,
        "priority_fields": [],
        "optional_fields": [],
        "estimated_improvement": "unknown",
        "effort_required": "medium",
        "reasoning": "",
    }
    
    # Priority fields (correlation >= 0.1)
    for field in high_value[:5]:
        recommendation["priority_fields"].append({
            "field": field["field"],
            "correlation": field.get("fraud_correlation"),
            "coverage": field.get("coverage"),
        })
    
    # Optional fields (correlation 0.05-0.1)
    for field in medium_value[:5]:
        recommendation["optional_fields"].append({
            "field": field["field"],
            "correlation": field.get("fraud_correlation"),
            "coverage": field.get("coverage"),
        })
    
    # Decision logic
    if len(high_value) >= 2:
        recommendation["should_add_features"] = True
        recommendation["estimated_improvement"] = "potentially significant (2-5% AUC-PR)"
        recommendation["reasoning"] = (
            f"Found {len(high_value)} high-value unused fields with correlation >= 0.1. "
            f"Top field has {high_value[0].get('fraud_correlation', 0):.2f} correlation."
        )
    elif len(high_value) >= 1:
        recommendation["should_add_features"] = True
        recommendation["estimated_improvement"] = "modest (1-2% AUC-PR)"
        recommendation["reasoning"] = (
            f"Found {len(high_value)} high-value field. Worth testing."
        )
    else:
        recommendation["should_add_features"] = False
        recommendation["estimated_improvement"] = "minimal"
        recommendation["reasoning"] = (
            "No high-value unused fields found. Current feature set is comprehensive."
        )
    
    return recommendation


def plot_audit_summary(categories: Dict[str, List[Dict]], output_path: Path) -> None:
    """Create visualization of field audit results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Field coverage by category
    ax1 = axes[0, 0]
    cat_sizes = {k: len(v) for k, v in categories.items()}
    colors = ['#27ae60', '#e74c3c', '#f39c12', '#95a5a6', '#bdc3c7']
    labels = ['Already Used', 'High Value\nUnused', 'Medium Value\nUnused', 'Low Coverage', 'No Signal']
    sizes = [cat_sizes.get(k, 0) for k in ['already_used', 'high_value_unused', 'medium_value_unused', 'low_coverage', 'no_signal']]
    
    bars = ax1.bar(labels, sizes, color=colors)
    ax1.set_ylabel('Number of Fields')
    ax1.set_title('Field Categorization (292 Total)', fontweight='bold')
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(size), ha='center', va='bottom', fontsize=10)
    
    # 2. Top unused fields by correlation
    ax2 = axes[0, 1]
    high_value = categories.get('high_value_unused', [])[:10]
    if high_value:
        fields = [f["field"].split(".")[-1][:20] for f in high_value]
        corrs = [abs(f.get("fraud_correlation") or 0) for f in high_value]
        
        y_pos = range(len(fields))
        bars = ax2.barh(y_pos, corrs, color='#e74c3c')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(fields)
        ax2.set_xlabel('|Fraud Correlation|')
        ax2.set_title('Top 10 High-Value Unused Fields', fontweight='bold')
        ax2.axvline(x=0.1, color='#27ae60', linestyle='--', label='Threshold (0.1)')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No high-value unused fields', ha='center', va='center')
        ax2.set_title('Top High-Value Unused Fields', fontweight='bold')
    
    # 3. Coverage vs Correlation scatter
    ax3 = axes[1, 0]
    all_results = []
    for cat, items in categories.items():
        for item in items:
            if item.get("fraud_correlation") is not None:
                all_results.append({
                    "coverage": item.get("coverage", 0),
                    "correlation": abs(item.get("fraud_correlation", 0)),
                    "category": cat,
                })
    
    if all_results:
        df_plot = pd.DataFrame(all_results)
        color_map = {
            'already_used': '#27ae60',
            'high_value_unused': '#e74c3c',
            'medium_value_unused': '#f39c12',
            'low_coverage': '#95a5a6',
            'no_signal': '#bdc3c7',
        }
        for cat, color in color_map.items():
            mask = df_plot['category'] == cat
            ax3.scatter(df_plot.loc[mask, 'coverage'], 
                       df_plot.loc[mask, 'correlation'],
                       c=color, alpha=0.6, label=cat.replace('_', ' ').title(), s=50)
        
        ax3.axhline(y=0.1, color='#e74c3c', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.05, color='#f39c12', linestyle='--', alpha=0.5)
        ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Coverage')
        ax3.set_ylabel('|Fraud Correlation|')
        ax3.set_title('All Fields: Coverage vs Fraud Signal', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
    
    # 4. Recommendation summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    n_high = len(categories.get('high_value_unused', []))
    n_medium = len(categories.get('medium_value_unused', []))
    n_used = len(categories.get('already_used', []))
    
    if n_high > 0:
        top_field = categories['high_value_unused'][0]
        top_name = top_field['field'].split('.')[-1]
        top_corr = top_field.get('fraud_correlation', 0)
    else:
        top_name = "N/A"
        top_corr = 0
    
    summary_text = f"""
FIELD AUDIT SUMMARY
{'='*40}

Total fields analyzed: 292
Already in use:        {n_used}
High-value unused:     {n_high}
Medium-value unused:   {n_medium}

TOP UNUSED FIELD:
  {top_name}
  Correlation: {top_corr:.3f}

RECOMMENDATION:
  {'✅ ADD NEW FEATURES' if n_high >= 2 else '⚠️ OPTIONAL' if n_high >= 1 else '❌ NO ACTION NEEDED'}
  
  Priority fields to add: {n_high}
  Expected improvement: {'2-5%' if n_high >= 2 else '1-2%' if n_high >= 1 else 'minimal'}
{'='*40}
"""
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Audit summary saved to {output_path}")
    plt.close()


def main():
    """Run full field audit."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 8C: FULL FIELD AUDIT")
    logger.info("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    raw, listing = load_data()
    
    # Analyze all fields
    logger.info(f"\nAnalyzing {len(raw.columns)} fields...")
    results = []
    for col in raw.columns:
        result = analyze_field(raw, col)
        if result:
            results.append(result)
    
    logger.info(f"Analyzed {len(results)} fields (excluded {len(raw.columns) - len(results)} metadata/low-coverage)")
    
    # Categorize
    categories = categorize_fields(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("FIELD AUDIT RESULTS")
    print("=" * 70)
    
    print(f"\nField Categories:")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)} fields")
    
    print("\n" + "-" * 70)
    print("HIGH-VALUE UNUSED FIELDS (correlation >= 0.1, coverage >= 50%)")
    print("-" * 70)
    
    for item in categories["high_value_unused"][:10]:
        print(f"  {item['field']}")
        print(f"    Correlation: {item.get('fraud_correlation', 0):.4f}")
        print(f"    Coverage: {item.get('coverage', 0)*100:.1f}%")
        print(f"    Type: {item.get('dtype')}")
        print()
    
    print("-" * 70)
    print("MEDIUM-VALUE UNUSED FIELDS (correlation 0.05-0.1)")
    print("-" * 70)
    
    for item in categories["medium_value_unused"][:5]:
        print(f"  {item['field']}")
        print(f"    Correlation: {item.get('fraud_correlation', 0):.4f}")
        print(f"    Coverage: {item.get('coverage', 0)*100:.1f}%")
        print()
    
    # Generate recommendation
    recommendation = generate_recommendation(categories)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\nShould add new features: {'✅ YES' if recommendation['should_add_features'] else '❌ NO'}")
    print(f"Estimated improvement: {recommendation['estimated_improvement']}")
    print(f"Reasoning: {recommendation['reasoning']}")
    
    if recommendation["priority_fields"]:
        print("\nPriority fields to add:")
        for f in recommendation["priority_fields"]:
            print(f"  - {f['field']} (corr: {f['correlation']:.3f}, coverage: {f['coverage']*100:.0f}%)")
    
    # Save report
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_fields": len(raw.columns),
        "analyzed_fields": len(results),
        "categories": {k: len(v) for k, v in categories.items()},
        "high_value_unused": categories["high_value_unused"][:10],
        "medium_value_unused": categories["medium_value_unused"][:10],
        "recommendation": recommendation,
    }
    
    report_path = OUTPUT_DIR / "field_audit_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")
    
    # Generate visualization
    plot_path = OUTPUT_DIR / "field_audit_summary.png"
    plot_audit_summary(categories, plot_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("FIELD AUDIT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

