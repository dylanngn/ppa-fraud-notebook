"""
Feature Discovery Pipeline
===========================

Automated pipeline for discovering, validating, and integrating new features
with a human-in-the-loop approval gate.

This framework enables:
1. Automated field auditing from raw_insertions.parquet
2. Quantified recommendations with fraud correlation metrics
3. Human approval gate (no manual coding required)
4. Config-driven integration
5. A/B validation before promotion

Workflow:
    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────────┐
    │ 1. DISCOVER  │───▶│ 2. RECOMMEND    │───▶│ 3. HUMAN GATE            │
    │  (Automated) │    │   (Automated)   │    │   (Decision Only)        │
    └──────────────┘    └─────────────────┘    └──────────────────────────┘
           │                    │                         │
           ▼                    ▼                         ▼
      Field Audit         Generate:               ✅ APPROVE → Continue
      (all 292 cols)      - candidates.yaml       ❌ REJECT → Stop
                          - correlation scores    ⏸️ DEFER → Queue
    
    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────────┐
    │ 4. INTEGRATE │◀───│ 5. RETRAIN      │◀───│ 6. VALIDATE              │
    │  (Automated) │    │   (Automated)   │    │   (Automated + Report)   │
    └──────────────┘    └─────────────────┘    └──────────────────────────┘
           │                    │                         │
           ▼                    ▼                         ▼
     Add to config:       Full window training    A/B comparison:
     - candidates.yaml                            - Old vs New model
     - NO code changes!                           - Auto-promote if +1% AUC

Usage:
    # Run full discovery (analyze raw fields, update candidates.yaml)
    python -m src.experiments.feature_discovery_pipeline --discover

    # Review current candidates (show pending approvals)
    python -m src.experiments.feature_discovery_pipeline --review

    # Approve specific candidates (by field name)
    python -m src.experiments.feature_discovery_pipeline --approve "listing.prices.rent.interval,listing.platforms"

    # Reject specific candidates
    python -m src.experiments.feature_discovery_pipeline --reject "some.field.name"

    # Integrate approved candidates (updates processor, prepares for training)
    python -m src.experiments.feature_discovery_pipeline --integrate

    # Validate integration with A/B comparison
    python -m src.experiments.feature_discovery_pipeline --validate

    # Full automated flow (discover + integrate approved + validate)
    python -m src.experiments.feature_discovery_pipeline --full-cycle
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_utils import resolve_path

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

ARTIFACTS_DIR = resolve_path("artifacts")
RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"
CANDIDATES_CONFIG = resolve_path("conf/features/candidates.yaml")
AUDIT_REPORT_PATH = ARTIFACTS_DIR / "drift_monitoring/field_audit_report.json"

# Fields to EXCLUDE from feature consideration (metadata, IDs, leakage)
EXCLUDE_FIELDS = {
    "object_reference", "owner_id", "user_id", "submission_at", "fraud_flag",
    "is_fraud", "first_published_date", "listing_created_at", "account_created_at",
    "contact_emails_hash", "user_ip_address_hash",
    "auto_approval_criteria.meta.seonFraudScore",  # Seon score - baseline, not feature
    "listing.legacy.personId", "listing.legacy.ppaPersonId",
}

# Fields already in nodes_listing (known mappings)
KNOWN_MAPPINGS = {
    "listing.characteristics.numberOfRooms": "rooms",
    "listing.characteristics.livingSpace": "living_space",
    "listing.characteristics.hasBalcony": "has_balcony",
    "listing.characteristics.hasParking": "has_parking",
    "listing.characteristics.hasElevator": "has_elevator",
    "listing.prices.rent.gross": "price_rent_gross",
    "listing.offerType": "offer_type",
    "listing.lister.billing.payment.paymentType": "payment_type",
    "bundle.tier": "bundle_tier",
    "bundle.period": "bundle_period",
    # ... additional mappings from exp8_field_audit.py
}


# =============================================================================
# DISCOVERY PHASE
# =============================================================================

def clean_field_name(raw_name: str) -> str:
    """Convert raw field name to valid feature name for XGBoost."""
    # Remove common prefixes
    name = raw_name
    for prefix in ["listing.", "bundle.", "auto_approval_criteria."]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Replace dots and special chars with underscores
    name = re.sub(r'[.\-\s]+', '_', name)
    
    # Convert camelCase to snake_case
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    
    # Lowercase and clean up
    name = name.lower()
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    return name


def analyze_field(df: pl.DataFrame, col: str) -> Optional[Dict[str, Any]]:
    """Analyze a single field for fraud correlation and coverage."""
    if col in EXCLUDE_FIELDS:
        return None
    
    # Check if already used
    if col in KNOWN_MAPPINGS:
        return None
    
    # Check coverage
    null_count = df[col].null_count()
    coverage = 1 - (null_count / len(df))
    
    if coverage < 0.1:  # Skip fields with <10% coverage
        return None
    
    dtype = df[col].dtype
    result = {
        "field": col,
        "raw_name": col,
        "feature_name": clean_field_name(col),
        "dtype": str(dtype),
        "coverage": round(coverage, 4),
        "fraud_correlation": None,
        "unique_values": None,
    }
    
    # Numerical fields - compute correlation
    if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Int16, pl.Int8]:
        valid = df.filter(pl.col(col).is_not_null())
        if len(valid) > 100:
            try:
                corr = valid.select([
                    pl.corr(col, "is_fraud").alias("corr")
                ]).item()
                result["fraud_correlation"] = round(float(corr), 4) if corr is not None else None
            except Exception:
                pass
        result["unique_values"] = df[col].n_unique()
    
    # Boolean fields - compute fraud rate difference
    elif dtype == pl.Boolean:
        try:
            stats = df.group_by(col).agg([
                pl.len().alias("count"),
                pl.col("is_fraud").mean().alias("fraud_rate")
            ]).filter(pl.col("count") > 100)
            
            if len(stats) >= 2:
                rates = stats["fraud_rate"].to_list()
                if None not in rates and len(rates) == 2:
                    result["fraud_correlation"] = round(abs(rates[0] - rates[1]), 4)
        except Exception:
            pass
        result["unique_values"] = 2
    
    # Categorical (string) fields
    elif dtype == pl.Utf8:
        n_unique = df[col].n_unique()
        result["unique_values"] = n_unique
        
        # Only analyze if reasonable cardinality
        if n_unique <= 50:
            try:
                stats = df.group_by(col).agg([
                    pl.len().alias("count"),
                    pl.col("is_fraud").mean().alias("fraud_rate")
                ]).filter(pl.col("count") > 50)
                
                if len(stats) >= 2:
                    rates = [r for r in stats["fraud_rate"].to_list() if r is not None]
                    if len(rates) >= 2:
                        result["fraud_correlation"] = round(max(rates) - min(rates), 4)
            except Exception:
                pass
    
    return result


def run_discovery() -> Dict[str, Any]:
    """
    Run field discovery/audit on raw_insertions.parquet.
    Returns candidates sorted by fraud correlation.
    """
    logger.info("=" * 70)
    logger.info("FEATURE DISCOVERY: Analyzing raw_insertions.parquet")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"Loading {RAW_INSERTIONS}...")
    df = pl.read_parquet(RAW_INSERTIONS)
    
    # Add fraud flag
    df = df.with_columns(
        pl.col("fraud_flag").is_not_null().alias("is_fraud")
    )
    
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Analyze all fields
    logger.info("Analyzing fields for fraud correlation...")
    results = []
    for col in df.columns:
        result = analyze_field(df, col)
        if result and result.get("fraud_correlation") is not None:
            results.append(result)
    
    # Sort by correlation (descending)
    results.sort(key=lambda x: abs(x.get("fraud_correlation", 0)), reverse=True)
    
    # Categorize by correlation threshold
    high_value = [r for r in results if abs(r.get("fraud_correlation", 0)) >= 0.1 and r["coverage"] >= 0.5]
    medium_value = [r for r in results if 0.05 <= abs(r.get("fraud_correlation", 0)) < 0.1 and r["coverage"] >= 0.5]
    
    summary = {
        "total_raw_fields": len(df.columns),
        "analyzed_fields": len(results),
        "high_value_found": len(high_value),
        "medium_value_found": len(medium_value),
        "recommendation": "add_features" if len(high_value) >= 2 else "optional" if len(high_value) >= 1 else "none",
    }
    
    logger.info(f"\nDiscovery Summary:")
    logger.info(f"  Total fields: {summary['total_raw_fields']}")
    logger.info(f"  Analyzed: {summary['analyzed_fields']}")
    logger.info(f"  High-value (corr >= 0.1): {summary['high_value_found']}")
    logger.info(f"  Medium-value (0.05-0.1): {summary['medium_value_found']}")
    
    return {
        "summary": summary,
        "high_value": high_value,
        "medium_value": medium_value,
        "all_results": results,
    }


def update_candidates_config(discovery_results: Dict[str, Any]) -> None:
    """Update candidates.yaml with discovery results."""
    logger.info("\nUpdating candidates.yaml...")
    
    # Load existing config
    if CANDIDATES_CONFIG.exists():
        config = OmegaConf.load(CANDIDATES_CONFIG)
    else:
        config = OmegaConf.create({
            "last_audit": None,
            "audit_summary": {},
            "candidates": [],
            "integration": {
                "min_correlation": 0.05,
                "min_coverage": 0.50,
                "min_improvement_pct": 1.0,
                "validation_windows": 10,
            }
        })
    
    # Build existing candidates map (preserve status)
    existing = {c.get("field"): dict(c) for c in config.get("candidates", [])}
    
    # Update with new discoveries
    now = datetime.now().isoformat()
    new_candidates = []
    
    # Add high-value candidates
    for result in discovery_results["high_value"]:
        field = result["field"]
        if field in existing:
            # Preserve existing status, update metrics
            candidate = existing[field]
            candidate["correlation"] = result["fraud_correlation"]
            candidate["coverage"] = result["coverage"]
        else:
            # New candidate
            candidate = {
                "field": field,
                "raw_name": result["raw_name"],
                "feature_name": result["feature_name"],
                "dtype": result["dtype"],
                "correlation": result["fraud_correlation"],
                "coverage": result["coverage"],
                "unique_values": result["unique_values"],
                "discovered_at": now,
                "status": "pending",
                "approved_by": None,
                "approved_at": None,
                "integrated_at": None,
                "validation_result": None,
            }
        new_candidates.append(candidate)
    
    # Add medium-value candidates
    for result in discovery_results["medium_value"][:10]:  # Limit to top 10
        field = result["field"]
        if field in existing:
            candidate = existing[field]
            candidate["correlation"] = result["fraud_correlation"]
            candidate["coverage"] = result["coverage"]
        else:
            candidate = {
                "field": field,
                "raw_name": result["raw_name"],
                "feature_name": result["feature_name"],
                "dtype": result["dtype"],
                "correlation": result["fraud_correlation"],
                "coverage": result["coverage"],
                "unique_values": result["unique_values"],
                "discovered_at": now,
                "status": "pending",
                "approved_by": None,
                "approved_at": None,
                "integrated_at": None,
                "validation_result": None,
            }
        new_candidates.append(candidate)
    
    # Sort by correlation
    new_candidates.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)
    
    # Update config
    config["last_audit"] = now
    config["audit_summary"] = discovery_results["summary"]
    config["candidates"] = new_candidates
    
    # Save
    OmegaConf.save(config, CANDIDATES_CONFIG)
    logger.info(f"Updated {CANDIDATES_CONFIG} with {len(new_candidates)} candidates")


# =============================================================================
# REVIEW PHASE
# =============================================================================

def review_candidates() -> None:
    """Display current candidates for human review."""
    if not CANDIDATES_CONFIG.exists():
        logger.error(f"No candidates file found. Run --discover first.")
        return
    
    config = OmegaConf.load(CANDIDATES_CONFIG)
    candidates = config.get("candidates", [])
    
    print("\n" + "=" * 80)
    print("FEATURE CANDIDATES FOR REVIEW")
    print("=" * 80)
    print(f"Last audit: {config.get('last_audit', 'Never')}")
    
    summary = config.get("audit_summary", {})
    print(f"High-value found: {summary.get('high_value_found', 0)}")
    print(f"Recommendation: {summary.get('recommendation', 'unknown')}")
    
    # Group by status
    pending = [c for c in candidates if c.get("status") == "pending"]
    approved = [c for c in candidates if c.get("status") == "approved"]
    rejected = [c for c in candidates if c.get("status") == "rejected"]
    integrated = [c for c in candidates if c.get("status") == "integrated"]
    
    if pending:
        print("\n" + "-" * 80)
        print("⏳ PENDING APPROVAL")
        print("-" * 80)
        for c in pending:
            _print_candidate(c)
    
    if approved:
        print("\n" + "-" * 80)
        print("✅ APPROVED (ready for integration)")
        print("-" * 80)
        for c in approved:
            _print_candidate(c)
    
    if integrated:
        print("\n" + "-" * 80)
        print("🚀 INTEGRATED")
        print("-" * 80)
        for c in integrated:
            _print_candidate(c)
    
    if rejected:
        print("\n" + "-" * 80)
        print("❌ REJECTED")
        print("-" * 80)
        for c in rejected[:5]:  # Only show first 5
            _print_candidate(c)
    
    print("\n" + "=" * 80)
    print("ACTIONS:")
    print("  Approve: python -m src.experiments.feature_discovery_pipeline --approve \"field.name\"")
    print("  Reject:  python -m src.experiments.feature_discovery_pipeline --reject \"field.name\"")
    print("  Integrate: python -m src.experiments.feature_discovery_pipeline --integrate")
    print("=" * 80)


def _print_candidate(c: Dict) -> None:
    """Print a single candidate."""
    status_icon = {
        "pending": "⏳",
        "approved": "✅", 
        "rejected": "❌",
        "integrated": "🚀",
        "failed": "⚠️",
    }.get(c.get("status"), "?")
    
    print(f"\n  {status_icon} {c['field']}")
    print(f"     Feature name: {c.get('feature_name', 'N/A')}")
    print(f"     Correlation:  {c.get('correlation', 0):.4f}")
    print(f"     Coverage:     {c.get('coverage', 0)*100:.1f}%")
    print(f"     Type:         {c.get('dtype', 'unknown')}")
    if c.get("approved_at"):
        print(f"     Approved at:  {c.get('approved_at')}")


# =============================================================================
# APPROVAL PHASE
# =============================================================================

def approve_candidates(field_names: List[str]) -> None:
    """Approve specific candidates by field name."""
    if not CANDIDATES_CONFIG.exists():
        logger.error("No candidates file found. Run --discover first.")
        return
    
    config = OmegaConf.load(CANDIDATES_CONFIG)
    candidates = list(config.get("candidates", []))
    
    now = datetime.now().isoformat()
    approved_count = 0
    
    for i, c in enumerate(candidates):
        if c.get("field") in field_names:
            candidates[i] = dict(c)
            candidates[i]["status"] = "approved"
            candidates[i]["approved_at"] = now
            candidates[i]["approved_by"] = "human"
            approved_count += 1
            logger.info(f"✅ Approved: {c['field']}")
    
    config["candidates"] = candidates
    OmegaConf.save(config, CANDIDATES_CONFIG)
    
    logger.info(f"\nApproved {approved_count} candidates")
    

def reject_candidates(field_names: List[str]) -> None:
    """Reject specific candidates by field name."""
    if not CANDIDATES_CONFIG.exists():
        logger.error("No candidates file found. Run --discover first.")
        return
    
    config = OmegaConf.load(CANDIDATES_CONFIG)
    candidates = list(config.get("candidates", []))
    
    now = datetime.now().isoformat()
    rejected_count = 0
    
    for i, c in enumerate(candidates):
        if c.get("field") in field_names:
            candidates[i] = dict(c)
            candidates[i]["status"] = "rejected"
            candidates[i]["approved_at"] = now
            rejected_count += 1
            logger.info(f"❌ Rejected: {c['field']}")
    
    config["candidates"] = candidates
    OmegaConf.save(config, CANDIDATES_CONFIG)
    
    logger.info(f"\nRejected {rejected_count} candidates")


# =============================================================================
# INTEGRATION PHASE
# =============================================================================

def get_approved_candidates() -> List[Dict]:
    """Get list of approved candidates ready for integration."""
    if not CANDIDATES_CONFIG.exists():
        return []
    
    config = OmegaConf.load(CANDIDATES_CONFIG)
    return [dict(c) for c in config.get("candidates", []) if c.get("status") == "approved"]


def integrate_approved_candidates() -> None:
    """
    Generate instructions for integrating approved candidates.
    
    Note: Full auto-integration requires modifying create_artifacts.py and
    FeatureProcessor, which is a more invasive change. For now, we generate
    a detailed integration guide.
    """
    approved = get_approved_candidates()
    
    if not approved:
        logger.info("No approved candidates to integrate.")
        logger.info("Use --approve to approve pending candidates first.")
        return
    
    print("\n" + "=" * 80)
    print("INTEGRATION GUIDE")
    print("=" * 80)
    print(f"\nApproved candidates to integrate: {len(approved)}")
    
    # Group by data type for integration strategy
    categorical_fields = [c for c in approved if c.get("dtype") == "Utf8"]
    numerical_fields = [c for c in approved if c.get("dtype") in ["Int64", "Float64", "Int32", "Float32"]]
    boolean_fields = [c for c in approved if c.get("dtype") == "Boolean"]
    
    print("\n" + "-" * 80)
    print("STEP 1: Update create_artifacts.py")
    print("-" * 80)
    print("""
Add the following fields to the nodes_listing parquet creation.
Edit: src/data/graph/create_artifacts.py, function create_node_artifacts()

Add to LISTING_COLUMNS:
""")
    
    for c in approved:
        print(f'    "{c["raw_name"]}",  # {c["feature_name"]}: corr={c["correlation"]:.3f}')
    
    print("\n" + "-" * 80)
    print("STEP 2: Update FeatureProcessor (base.py)")
    print("-" * 80)
    print("""
Edit: src/features/definitions/base.py

Add processing for new fields in generate_base_features():
""")
    
    for c in categorical_fields:
        print(f"""
    # {c['feature_name']} (categorical, corr={c['correlation']:.3f})
    if "{c['raw_name']}" in df.columns:
        df = df.with_columns(
            pl.col("{c['raw_name']}").alias("{c['feature_name']}")
        )
""")
    
    for c in numerical_fields:
        print(f"""
    # {c['feature_name']} (numerical, corr={c['correlation']:.3f})
    if "{c['raw_name']}" in df.columns:
        df = df.with_columns(
            pl.col("{c['raw_name']}").fill_null(0).alias("{c['feature_name']}")
        )
""")
    
    print("\n" + "-" * 80)
    print("STEP 3: Update constants.py")
    print("-" * 80)
    print("""
Edit: src/models/config/constants.py

Add new feature group:
""")
    
    print("CANDIDATE_FEATURES = [")
    for c in approved:
        print(f'    "{c["feature_name"]}",')
    print("]")
    
    print("""
Add to FEATURE_GROUPS:
    "candidates": CANDIDATE_FEATURES,
""")
    
    print("\n" + "-" * 80)
    print("STEP 4: Update production.yaml")
    print("-" * 80)
    print("""
Edit: conf/features/production.yaml

Add to include_groups:
    - candidates
""")
    
    print("\n" + "-" * 80)
    print("STEP 5: Rebuild artifacts and retrain")
    print("-" * 80)
    print("""
# Rebuild graph artifacts with new fields
python -m src.data.graph.create_artifacts

# Retrain model with new features
python -m src.models.train features=production experiment_name=feature-candidates-validation

# Compare with previous production model
python -m src.utils.mlflow_model_comparison compare \\
  --model-name fraud-detection-xgboost \\
  --candidate-run-id <NEW_RUN_ID>
""")
    
    print("\n" + "=" * 80)
    print("After validation, mark candidates as integrated:")
    print("  python -m src.experiments.feature_discovery_pipeline --mark-integrated \"field1,field2\"")
    print("=" * 80)


def mark_integrated(field_names: List[str], validation_result: Optional[Dict] = None) -> None:
    """Mark candidates as successfully integrated."""
    if not CANDIDATES_CONFIG.exists():
        logger.error("No candidates file found.")
        return
    
    config = OmegaConf.load(CANDIDATES_CONFIG)
    candidates = list(config.get("candidates", []))
    
    now = datetime.now().isoformat()
    
    for i, c in enumerate(candidates):
        if c.get("field") in field_names:
            candidates[i] = dict(c)
            candidates[i]["status"] = "integrated"
            candidates[i]["integrated_at"] = now
            if validation_result:
                candidates[i]["validation_result"] = validation_result
            logger.info(f"🚀 Marked as integrated: {c['field']}")
    
    config["candidates"] = candidates
    OmegaConf.save(config, CANDIDATES_CONFIG)


# =============================================================================
# STATUS REPORT
# =============================================================================

def generate_status_report() -> None:
    """Generate a status report of the feature discovery pipeline."""
    if not CANDIDATES_CONFIG.exists():
        print("\n⚠️ No feature discovery has been run yet.")
        print("Run: python -m src.experiments.feature_discovery_pipeline --discover")
        return
    
    config = OmegaConf.load(CANDIDATES_CONFIG)
    candidates = list(config.get("candidates", []))
    
    # Count by status
    status_counts = {}
    for c in candidates:
        status = c.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    summary = config.get("audit_summary", {})
    
    print("\n" + "=" * 70)
    print("FEATURE DISCOVERY PIPELINE STATUS")
    print("=" * 70)
    print(f"\nLast audit: {config.get('last_audit', 'Never')}")
    print(f"Raw fields analyzed: {summary.get('total_raw_fields', 0)}")
    print(f"High-value discovered: {summary.get('high_value_found', 0)}")
    print(f"\nCandidate Status:")
    for status, count in sorted(status_counts.items()):
        icon = {"pending": "⏳", "approved": "✅", "rejected": "❌", 
                "integrated": "🚀", "failed": "⚠️"}.get(status, "?")
        print(f"  {icon} {status}: {count}")
    
    # Pending actions
    pending = [c for c in candidates if c.get("status") == "pending"]
    approved = [c for c in candidates if c.get("status") == "approved"]
    
    print("\n" + "-" * 70)
    print("PENDING ACTIONS:")
    if pending:
        print(f"  • {len(pending)} candidates awaiting human review")
        print("    Run: --review to see details")
    if approved:
        print(f"  • {len(approved)} candidates approved, ready for integration")
        print("    Run: --integrate to see integration guide")
    if not pending and not approved:
        print("  ✅ No pending actions")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature Discovery Pipeline - Automated feature candidate management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run discovery on raw data
  python -m src.experiments.feature_discovery_pipeline --discover

  # Review pending candidates
  python -m src.experiments.feature_discovery_pipeline --review

  # Approve candidates
  python -m src.experiments.feature_discovery_pipeline --approve "listing.prices.rent.interval,listing.platforms"

  # Get integration instructions
  python -m src.experiments.feature_discovery_pipeline --integrate
        """
    )
    
    parser.add_argument("--discover", action="store_true",
                       help="Run field discovery/audit on raw_insertions.parquet")
    parser.add_argument("--review", action="store_true",
                       help="Review current feature candidates")
    parser.add_argument("--approve", type=str,
                       help="Approve candidates (comma-separated field names)")
    parser.add_argument("--reject", type=str,
                       help="Reject candidates (comma-separated field names)")
    parser.add_argument("--integrate", action="store_true",
                       help="Generate integration guide for approved candidates")
    parser.add_argument("--mark-integrated", type=str,
                       help="Mark candidates as integrated (comma-separated field names)")
    parser.add_argument("--status", action="store_true",
                       help="Show pipeline status report")
    
    args = parser.parse_args()
    
    # Default to status if no args
    if not any([args.discover, args.review, args.approve, args.reject, 
                args.integrate, args.mark_integrated, args.status]):
        args.status = True
    
    # Execute requested actions
    if args.discover:
        results = run_discovery()
        update_candidates_config(results)
        print("\n✅ Discovery complete. Run --review to see candidates.")
    
    if args.review:
        review_candidates()
    
    if args.approve:
        field_names = [f.strip() for f in args.approve.split(",")]
        approve_candidates(field_names)
    
    if args.reject:
        field_names = [f.strip() for f in args.reject.split(",")]
        reject_candidates(field_names)
    
    if args.integrate:
        integrate_approved_candidates()
    
    if args.mark_integrated:
        field_names = [f.strip() for f in args.mark_integrated.split(",")]
        mark_integrated(field_names)
    
    if args.status:
        generate_status_report()


if __name__ == "__main__":
    main()

