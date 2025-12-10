"""
Feature column constants.

SIMPLIFIED after Experiment 9 validation (2025-12-09):
- Tabular features: AUTO mode (all columns minus exclusions)
- Graph features: EXPLICIT (require computation from graph structure)

Experiment 9 showed:
- All 278 raw fields: 0.784 AUC-PR
- Hand-picked 54 fields: 0.776 AUC-PR  
- Difference: +1% (not significant given std=0.09)

Decision: Simplify pipeline by using auto tabular + explicit graph.
"""

# =============================================================================
# EXCLUSION LIST (columns to never use as features)
# =============================================================================
EXCLUDED_COLUMNS = {
    # Target/label columns
    "fraud_flag",
    "is_fraud",
    
    # ID columns (not features)
    "insertion_id",
    "object_reference",
    "owner_id",
    "user_id",
    "listing.id",
    "listing.lister.id",
    
    # Timestamp columns (used for temporal splitting, not features)
    "submission_at",
    "first_published_date",
    "listing_created_at",
    "account_created_at",
    
    # External fraud detection outputs (would leak information - SEON is the baseline we want to beat!)
    "seonFraudScore",
    "seonApproved",
    "seonSession",
    "auto_approval_criteria.meta.seonFraudScore",
    "auto_approval_criteria.criteria.seonApproved",
    "auto_approval_criteria.seonSession",
    "listing.autoApprovalCriteria.meta.seonFraudScore",
    "listing.autoApprovalCriteria.criteria.seonApproved",
    "listing.autoApprovalCriteria.seonSession",
}

# Patterns to exclude (checked via endswith/contains)
EXCLUDED_PATTERNS = [
    "_hash",           # All hash columns are identifiers
    "legacy.personId", # Legacy ID field
]


# =============================================================================
# GRAPH-DERIVED FEATURES (Require explicit computation)
# These are computed from graph structure in src/features/definitions/graph.py
# =============================================================================
GRAPH_FEATURES = [
    # Contact Reuse (strongest graph signal per Exp 10C: 5.72x lift)
    "contact_email_count",
    "shared_contact_email_count",
    "max_shared_contact_email",
    "contact_phone_count",
    "shared_contact_phone_count",
    "max_shared_contact_phone",
    
    # User Behavior
    "user_listing_count",
    "user_unique_ip_count",
    "shared_ip_user_count",
    "max_shared_ip_users",
    
    # Component Analysis
    "listing_component_size",
    "listing_pagerank",
    
    # Advanced
    "degree_total",
    "is_isolated",
    "unique_identifier_count",
    "neighbor_overlap_score",
    "avg_neighbor_degree",
]


# =============================================================================
# CATEGORICAL FEATURES (need special handling in XGBoost)
# =============================================================================
CATEGORICAL_FEATURES = [
    "payment_type",
    "bundle_tier", 
    "offer_type",
    "listing_platform",
    "user_platform",
    "customer_segment",
]
