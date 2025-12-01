"""
Feature column constants used across all models.

Feature Tiers (based on data quality analysis 2025-11-30):
- TIER 1 (CORE): High coverage (>90%), proven predictive power
- TIER 2 (GRAPH): Computed from graph structure, +11.9% AUC-PR improvement
- TIER 3 (ENHANCED): Time-weighted & text features, +0.98% incremental improvement
- DEPRECATED: Low coverage, redundant, or proven ineffective features

See data_quality_reports/data_quality_report.txt for full analysis.
"""

# =============================================================================
# TIER 1: CORE FEATURES (Always Include)
# High coverage (>77%), essential for baseline model
# =============================================================================
TIER1_CORE_FEATURES = [
    # Identity & Time (100% coverage)
    "account_age_days",
    
    # Payment (98% coverage, highest predictive power per Exp 10: 17.5x fraud lift)
    "payment_type",        # "DIRECT" vs "INVOICE"
    
    # Bundle (77-98% coverage)
    "bundle_tier",
    "bundle_period",
    
    # Price (80% coverage)
    "log_price",           # Derived from price_rent_gross
    
    # Location (99.4% coverage)
    "latitude",
    "longitude",
    
    # Listing Type (100% coverage)
    "offer_type",          # "BUY" vs "RENT"
    
    # Property Characteristics (high coverage only)
    "living_space",        # 84.7% coverage
    "rooms",               # 92.7% coverage
]

# Boolean features with moderate coverage (include but document caveats)
TIER1_BOOLEAN_FEATURES = [
    "has_balcony",         # 71.2% coverage
    "has_parking",         # 55.1% coverage
]


# =============================================================================
# TIER 2: GRAPH-DERIVED FEATURES (Proven Value)
# Computed from graph structure, provided +11.9% AUC-PR improvement (Exp 4)
# =============================================================================
GRAPH_FEATURE_COLUMNS = [
    # Contact Reuse (strongest graph signal)
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
]

ADVANCED_GRAPH_FEATURE_COLUMNS = [
    "degree_total",
    "is_isolated",
    "unique_identifier_count",
    "neighbor_overlap_score",
    "avg_neighbor_degree",
]

TIER2_GRAPH_FEATURES = GRAPH_FEATURE_COLUMNS + ADVANCED_GRAPH_FEATURE_COLUMNS


# =============================================================================
# TIER 3: ENHANCED FEATURES (Optional, Incremental Value)
# Time-weighted (+0.98%) and text features - enable for production
# =============================================================================

# Time-weighted features that showed impact in Exp 9
TIME_WEIGHTED_CORE_FEATURES = [
    # Email time-weighted (proven value)
    "email_time_spread",           # Appeared in top 11 features
    "email_recency_weighted",      # Appeared in top 17 features
    "email_total_historical",
    "email_count_30d",
    "email_velocity_7d",
    
    # Phone time-weighted
    "phone_time_spread",
    "phone_recency_weighted",
    "phone_total_historical",
    "phone_count_30d",
    
    # Combined
    "combined_recency_weighted",   # Appeared in top 15 features
]

# Additional time-weighted features (lower impact, optional)
TIME_WEIGHTED_EXTRA_FEATURES = [
    "email_count_7d",
    "email_count_prev_7d",
    "email_count_30_90d",
    "email_recent_weighted",
    "email_acceleration",
    "phone_count_7d",
    "phone_count_prev_7d",
    "phone_count_30_90d",
    "phone_recent_weighted",
    "phone_acceleration",
    "combined_velocity_7d",
    "combined_acceleration",
]

# Full time-weighted column list (for backwards compatibility)
TIME_WEIGHTED_FEATURE_COLUMNS = (
    TIME_WEIGHTED_CORE_FEATURES + TIME_WEIGHTED_EXTRA_FEATURES + [
        # Burst detection (proven ineffective in Exp 9, kept for reference)
        "email_is_burst",
        "email_is_dormant_reactivation",
        "phone_is_burst",
        "phone_is_dormant_reactivation",
        "any_burst",
        "any_dormant_reactivation",
    ]
)

# Text features
TEXT_FEATURE_COLUMNS = [
    "description_length",
    "description_word_count",
    "description_has_url",
    "description_has_email",
    "description_has_phone",
    "description_caps_ratio",
    "description_exclamation_count",
    "description_question_count",
    "description_all_caps_words",
    "description_avg_word_length",
]

TIER3_ENHANCED_FEATURES = TIME_WEIGHTED_CORE_FEATURES + TEXT_FEATURE_COLUMNS


# =============================================================================
# DEPRECATED FEATURES (Remove from model training)
# Low coverage, redundant, or proven ineffective in experiments
# =============================================================================

# Burst detection features (Exp 9: no significant impact)
DEPRECATED_BURST_FEATURES = [
    "email_is_burst",
    "email_is_dormant_reactivation",
    "phone_is_burst",
    "phone_is_dormant_reactivation",
    "any_burst",
    "any_dormant_reactivation",
]

# Interaction features (Exp 10: XGBoost learns these automatically)
# Keep in code for rule-based flagging, but exclude from model training
INTERACTION_FEATURE_COLUMNS = [
    # Binary interactions (for explainability, not training)
    "new_account_high_email_reuse",
    "new_account_high_phone_reuse",
    "new_account_high_reuse_any",
    "new_account_invoice_payment",
    "very_new_account_invoice",
    "is_small_listing",
    "new_account_small_listing",
    "in_large_component",
    "new_account_large_component",
    "new_account_isolated",
    # Continuous scores
    "account_age_risk_score",
    "email_reuse_intensity",
    "component_risk_score",
    "suspicious_combo_score",
]

# Low coverage boolean features (removed from BASE_FEATURES)
DEPRECATED_LOW_COVERAGE_FEATURES = [
    "is_new",              # 99.99% NULL - almost never populated
    "has_elevator",        # ~60% NULL - unreliable signal
]

# All deprecated features
DEPRECATED_FEATURES = (
    DEPRECATED_BURST_FEATURES
    + DEPRECATED_LOW_COVERAGE_FEATURES
)


# =============================================================================
# BASE FEATURES (Core tabular features with good coverage)
# =============================================================================
BASE_FEATURES = [
    "account_age_days",
    "log_price",
    "living_space",
    "rooms",
    "has_balcony",
    "has_parking",
    "bundle_period",
    "bundle_tier",         # Categorical: "basic", "premium", "top"
    "payment_type",        # Categorical: "DIRECT", "INVOICE"
    "offer_type",          # Categorical: "BUY", "RENT"
    "latitude",
    "longitude",
]


# =============================================================================
# AGGREGATED FEATURE SETS (For Different Use Cases)
# =============================================================================

# Quick iteration: Core features only (fastest training)
QUICK_PROFILE_FEATURES = TIER1_CORE_FEATURES + TIER1_BOOLEAN_FEATURES

# Standard: Core + Graph (good balance)
STANDARD_PROFILE_FEATURES = (
    TIER1_CORE_FEATURES
    + TIER1_BOOLEAN_FEATURES
    + TIER2_GRAPH_FEATURES
)

# Production: All proven features (best performance)
PRODUCTION_PROFILE_FEATURES = (
    TIER1_CORE_FEATURES
    + TIER1_BOOLEAN_FEATURES
    + TIER2_GRAPH_FEATURES
    + TIER3_ENHANCED_FEATURES
)

# Full: Everything except deprecated (backwards compatible)
ALL_GRAPH_FEATURE_COLUMNS = (
    GRAPH_FEATURE_COLUMNS
    + ADVANCED_GRAPH_FEATURE_COLUMNS
    + TIME_WEIGHTED_FEATURE_COLUMNS
    + INTERACTION_FEATURE_COLUMNS
    + TEXT_FEATURE_COLUMNS
)


# =============================================================================
# FEATURE PROFILES (For Configuration)
# =============================================================================
FEATURE_PROFILES = {
    "quick": {
        "description": "Fast iteration during development",
        "features": QUICK_PROFILE_FEATURES,
        "categories": ["base"],
    },
    "standard": {
        "description": "Good balance of speed and performance",
        "features": STANDARD_PROFILE_FEATURES,
        "categories": ["base", "graph", "advanced_graph"],
    },
    "production": {
        "description": "Best performance for production deployment",
        "features": PRODUCTION_PROFILE_FEATURES,
        "categories": ["base", "graph", "advanced_graph", "time_weighted", "text"],
    },
    "full": {
        "description": "All features except deprecated (backwards compatible)",
        "features": ALL_GRAPH_FEATURE_COLUMNS + BASE_FEATURES,
        "categories": ["base", "graph", "advanced_graph", "time_weighted", "text", "interaction"],
    },
}
