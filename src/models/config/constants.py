"""
Feature column constants used across all models.

Feature Tiers (based on data quality analysis 2025-12-01):
- TIER 1 (CORE): High coverage (>77%), proven predictive power
- TIER 1b (BOOLEAN): Boolean indicators with 100% semantic coverage (NULL = FALSE)
- TIER 2 (GRAPH): Computed from graph structure, +11.9% AUC-PR improvement
- TIER 3 (ENHANCED): Time-weighted & text features, +0.98% incremental improvement
- DEPRECATED: Proven ineffective features (NOT low coverage - that was a bug)

IMPORTANT: Boolean indicator fields use NULL = FALSE semantics.
           They have 100% semantic coverage regardless of null_pct.
           See docs/knowledge_base.md for details.
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

# =============================================================================
# TIER 1b: BOOLEAN INDICATOR FEATURES (100% Semantic Coverage)
# NULL = FALSE semantics - these are NOT low coverage!
# =============================================================================

# High TRUE rate (>40%) - common property features
BOOLEAN_HIGH_TRUE_RATE = [
    "has_balcony",              # 71.2% TRUE
    "has_parking",              # 55.1% TRUE
    "has_nice_view",            # 47.7% TRUE
    "has_garage",               # 44.2% TRUE
    "is_child_friendly",        # 43.5% TRUE
    "is_quiet",                 # 42.7% TRUE
    "has_elevator",             # 40.9% TRUE 
]

# Moderate TRUE rate (20-40%) - still valuable
BOOLEAN_MODERATE_TRUE_RATE = [
    "has_washing_machine",      # 32.4% TRUE
    "are_pets_allowed",         # 28.4% TRUE
    "is_wheelchair_accessible", # 25.7% TRUE
]

# Low TRUE rate (<20%) - rare but potentially discriminative for fraud
# Fraudsters may over-claim or under-claim certain amenities
BOOLEAN_LOW_TRUE_RATE = [
    "is_old",                   # 16.7% TRUE
    "is_new_building",          # 16.2% TRUE
]

# Very rare TRUE rate (5-15%) - for ablation experiments
# Added to ETL (create_artifacts.py), GNN (graph_builder.py), and XGBoost (base.py)
BOOLEAN_RARE_EXPERIMENTAL = [
    "has_cable_tv",             # 14.76% TRUE
    "has_fireplace",            # 10.71% TRUE
    "is_minergie_general",      # 9.20% TRUE
    "is_minergie_certified",    # 6.81% TRUE
    "is_smoking_allowed",       # 4.88% TRUE
    "has_swimming_pool",        # 4.75% TRUE
]

# All boolean indicator features currently in model
TIER1_BOOLEAN_FEATURES = (
    BOOLEAN_HIGH_TRUE_RATE
    + BOOLEAN_MODERATE_TRUE_RATE
    + BOOLEAN_LOW_TRUE_RATE
)

# All booleans including experimental (for future experiments)
ALL_BOOLEAN_FEATURES = TIER1_BOOLEAN_FEATURES + BOOLEAN_RARE_EXPERIMENTAL

# Backward compatibility alias
TIER1B_BOOLEAN_FEATURES = TIER1_BOOLEAN_FEATURES


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
# Proven ineffective in experiments - NOT just "low coverage"
# =============================================================================

# Burst detection features (Exp 9: no significant impact, didn't appear in top 20)
DEPRECATED_BURST_FEATURES = [
    "email_is_burst",
    "email_is_dormant_reactivation",
    "phone_is_burst",
    "phone_is_dormant_reactivation",
    "any_burst",
    "any_dormant_reactivation",
]

# Uninformative features (almost never set, not useful for discrimination)
UNINFORMATIVE_FEATURES = [
    "is_new",              # 0.04% TRUE - almost never set, uninformative
]

# All deprecated features (proven ineffective, not just low coverage)
DEPRECATED_FEATURES = DEPRECATED_BURST_FEATURES + UNINFORMATIVE_FEATURES


# =============================================================================
# BASE FEATURES (Core tabular features)
# =============================================================================
BASE_FEATURES = (
    TIER1_CORE_FEATURES
    + TIER1_BOOLEAN_FEATURES
)


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
    + TEXT_FEATURE_COLUMNS
)


# =============================================================================
# FEATURE GROUPS (For Ablation Experiments)
# Each group can be independently toggled on/off
# =============================================================================
FEATURE_GROUPS = {
    # Tier 1: Core features
    "core_numerical": TIER1_CORE_FEATURES,
    
    # Tier 1b: Boolean indicators (split for ablation)
    "boolean_high": BOOLEAN_HIGH_TRUE_RATE,
    "boolean_moderate": BOOLEAN_MODERATE_TRUE_RATE,
    "boolean_low": BOOLEAN_LOW_TRUE_RATE,
    "boolean_all": TIER1_BOOLEAN_FEATURES,
    "boolean_rare": BOOLEAN_RARE_EXPERIMENTAL,
    
    # Tier 2: Graph features
    "graph_basic": GRAPH_FEATURE_COLUMNS,
    "graph_advanced": ADVANCED_GRAPH_FEATURE_COLUMNS,
    "graph_all": TIER2_GRAPH_FEATURES,
    
    # Tier 3: Enhanced features
    "time_weighted_core": TIME_WEIGHTED_CORE_FEATURES,
    "time_weighted_extra": TIME_WEIGHTED_EXTRA_FEATURES,
    "text": TEXT_FEATURE_COLUMNS,
    
    # Deprecated (for comparison experiments)
    "burst_detection": DEPRECATED_BURST_FEATURES,
}


# =============================================================================
# FEATURE PROFILES (For Configuration)
# Now defined as combinations of INCLUDE groups (no exclusions!)
# =============================================================================
FEATURE_PROFILES = {
    "quick": {
        "description": "Fast iteration during development",
        "include_groups": ["core_numerical", "boolean_high"],
        "categories": ["base"],
    },
    "standard": {
        "description": "Good balance of speed and performance",
        "include_groups": ["core_numerical", "boolean_all", "graph_all"],
        "categories": ["base", "graph", "advanced_graph"],
    },
    "production": {
        "description": "Best performance for production deployment",
        "include_groups": [
            "core_numerical", "boolean_all",
            "graph_all", "time_weighted_core", "text"
        ],
        "categories": ["base", "graph", "advanced_graph", "time_weighted", "text"],
    },
    "ablation_no_boolean": {
        "description": "Ablation: No boolean features (test their value)",
        "include_groups": ["core_numerical", "graph_all"],
        "categories": ["base", "graph", "advanced_graph"],
    },
    "ablation_boolean_only_high": {
        "description": "Ablation: Only high TRUE rate booleans (>40%)",
        "include_groups": ["core_numerical", "boolean_high", "graph_all"],
        "categories": ["base", "graph", "advanced_graph"],
    },
    "ablation_low_true_rate": {
        "description": "Ablation: Include low TRUE rate booleans (<20%)",
        "include_groups": ["core_numerical", "boolean_all", "graph_all"],
        "categories": ["base", "graph", "advanced_graph"],
    },
}
