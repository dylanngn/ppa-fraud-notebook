"""
Feature column constants used across all models.
"""

GRAPH_FEATURE_COLUMNS = [
    "contact_email_count",
    "shared_contact_email_count",
    "max_shared_contact_email",
    "contact_phone_count",
    "shared_contact_phone_count",
    "max_shared_contact_phone",
    "user_listing_count",
    "user_unique_ip_count",
    "shared_ip_user_count",
    "max_shared_ip_users",
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

TIME_WEIGHTED_FEATURE_COLUMNS = [
    # Email time-weighted features
    "email_total_historical",
    "email_count_7d",
    "email_count_30d",
    "email_recent_weighted",
    "email_velocity_7d",
    "email_acceleration",
    "email_is_burst",
    "email_is_dormant_reactivation",
    "email_recency_weighted",
    "email_time_spread",
    # Phone time-weighted features
    "phone_total_historical",
    "phone_count_7d",
    "phone_count_30d",
    "phone_recent_weighted",
    "phone_velocity_7d",
    "phone_acceleration",
    "phone_is_burst",
    "phone_is_dormant_reactivation",
    "phone_recency_weighted",
    "phone_time_spread",
    # Combined features
    "combined_velocity_7d",
    "combined_acceleration",
    "any_burst",
    "any_dormant_reactivation",
    "combined_recency_weighted",
]

INTERACTION_FEATURE_COLUMNS = [
    # Binary interactions
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

BASE_FEATURES = [
    "account_age_days",
    "log_price",
    "living_space",
    "rooms",
    "is_new",
    "has_balcony",
    "has_elevator",
    "has_parking",
    "bundle_period",
    "bundle_tier",  # Categorical: "basic", "premium", "top"
    "payment_type",  # Categorical: "DIRECT", "INVOICE"
    "offer_type",  # Categorical: "BUY", "RENT"
    "latitude",
    "longitude",
]

ALL_GRAPH_FEATURE_COLUMNS = (
    GRAPH_FEATURE_COLUMNS
    + ADVANCED_GRAPH_FEATURE_COLUMNS
    + TIME_WEIGHTED_FEATURE_COLUMNS
    + INTERACTION_FEATURE_COLUMNS
    + TEXT_FEATURE_COLUMNS
)

