"""
Feature Schema for Fraud Detection Pipeline.

Single source of truth for feature definitions, graph identities, and temporal config.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum


class ModelVariant(Enum):
    """Model variants for hybrid pipeline."""
    VANILLA_XGBOOST = "vanilla_xgboost"
    GNN_XGBOOST = "gnn_xgboost"


class TemporalSplitStrategy(Enum):
    """Strategies for temporal train/test splitting."""
    FIXED = "fixed"
    EXPANDING = "expanding"
    SLIDING = "sliding"


class LabelPropagation(Enum):
    """How fraud labels propagate across events."""
    INSERTION_LEVEL = "insertion_level"
    EVENT_LEVEL = "event_level"
    FIRST_EVENT = "first_event"


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration for temporal splits."""
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT"
    insertion_id_column: str = "INSERTION_ID_hash"
    gap_days: int = 7
    label_strategy: LabelPropagation = LabelPropagation.INSERTION_LEVEL
    min_events_per_insertion: int = 2
    split_strategy: TemporalSplitStrategy = TemporalSplitStrategy.FIXED


@dataclass(frozen=True)
class GraphIdentityConfig:
    """Configuration for graph construction identities."""
    
    # Device fingerprints from SEON
    device_fingerprint_columns: Tuple[str, ...] = (
        "session/device_hash",
        "session/browser_hash",
        "session/os",
        "session/device_type",
    )
    
    # IP-based connections
    ip_identity_columns: Tuple[str, ...] = (
        "ip_hash",
        "ip_network_hash",
        "ip_isp_name",
    )
    
    # Email identities (hashed)
    email_identity_columns: Tuple[str, ...] = (
        "LISTING_LISTER_EMAIL_hash",
        "LISTING_LISTER_BILLING_EMAIL_hash",
    )
    
    # Phone identities (hashed)
    phone_identity_columns: Tuple[str, ...] = (
        "LISTING_LISTER_PHONE_hash",
        "LISTING_LISTER_BILLING_PHONEDAY_hash",
    )
    
    # User identities
    user_identity_columns: Tuple[str, ...] = (
        "user_id_hash",
        "CREATEDATUSERNAME_hash",
    )
    
    @property
    def all_identity_columns(self) -> Tuple[str, ...]:
        return (
            self.device_fingerprint_columns +
            self.ip_identity_columns +
            self.email_identity_columns +
            self.phone_identity_columns +
            self.user_identity_columns
        )
    
    @property
    def high_signal_columns(self) -> Tuple[str, ...]:
        return (
            "ip_hash",
            "session/device_hash",
            "user_id_hash",
            "LISTING_LISTER_EMAIL_hash",
            "LISTING_LISTER_PHONE_hash",
        )


@dataclass(frozen=True)
class FeatureSchema:
    """Feature schema for fraud detection."""
    
    # Identifiers
    id_columns: Tuple[str, ...] = (
        "INSERTION_ID_hash",
        "LISTINGID",
        "id",  # SEON unique transaction ID
    )
    
    # Time column
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT"
    
    # SEON risk scores
    seon_score_features: Tuple[str, ...] = (
        "fraud_score",
        "blackbox_score",
        "phone_score",
        "email_score",
        "proxy_score",
    )
    
    # SEON IP features (categorical - string)
    seon_ip_categorical: Tuple[str, ...] = (
        "ip_type",
        "ip_country",
        "ip_isp_name",
        "ip_state_prov",
        "ip_timezone_offset",
    )
    
    # SEON IP features (numeric - geo coordinates)
    seon_ip_numeric: Tuple[str, ...] = (
        "ip_latitude",
        "ip_longitude",
    )
    
    # SEON IP features (boolean)
    seon_ip_boolean: Tuple[str, ...] = (
        "data_center_proxy",
        "residential_proxy",
        "public_proxy",
    )
    
    # SEON email features (boolean)
    seon_email_boolean: Tuple[str, ...] = (
        "email/domain/disposable",
        "email/domain/free",
        "email/deliverable",
        "email/haveibeenpwned_listed",
        "email/domain/suspicious_tld",
        "email/domain/custom",
        "email/domain/registered",
        "email/domain/dmarc_enforced",
        "email/domain/website_exists",
    )
    
    # SEON email features (numeric)
    seon_email_numeric: Tuple[str, ...] = (
        "email/number_of_breaches",
    )
    
    # SEON phone features (categorical)
    seon_phone_categorical: Tuple[str, ...] = (
        "phone_type",
        "phone_country",
    )
    
    # SEON phone features (boolean)
    seon_phone_boolean: Tuple[str, ...] = (
        "phone_is_disposable",
        "phone_is_valid",
    )
    
    # SEON phone features (categorical) - more
    seon_phone_categorical_extra: Tuple[str, ...] = (
        "phone_carrier",
    )
    
    # SEON session features (categorical)
    seon_session_categorical: Tuple[str, ...] = (
        "session/device_type",
        "session/os",
        "session/browser",
    )
    
    # SEON session features (boolean)
    seon_session_boolean: Tuple[str, ...] = (
        "session/private",
        "session/device_vpn",
        "session/adblock",
        "session/cookie_enabled",
    )
    
    # SEON session features (numeric)
    seon_session_numeric: Tuple[str, ...] = (
        "session/video_input_count",
        "session/audio_input_count",
        "session/audio_output_count",
        "session/font_count",
        "session/plugin_count",
    )
    
    # SEON session features (categorical) - more
    seon_session_categorical_extra: Tuple[str, ...] = (
        "session/platform",
        "session/region_timezone",
        "session/screen_resolution",
    )
    
    # Event features
    event_features: Tuple[str, ...] = (
        "STATUS",
    )
    
    # Listing numeric features
    listing_numeric_features: Tuple[str, ...] = (
        "LISTING_PRICES_BUY_PRICE",
        "LISTING_PRICES_RENT_GROSS",
        "LISTING_PRICES_RENT_NET",
        "LISTING_CHARACTERISTICS_NUMBEROFROOMS",
        "LISTING_CHARACTERISTICS_NUMBEROFBATHROOMS",
        "LISTING_CHARACTERISTICS_LOTSIZE",
        "LISTING_CHARACTERISTICS_LIVINGSPACE",
        "LISTING_CHARACTERISTICS_YEARBUILT",
        "LISTING_CHARACTERISTICS_FLOOR",
        "LISTING_CHARACTERISTICS_NUMBEROFFLOORS",
    )
    
    # Listing categorical features
    listing_categorical_features: Tuple[str, ...] = (
        "LISTING_OFFERTYPE",
        "LISTING_CATEGORIES",
        "LISTING_PLATFORMS",
        "CUSTOMERSEGMENT",
        "LISTING_PRICES_CURRENCY",
        "LISTING_ADDRESS_COUNTRY",
        "LISTING_LISTER_BILLING_ADDRESS_COUNTRY",
    )
    
    # Billing/transaction categorical
    billing_categorical_features: Tuple[str, ...] = (
        "billing_country",
        "payment_mode",
        "action_type",
        "TARGETPLATFORM",
    )
    
    # Temporal features (derived)
    temporal_features: Tuple[str, ...] = (
        "event_hour",
        "event_weekday",
        "event_is_weekend",
        "event_is_business_hours",
    )
    
    # Target
    target: str = "is_fraud"
    raw_target_source: str = "FLAGGEDFORFRAUD"
    
    # SEON benchmark
    benchmark_column: str = "state"
    
    # GNN config
    gnn_embedding_dim: int = 64
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 2
    
    # Sub-configs
    graph_config: GraphIdentityConfig = field(default_factory=GraphIdentityConfig)
    temporal_config: TemporalConfig = field(default_factory=TemporalConfig)
    
    @property
    def all_seon_boolean(self) -> Tuple[str, ...]:
        """All SEON boolean features."""
        return (
            self.seon_ip_boolean +
            self.seon_email_boolean +
            self.seon_phone_boolean +
            self.seon_session_boolean
        )
    
    @property
    def all_seon_numeric(self) -> Tuple[str, ...]:
        """All SEON numeric features (non-score)."""
        return (
            self.seon_ip_numeric +
            self.seon_email_numeric +
            self.seon_session_numeric
        )
    
    @property
    def all_seon_categorical(self) -> Tuple[str, ...]:
        """All SEON categorical (string) features."""
        return (
            self.seon_ip_categorical +
            self.seon_phone_categorical +
            self.seon_phone_categorical_extra +
            self.seon_session_categorical +
            self.seon_session_categorical_extra
        )
    
    @property
    def all_numeric_features(self) -> Tuple[str, ...]:
        """All numeric features (scores + listing + temporal + booleans + seon numeric)."""
        return (
            self.seon_score_features +
            self.all_seon_numeric +
            self.listing_numeric_features +
            self.temporal_features +
            self.all_seon_boolean  # Booleans treated as numeric (0/1)
        )
    
    @property
    def all_categorical_features(self) -> Tuple[str, ...]:
        """All categorical features (string types only)."""
        return (
            self.all_seon_categorical +
            self.listing_categorical_features +
            self.billing_categorical_features +
            ("STATUS",)
        )
    
    @property
    def all_base_features(self) -> Tuple[str, ...]:
        return self.all_numeric_features + self.all_categorical_features
    
    def get_features_for_variant(self, variant: ModelVariant) -> List[str]:
        base = list(self.all_base_features)
        if variant == ModelVariant.GNN_XGBOOST:
            gnn_features = [f"gnn_emb_{i}" for i in range(self.gnn_embedding_dim)]
            return base + gnn_features
        return base
    
    def get_gnn_input_features(self) -> List[str]:
        return list(self.seon_score_features) + list(self.listing_numeric_features[:6])
    
    def get_graph_identity_columns(self) -> List[str]:
        return list(self.graph_config.all_identity_columns)


FEATURE_SCHEMA = FeatureSchema()
