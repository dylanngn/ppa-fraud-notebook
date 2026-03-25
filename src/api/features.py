"""
Feature Engineering for Real-Time Prediction.

Computes features from an incoming event payload,
using the internal event store for graph-like features.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
import numpy as np

from src.api.models import EventPayload
from src.api.store import EventStore

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes features for real-time fraud prediction.
    
    Features computed:
    1. Direct SEON features (from event payload)
    2. Listing features (from event payload)
    3. Temporal features (computed from timestamp)
    4. Graph-like features (from event store)
    """
    
    # Expected feature order — must match training exactly (model.get_booster().feature_names)
    FEATURE_ORDER = [
        # SEON IP numeric
        "ip_latitude",
        "ip_longitude",

        # SEON Email numeric
        "email/number_of_breaches",

        # SEON Session counts
        "session/video_input_count",
        "session/audio_input_count",
        "session/audio_output_count",
        "session/font_count",
        "session/plugin_count",

        # Listing numeric
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

        # Temporal
        "event_hour",
        "event_weekday",
        "event_is_weekend",
        "event_is_business_hours",

        # SEON IP booleans
        "data_center_proxy",
        "residential_proxy",
        "public_proxy",

        # SEON Email booleans
        "email/domain/disposable",
        "email/domain/free",
        "email/deliverable",
        "email/haveibeenpwned_listed",
        "email/domain/suspicious_tld",
        "email/domain/custom",
        "email/domain/registered",
        "email/domain/dmarc_enforced",
        "email/domain/website_exists",

        # SEON Phone booleans
        "phone_is_disposable",
        "phone_is_valid",

        # SEON Session booleans
        "session/private",
        "session/device_vpn",
        "session/adblock",
        "session/cookie_enabled",

        # Categorical (ordinal-encoded at train time; NaN for unknown at inference)
        "ip_type",
        "ip_country",
        "ip_isp_name",
        "ip_state_prov",
        "ip_timezone_offset",
        "phone_type",
        "phone_country",
        "phone_carrier",
        "session/device_type",
        "session/os",
        "session/browser",
        "session/platform",
        "session/region_timezone",
        "session/screen_resolution",
        "LISTING_OFFERTYPE",
        "LISTING_CATEGORIES",
        "LISTING_PLATFORMS",
        "CUSTOMERSEGMENT",
        "LISTING_PRICES_CURRENCY",
        "LISTING_ADDRESS_COUNTRY",
        "LISTING_LISTER_BILLING_ADDRESS_COUNTRY",
        "billing_country",
        "payment_mode",
        "action_type",
        "TARGETPLATFORM",
    ]
    
    def __init__(self, event_store: EventStore):
        """
        Initialize feature engineer.

        Args:
            event_store: Event store for graph-like features
        """
        self.event_store = event_store
        self._cat_encoding: dict = {}   # col → list[category_string] (train-time order)

    def set_categorical_encoding(self, encoding: dict) -> None:
        """Load train-time ordinal category mappings (from categorical_encoding.json artifact)."""
        self._cat_encoding = encoding
    
    def compute_features(
        self,
        event: EventPayload,
    ) -> pd.DataFrame:
        """
        Compute all features for an event.
        
        Args:
            event: Incoming event payload
            
        Returns:
            DataFrame with single row of features
        """
        features = {}

        # Extract raw SEON signals (not SEON's ML prediction scores)

        # 1. SEON IP Features
        features.update(self._extract_seon_ip(event))
        
        # 3. SEON Email Features
        features.update(self._extract_seon_email(event))
        
        # 4. SEON Phone Features
        features.update(self._extract_seon_phone(event))
        
        # 5. SEON Session Features
        features.update(self._extract_seon_session(event))
        
        # 6. Listing Features
        features.update(self._extract_listing(event))
        
        # 7. Temporal Features
        features.update(self._compute_temporal(event.event_timestamp))
        
        # 8. Graph-like Features (from store)
        features.update(self._compute_graph_features(event))
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure column order matches training
        for col in self.FEATURE_ORDER:
            if col not in df.columns:
                df[col] = 0 if "count" in col or col.endswith("_count") else None
        
        # Select only known columns (in order)
        available = [c for c in self.FEATURE_ORDER if c in df.columns]
        df = df[available]
        
        # Handle data types
        df = self._convert_types(df)
        
        return df
    
    
    def _extract_seon_ip(self, event: EventPayload) -> Dict[str, Any]:
        """Extract SEON IP features."""
        return {
            "ip_type": event.seon_ip_type,
            "ip_country": event.seon_ip_country,
            "ip_isp_name": event.seon_ip_isp,
            "ip_latitude": None,  # Not in simplified payload
            "ip_longitude": None,
            "data_center_proxy": event.seon_datacenter,
            "residential_proxy": None,
            "public_proxy": event.seon_proxy,
            "tor": event.seon_tor,
            "vpn": event.seon_vpn,
        }
    
    def _extract_seon_email(self, event: EventPayload) -> Dict[str, Any]:
        """Extract SEON email features."""
        return {
            "email/number_of_breaches": event.seon_email_breaches,
            "email/domain/disposable": event.seon_email_disposable,
            "email/domain/free": event.seon_email_free,
            "email/deliverable": event.seon_email_deliverable,
            # Extended email signals — not in EventPayload, default to None
            "email/haveibeenpwned_listed": None,
            "email/domain/suspicious_tld": None,
            "email/domain/custom": None,
            "email/domain/registered": None,
            "email/domain/dmarc_enforced": None,
            "email/domain/website_exists": None,
        }
    
    def _extract_seon_phone(self, event: EventPayload) -> Dict[str, Any]:
        """Extract SEON phone features."""
        return {
            "phone_is_disposable": None,  # Derived from phone_type
            "phone_is_valid": event.seon_phone_valid,
            "phone_type": event.seon_phone_type,
            "phone_country": event.seon_phone_country,
            "phone_carrier": event.seon_phone_carrier,
        }
    
    def _extract_seon_session(self, event: EventPayload) -> Dict[str, Any]:
        """Extract SEON session features."""
        return {
            "session/device_type": event.seon_device_type,
            "session/os": event.seon_os,
            "session/browser": event.seon_browser,
            "session/screen_resolution": event.seon_screen_resolution,
            "session/adblock": event.seon_adblock,
            "session/private": event.seon_private_mode,
            "session/device_vpn": event.seon_vpn,
            "session/cookie_enabled": None,
            "session/video_input_count": None,
            "session/audio_input_count": None,
            "session/audio_output_count": None,
            "session/font_count": None,
            "session/plugin_count": None,
            "session/platform": None,
        }
    
    def _extract_listing(self, event: EventPayload) -> Dict[str, Any]:
        """Extract listing features."""
        # Map price based on offer type
        price = event.listing_price or 0
        is_rent = event.listing_offer_type == "RENT" if event.listing_offer_type else False
        
        return {
            "LISTING_PRICES_BUY_PRICE": price if not is_rent else None,
            "LISTING_PRICES_RENT_GROSS": price if is_rent else None,
            "LISTING_PRICES_RENT_NET": price * 0.9 if is_rent else None,  # Approximate
            "LISTING_CHARACTERISTICS_NUMBEROFROOMS": event.listing_rooms,
            "LISTING_CHARACTERISTICS_LIVINGSPACE": event.listing_living_space,
            "LISTING_OFFERTYPE": event.listing_offer_type,
            "LISTING_CATEGORIES": event.listing_category,
            "LISTING_PLATFORMS": event.listing_platform,
            "LISTING_ADDRESS_COUNTRY": event.listing_country,
        }
    
    def _compute_temporal(self, timestamp: datetime) -> Dict[str, Any]:
        """Compute temporal features from event timestamp."""
        # Ensure timezone aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        return {
            "event_hour": hour,
            "event_weekday": weekday,
            "event_is_weekend": 1 if weekday >= 5 else 0,
            "event_is_business_hours": 1 if 9 <= hour <= 17 and weekday < 5 else 0,
        }
    
    def _compute_graph_features(self, event: EventPayload) -> Dict[str, int]:
        """
        Compute graph-like features from event store.
        
        These approximate GNN features by counting related entities.
        """
        return self.event_store.count_related(event)
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to appropriate types."""
        # Boolean columns to Int8
        bool_cols = [
            "data_center_proxy", "residential_proxy", "public_proxy",
            "tor", "vpn", "email/domain/disposable", "email/domain/free",
            "email/deliverable", "phone_is_disposable", "phone_is_valid",
            "session/private", "session/device_vpn", "session/adblock",
            "session/cookie_enabled", "event_is_weekend", "event_is_business_hours",
        ]
        
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype("Int8")
        
        # Categorical columns: ordinal-encode using train-time mappings when available,
        # otherwise NaN (XGBoost treats as missing).
        cat_cols = [
            "ip_type", "ip_country", "ip_isp_name", "ip_state_prov",
            "ip_timezone_offset", "phone_type", "phone_country", "phone_carrier",
            "session/device_type", "session/os", "session/browser", "session/platform",
            "session/region_timezone", "session/screen_resolution",
            "LISTING_OFFERTYPE", "LISTING_CATEGORIES", "LISTING_PLATFORMS",
            "CUSTOMERSEGMENT", "LISTING_PRICES_CURRENCY", "LISTING_ADDRESS_COUNTRY",
            "LISTING_LISTER_BILLING_ADDRESS_COUNTRY", "billing_country",
            "payment_mode", "action_type", "TARGETPLATFORM",
        ]

        for col in cat_cols:
            if col not in df.columns:
                continue
            if col in self._cat_encoding:
                # Map string → ordinal index; unknown / None → NaN
                cat_index = {v: i for i, v in enumerate(self._cat_encoding[col])}
                df[col] = df[col].map(cat_index).astype("float64")
            else:
                df[col] = np.nan

        # Cast all remaining columns to float64.
        # Columns initialised to None have object dtype; XGBoost rejects object columns.
        for col in df.columns:
            if col not in bool_cols and col not in cat_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                except Exception:
                    df[col] = np.nan

        return df

