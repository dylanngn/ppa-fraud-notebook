"""
API Request/Response Models.

Defines the event payload structure that the fraud detection service expects.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Listing lifecycle events."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"
    FLAGGEDFORFRAUD = "FLAGGEDFORFRAUD"
    DELETED = "DELETED"


class RiskTier(str, Enum):
    """Risk classification tiers."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Decision(str, Enum):
    """Recommended action."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class EventPayload(BaseModel):
    """
    Full event data for fraud prediction.
    
    The service receives this payload directly - no database lookup needed.
    All required data for feature engineering should be included.
    """
    
    # === Core Identifiers ===
    insertion_id: str = Field(..., description="Unique listing identifier (hashed)")
    user_id: str = Field(..., description="User identifier (hashed)")
    event_type: EventType = Field(..., description="Lifecycle event type")
    event_timestamp: datetime = Field(..., description="When the event occurred")
    
    # === Listing Attributes ===
    listing_category: Optional[str] = Field(None, description="Property category")
    listing_platform: Optional[str] = Field(None, description="Platform (ImmoScout24, Homegate)")
    listing_offer_type: Optional[str] = Field(None, description="BUY, RENT")
    listing_price: Optional[float] = Field(None, description="Listing price")
    listing_country: Optional[str] = Field(None, description="Property country")
    listing_rooms: Optional[float] = Field(None, description="Number of rooms")
    listing_living_space: Optional[float] = Field(None, description="Living space m²")
    
    # === SEON Fraud Scores ===
    seon_fraud_score: Optional[float] = Field(None, ge=0, le=100, description="Overall fraud score")
    seon_email_score: Optional[float] = Field(None, ge=0, le=100, description="Email risk score")
    seon_phone_score: Optional[float] = Field(None, ge=0, le=100, description="Phone risk score")
    seon_blackbox_score: Optional[float] = Field(None, ge=0, le=100, description="Device risk score")
    seon_proxy_score: Optional[float] = Field(None, ge=0, le=100, description="Proxy risk score")
    
    # === SEON IP Analysis ===
    seon_ip_type: Optional[str] = Field(None, description="IP type (residential, mobile, etc)")
    seon_ip_country: Optional[str] = Field(None, description="IP country code")
    seon_ip_isp: Optional[str] = Field(None, description="Internet service provider")
    seon_tor: Optional[bool] = Field(None, description="TOR network detected")
    seon_vpn: Optional[bool] = Field(None, description="VPN detected")
    seon_proxy: Optional[bool] = Field(None, description="Proxy detected")
    seon_datacenter: Optional[bool] = Field(None, description="Datacenter IP")
    
    # === SEON Email Analysis ===
    seon_email_deliverable: Optional[bool] = Field(None, description="Email is deliverable")
    seon_email_disposable: Optional[bool] = Field(None, description="Disposable email domain")
    seon_email_free: Optional[bool] = Field(None, description="Free email provider")
    seon_email_breaches: Optional[int] = Field(None, description="Number of data breaches")
    
    # === SEON Phone Analysis ===
    seon_phone_valid: Optional[bool] = Field(None, description="Phone number valid")
    seon_phone_type: Optional[str] = Field(None, description="Phone type (mobile, landline)")
    seon_phone_carrier: Optional[str] = Field(None, description="Phone carrier")
    seon_phone_country: Optional[str] = Field(None, description="Phone country")
    
    # === SEON Session/Device ===
    seon_device_type: Optional[str] = Field(None, description="Device type")
    seon_browser: Optional[str] = Field(None, description="Browser")
    seon_os: Optional[str] = Field(None, description="Operating system")
    seon_screen_resolution: Optional[str] = Field(None, description="Screen resolution")
    seon_adblock: Optional[bool] = Field(None, description="Adblocker detected")
    seon_private_mode: Optional[bool] = Field(None, description="Private browsing mode")
    
    # === Identity Signals (Hashed) ===
    email_hash: str = Field(..., description="Hashed email for graph linking")
    phone_hash: str = Field(..., description="Hashed phone for graph linking")
    ip_hash: str = Field(..., description="Hashed IP for graph linking")
    device_hash: str = Field(..., description="Hashed device fingerprint for graph linking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "insertion_id": "abc123hash",
                "user_id": "user456hash",
                "event_type": "SUBMITTED",
                "event_timestamp": "2025-01-15T10:30:00Z",
                "listing_category": "APARTMENT",
                "listing_platform": "ImmoScout24",
                "listing_price": 500000,
                "seon_fraud_score": 25.5,
                "seon_email_score": 30.0,
                "seon_tor": False,
                "seon_vpn": False,
                "email_hash": "e3b0c44298fc...",
                "phone_hash": "d7a8fbb307d7...",
                "ip_hash": "5e884898da28...",
                "device_hash": "9f86d081884c...",
            }
        }


class RiskFactor(BaseModel):
    """Individual risk factor from SHAP explanation."""
    feature: str = Field(..., description="Feature name")
    value: str = Field(..., description="Feature value")
    impact: float = Field(..., description="SHAP impact on prediction")
    direction: str = Field(..., description="'increases_risk' or 'decreases_risk'")


class PredictResponse(BaseModel):
    """Fraud prediction response."""
    
    insertion_id: str
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability 0-1")
    risk_tier: RiskTier = Field(..., description="Risk classification")
    decision: Decision = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    
    # Explainability
    top_risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Top contributing factors to the prediction"
    )
    
    # Metadata
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(..., description="When prediction was made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "insertion_id": "abc123hash",
                "fraud_probability": 0.85,
                "risk_tier": "HIGH",
                "decision": "DECLINE",
                "confidence": 0.92,
                "top_risk_factors": [
                    {"feature": "seon_fraud_score", "value": "85", "impact": 0.35, "direction": "increases_risk"},
                    {"feature": "seon_tor", "value": "true", "impact": 0.25, "direction": "increases_risk"},
                ],
                "model_version": "v1.2.0",
                "prediction_timestamp": "2025-01-15T10:30:05Z",
            }
        }


class IngestResponse(BaseModel):
    """Response for event ingestion."""
    
    insertion_id: str
    status: str = Field(..., description="'stored' or 'updated'")
    events_stored: int = Field(..., description="Total events for this insertion")


class BulkIngestRequest(BaseModel):
    """Bulk event ingestion for bootstrapping."""
    
    events: List[EventPayload]


class BulkIngestResponse(BaseModel):
    """Response for bulk ingestion."""
    
    events_processed: int
    events_stored: int
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health check response."""
    
    status: str = Field(..., description="'healthy' or 'unhealthy'")
    model_loaded: bool
    events_in_store: int
    graph_nodes: int
    graph_edges: int
    last_graph_update: Optional[datetime]

