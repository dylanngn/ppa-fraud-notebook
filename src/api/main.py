"""
Fraud Detection Microservice.

A self-contained FastAPI service that:
1. Receives events directly (no external DB dependency)
2. Stores events internally for graph feature computation
3. Runs real-time fraud predictions

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    
    # Or with reload for development
    uvicorn src.api.main:app --reload
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import mlflow
import xgboost as xgb

from src.api.models import (
    EventPayload,
    PredictResponse,
    IngestResponse,
    BulkIngestRequest,
    BulkIngestResponse,
    HealthResponse,
    RiskTier,
    Decision,
    RiskFactor,
)
from src.api.store import EventStore
from src.api.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///ppa-fraud-detection-mlflow.db")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud_detection")
MODEL_PATH = os.getenv("MODEL_PATH", None)  # Optional direct path to model
STORE_PERSIST_PATH = os.getenv("STORE_PERSIST_PATH", "artifacts/api/event_store.json")
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "1000000"))

# Thresholds for decision
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.7"))
THRESHOLD_MEDIUM = float(os.getenv("THRESHOLD_MEDIUM", "0.3"))

# Global state
event_store: Optional[EventStore] = None
feature_engineer: Optional[FeatureEngineer] = None
model: Optional[xgb.XGBClassifier] = None


def load_model():
    """Load XGBoost model from MLflow or file path."""
    global model
    
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from path: {MODEL_PATH}")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
    else:
        # Try loading from MLflow
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"Loading model from MLflow: {MLFLOW_TRACKING_URI}")
            
            # Load latest model from experiment
            # In production, use Model Registry with stage="Production"
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_uri}")
            
        except Exception as e:
            logger.warning(f"Could not load from MLflow: {e}")
            logger.info("Creating placeholder model for demo")
            # Create a simple placeholder for demo
            model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    global event_store, feature_engineer, model
    
    # Startup
    logger.info("Starting Fraud Detection Service...")
    
    # Initialize event store
    event_store = EventStore(
        max_events=MAX_EVENTS,
        persist_path=STORE_PERSIST_PATH,
    )
    logger.info(f"Event store initialized (max_events={MAX_EVENTS})")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(event_store)
    logger.info("Feature engineer initialized")
    
    # Load model
    load_model()
    if model:
        logger.info("Model loaded successfully")
    else:
        logger.warning("No model loaded - predictions will return placeholder values")
    
    logger.info("Fraud Detection Service ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fraud Detection Service...")
    if event_store:
        event_store.persist()
    logger.info("Event store persisted")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="""
    Real-time fraud detection for real estate listings.
    
    ## Features
    - **Real-time prediction** on listing submission events
    - **Self-contained** - no external database dependencies
    - **Graph-aware** - tracks relationships between listings
    - **Explainable** - provides top risk factors
    
    ## Workflow
    1. Ingest historical events via `/ingest` or `/bulk-ingest`
    2. For each submission, call `/predict` with the event payload
    3. Service returns fraud probability, decision, and explanation
    """,
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(event: EventPayload, background_tasks: BackgroundTasks):
    """
    Predict fraud probability for a listing event.
    
    This is the main prediction endpoint. Send the full event payload
    and receive a fraud prediction with explanation.
    
    The event is automatically stored for future graph feature computation.
    """
    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Store event for graph features (async)
    background_tasks.add_task(event_store.store_event, event)
    
    # Compute features
    features_df = feature_engineer.compute_features(event)
    
    # Run prediction
    if model is not None:
        try:
            proba = model.predict_proba(features_df)[0, 1]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback: use fraud_score if available
            proba = (event.seon_fraud_score or 0) / 100.0
    else:
        # Placeholder: use SEON fraud score as proxy
        proba = (event.seon_fraud_score or 0) / 100.0
    
    # Determine risk tier and decision
    if proba >= THRESHOLD_HIGH:
        risk_tier = RiskTier.HIGH
        decision = Decision.DECLINE
    elif proba >= THRESHOLD_MEDIUM:
        risk_tier = RiskTier.MEDIUM
        decision = Decision.REVIEW
    else:
        risk_tier = RiskTier.LOW
        decision = Decision.APPROVE
    
    # Generate risk factors (simplified - full SHAP would be slow)
    risk_factors = _generate_risk_factors(event, features_df)
    
    # Compute confidence (simplified)
    confidence = abs(proba - 0.5) * 2  # 0.5 -> 0, 0 or 1 -> 1
    
    return PredictResponse(
        insertion_id=event.insertion_id,
        fraud_probability=round(proba, 4),
        risk_tier=risk_tier,
        decision=decision,
        confidence=round(confidence, 4),
        top_risk_factors=risk_factors,
        model_version=MODEL_VERSION,
        prediction_timestamp=datetime.now(timezone.utc),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(event: EventPayload):
    """
    Ingest an event into the store.
    
    Use this to populate historical events for graph feature computation.
    Events are also automatically ingested during prediction.
    """
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    status = event_store.store_event(event)
    events_count = len(event_store.get_events(event.insertion_id))
    
    return IngestResponse(
        insertion_id=event.insertion_id,
        status=status,
        events_stored=events_count,
    )


@app.post("/bulk-ingest", response_model=BulkIngestResponse)
async def bulk_ingest(request: BulkIngestRequest):
    """
    Bulk ingest historical events.
    
    Use this to bootstrap the service with historical data
    for accurate graph feature computation.
    """
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    errors = []
    stored = 0
    
    for i, event in enumerate(request.events):
        try:
            event_store.store_event(event)
            stored += 1
        except Exception as e:
            errors.append(f"Event {i}: {str(e)}")
    
    return BulkIngestResponse(
        events_processed=len(request.events),
        events_stored=stored,
        errors=errors[:10],  # Limit error messages
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns service status, model state, and store statistics.
    """
    stats = event_store.stats() if event_store else {}
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        events_in_store=stats.get("total_events", 0),
        graph_nodes=stats.get("total_insertions", 0),
        graph_edges=sum([
            stats.get("unique_emails", 0),
            stats.get("unique_phones", 0),
            stats.get("unique_ips", 0),
            stats.get("unique_devices", 0),
        ]),
        last_graph_update=datetime.fromisoformat(stats["last_updated"]) if stats.get("last_updated") else None,
    )


@app.post("/rebuild-graph")
async def rebuild_graph():
    """
    Trigger graph rebuild from stored events.
    
    Admin endpoint for manual graph refresh.
    In production, this would rebuild the full graph for GNN features.
    """
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # In a full implementation, this would:
    # 1. Export events to DataFrame
    # 2. Build heterogeneous graph
    # 3. Run GNN to generate embeddings
    # 4. Cache embeddings for fast lookup
    
    stats = event_store.stats()
    
    return {
        "status": "graph_rebuilt",
        "events_processed": stats["total_events"],
        "nodes": stats["total_insertions"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _generate_risk_factors(
    event: EventPayload,
    features_df,
) -> list[RiskFactor]:
    """
    Generate top risk factors for explanation.
    
    This is a simplified version. For full SHAP explanations,
    use the SHAPAnalyzer class (slower but more accurate).
    """
    factors = []
    
    # Check SEON scores
    if event.seon_fraud_score and event.seon_fraud_score > 50:
        factors.append(RiskFactor(
            feature="seon_fraud_score",
            value=str(event.seon_fraud_score),
            impact=0.3,
            direction="increases_risk",
        ))
    
    if event.seon_tor:
        factors.append(RiskFactor(
            feature="tor_network",
            value="true",
            impact=0.25,
            direction="increases_risk",
        ))
    
    if event.seon_vpn:
        factors.append(RiskFactor(
            feature="vpn_detected",
            value="true",
            impact=0.15,
            direction="increases_risk",
        ))
    
    if event.seon_email_disposable:
        factors.append(RiskFactor(
            feature="disposable_email",
            value="true",
            impact=0.2,
            direction="increases_risk",
        ))
    
    if event.seon_datacenter:
        factors.append(RiskFactor(
            feature="datacenter_ip",
            value="true",
            impact=0.15,
            direction="increases_risk",
        ))
    
    # Check graph features
    graph_features = event_store.count_related(event) if event_store else {}
    
    device_links = graph_features.get("device_link_count", 0)
    if device_links > 5:
        factors.append(RiskFactor(
            feature="shared_device",
            value=f"{device_links} other listings",
            impact=0.2,
            direction="increases_risk",
        ))
    
    ip_links = graph_features.get("ip_link_count", 0)
    if ip_links > 10:
        factors.append(RiskFactor(
            feature="shared_ip",
            value=f"{ip_links} other listings",
            impact=0.15,
            direction="increases_risk",
        ))
    
    # Sort by impact and return top 5
    factors.sort(key=lambda x: abs(x.impact), reverse=True)
    return factors[:5]


# CLI for running the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

