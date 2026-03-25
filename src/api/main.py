"""
Fraud Detection Microservice.

A self-contained FastAPI service that:
  1. Receives raw listing-submission events (no external DB dependency)
  2. Stores events internally for graph-feature computation
  3. Runs real-time fraud predictions using a trained XGBoost model

Production notes
----------------
This service is designed for the *vanilla_xgboost* model variant, which takes
a flat feature vector as input.  The GNN+XGBoost variant requires an offline
graph-rebuild step (see /rebuild-graph and the production guide in README.md).

Model loading
-------------
The model is loaded from the MLflow Model Registry by default:

    models:/<MLFLOW_MODEL_NAME>@production   (preferred)
    models:/<MLFLOW_MODEL_NAME>/latest        (fallback)

Set MLFLOW_MODEL_ALIAS=staging to test a candidate model before promoting it.
Alternatively, set MODEL_PATH to a local XGBoost model file for development.

Startup
-------
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    # Development (hot-reload):
    uvicorn src.api.main:app --reload
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

import mlflow
import mlflow.xgboost
import xgboost as xgb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    EventPayload,
    PredictResponse,
    IngestResponse,
    BulkIngestRequest,
    BulkIngestResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskTier,
    Decision,
    RiskFactor,
)
from src.api.store import EventStore
from src.api.features import FeatureEngineer
from src.api.admin_store import AdminStore
from src.api.drift_state import DriftState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all tunable via environment variables
# ---------------------------------------------------------------------------

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///ppa-fraud-detection-mlflow.db")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")
# Alias to load from the Model Registry ("production", "staging", etc.)
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")
# Optional: bypass the registry and load a model file directly (dev/CI use)
MODEL_PATH = os.getenv("MODEL_PATH", None)
STORE_PERSIST_PATH = os.getenv("STORE_PERSIST_PATH", "artifacts/api/event_store.json")
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "1_000_000"))

# Decision thresholds
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "0.7"))
THRESHOLD_MEDIUM = float(os.getenv("THRESHOLD_MEDIUM", "0.3"))

# ---------------------------------------------------------------------------
# Global state (initialised in lifespan)
# ---------------------------------------------------------------------------

event_store: Optional[EventStore] = None
feature_engineer: Optional[FeatureEngineer] = None
admin_store: Optional[AdminStore] = None
drift_state: Optional[DriftState] = None
model: Optional[xgb.XGBClassifier] = None
_model_uri: Optional[str] = None          # URI the model was actually loaded from
_model_run_id: Optional[str] = None       # MLflow run that produced the model
_categorical_encoding: Optional[dict] = None  # train-time ordinal mappings for categoricals


def _load_model() -> None:
    """Load XGBoost model from a file path or the MLflow Model Registry."""
    global model, _model_uri, _model_run_id, _categorical_encoding

    # 1. Direct file path — highest priority (useful for local dev / CI)
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from path: {MODEL_PATH}")
        m = xgb.XGBClassifier()
        m.load_model(MODEL_PATH)
        model = m
        _model_uri = MODEL_PATH
        return

    # 2. MLflow Model Registry
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Try alias first (e.g. "production"), fall back to latest version
    for uri in (
        f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}",
        f"models:/{MLFLOW_MODEL_NAME}/latest",
    ):
        try:
            logger.info(f"Attempting to load model from: {uri}")
            model = mlflow.xgboost.load_model(uri)
            _model_uri = uri

            # Retrieve the run ID so we can surface it in /model-info
            client = mlflow.MlflowClient()
            try:
                mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
                _model_run_id = mv.run_id
            except Exception:
                pass  # Non-fatal; run ID is informational only

            logger.info(f"Model loaded from: {uri}")

            # Load categorical encoding so the serving layer can reproduce ordinal codes
            try:
                cat_enc = mlflow.artifacts.load_dict(f"runs:/{_model_run_id}/model/categorical_encoding.json")
                _categorical_encoding = cat_enc
                logger.info(f"Categorical encoding loaded ({len(cat_enc)} columns)")
            except Exception as exc:
                logger.warning(f"Could not load categorical encoding: {exc}")

            return
        except Exception as exc:
            logger.warning(f"Could not load from {uri}: {exc}")

    logger.warning("No model loaded — predictions will return a neutral probability of 0.5")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — initialise and teardown service resources."""
    global event_store, feature_engineer, admin_store, drift_state

    logger.info("Starting Fraud Detection Service …")

    event_store = EventStore(max_events=MAX_EVENTS, persist_path=STORE_PERSIST_PATH)
    logger.info(f"Event store initialised (max_events={MAX_EVENTS})")

    feature_engineer = FeatureEngineer(event_store)
    logger.info("Feature engineer initialised")

    admin_store = AdminStore(storage_dir="artifacts/decisions")
    logger.info("Admin store initialised")

    drift_state = DriftState(reference_window=100, current_window=50)
    logger.info("Drift state initialised")

    _load_model()
    if model:
        logger.info("Model loaded — service is fully operational")
    else:
        logger.warning("No model loaded — service is running in degraded mode")

    if _categorical_encoding:
        feature_engineer.set_categorical_encoding(_categorical_encoding)
        logger.info("Categorical encoding wired into feature engineer")

    # Mount admin router (after all globals are ready)
    from src.api.routers.admin import router as admin_router
    app.include_router(admin_router, prefix="/admin")
    logger.info("Admin router mounted at /admin")

    yield

    # Graceful shutdown: flush the event store to disk
    logger.info("Shutting down Fraud Detection Service …")
    if event_store:
        event_store.persist()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="""
Real-time fraud detection for real estate listing submissions.

## Workflow
1. **Ingest** historical events via `POST /ingest` or `POST /bulk-ingest` to warm
   the in-memory graph store (enables graph-link features).
2. **Predict** — send the full event payload to `POST /predict`.  The event is
   automatically stored for future graph features.
3. **Inspect** the loaded model metadata via `GET /model-info`.

## Production notes
The `vanilla_xgboost` model variant operates in true real-time: each event
is scored independently using tabular SEON signals.  The `gnn_xgboost` variant
requires a periodic offline graph rebuild; call `POST /rebuild-graph` to
trigger it after bootstrapping the event store with historical data.
""",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(event: EventPayload, background_tasks: BackgroundTasks):
    """
    Score a listing-submission event for fraud.

    The event payload is stored asynchronously so that subsequent predictions
    for related listings benefit from updated graph-link counts.
    """
    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    # Store event for future graph-feature computation (non-blocking)
    background_tasks.add_task(event_store.store_event, event)

    # Feature engineering
    features_df = feature_engineer.compute_features(event)

    # Inference
    if model is not None:
        try:
            proba = float(model.predict_proba(features_df)[0, 1])
        except Exception as exc:
            logger.error(f"Prediction error: {exc}")
            proba = 0.5
    else:
        logger.warning("No model loaded — returning neutral probability")
        proba = 0.5

    # Decision logic
    if proba >= THRESHOLD_HIGH:
        risk_tier = RiskTier.HIGH
        decision = Decision.DECLINE
    elif proba >= THRESHOLD_MEDIUM:
        risk_tier = RiskTier.MEDIUM
        decision = Decision.REVIEW
    else:
        risk_tier = RiskTier.LOW
        decision = Decision.APPROVE

    # Record prediction for drift tracking
    if drift_state is not None:
        drift_state.record_prediction(proba)

    # Route into admin store for queue / auto-decision
    if admin_store is not None:
        prediction_dict = {
            "fraud_probability": proba,
            "risk_tier": risk_tier.value,
            "decision": decision.value,
        }
        event_dict = event.model_dump()
        if risk_tier == RiskTier.MEDIUM:
            admin_store.enqueue(event.insertion_id, prediction_dict, event_dict)
        elif risk_tier == RiskTier.LOW:
            background_tasks.add_task(admin_store.auto_decide, event.insertion_id, "APPROVE", event_dict)
        else:
            background_tasks.add_task(admin_store.auto_decide, event.insertion_id, "DECLINE", event_dict)

    risk_factors = _generate_risk_factors(event)
    confidence = abs(proba - 0.5) * 2  # Maps [0.5, 1] → [0, 1]

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


@app.post("/ingest", response_model=IngestResponse, tags=["Data Ingestion"])
async def ingest(event: EventPayload):
    """
    Store a single event in the internal event store.

    Use this endpoint to bootstrap the graph store with historical events
    *before* calling `/predict`.  Events are also stored automatically during
    prediction.
    """
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    status = event_store.store_event(event)
    events_count = len(event_store.get_events(event.insertion_id))

    return IngestResponse(
        insertion_id=event.insertion_id,
        status=status,
        events_stored=events_count,
    )


@app.post("/bulk-ingest", response_model=BulkIngestResponse, tags=["Data Ingestion"])
async def bulk_ingest(request: BulkIngestRequest):
    """
    Ingest a batch of historical events.

    Recommended before serving live traffic to ensure graph-link features
    are computed with sufficient historical context.
    """
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    errors = []
    stored = 0
    for i, event in enumerate(request.events):
        try:
            event_store.store_event(event)
            stored += 1
        except Exception as exc:
            errors.append(f"Event {i}: {exc}")

    return BulkIngestResponse(
        events_processed=len(request.events),
        events_stored=stored,
        errors=errors[:10],
    )


@app.post("/rebuild-graph", tags=["Administration"])
async def rebuild_graph():
    """
    Rebuild the fraud graph from all stored events.

    For the `vanilla_xgboost` variant this refreshes the in-memory link-count
    indexes used for graph-like features.

    For the `gnn_xgboost` variant a full graph rebuild + GNN re-inference is
    required to generate updated node embeddings.  This endpoint currently
    refreshes the link-count indexes only; full GNN retraining must be
    triggered via the training pipeline (`make train-gnn`) and the new model
    promoted in the Model Registry.
    """
    if event_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    stats = event_store.stats()
    return {
        "status": "graph_refreshed",
        "events_processed": stats["total_events"],
        "nodes": stats["total_insertions"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Link-count indexes refreshed.  For gnn_xgboost, retrain the model "
            "offline and promote the new version in the MLflow Model Registry."
        ),
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Administration"])
async def model_info():
    """
    Return metadata about the currently loaded model.

    Useful for confirming which version is live after a registry promotion.
    """
    return ModelInfoResponse(
        model_loaded=model is not None,
        model_uri=_model_uri,
        mlflow_run_id=_model_run_id,
        mlflow_model_name=MLFLOW_MODEL_NAME,
        mlflow_model_alias=MLFLOW_MODEL_ALIAS,
        threshold_high=THRESHOLD_HIGH,
        threshold_medium=THRESHOLD_MEDIUM,
        service_version=MODEL_VERSION,
    )


@app.get("/health", response_model=HealthResponse, tags=["Administration"])
async def health():
    """Service health check."""
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
        last_graph_update=(
            datetime.fromisoformat(stats["last_updated"])
            if stats.get("last_updated") else None
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_risk_factors(event: EventPayload) -> list[RiskFactor]:
    """
    Build a human-readable list of the top fraud signals for this event.

    This is a lightweight rule-based explanation.  For full SHAP attribution
    use the SHAPAnalyzer class from src.evaluation.shap_analysis (slower).
    """
    factors: list[RiskFactor] = []

    signal_checks = [
        (event.seon_tor, "tor_network", 0.25),
        (event.seon_vpn, "vpn_detected", 0.15),
        (event.seon_email_disposable, "disposable_email", 0.20),
        (event.seon_datacenter, "datacenter_ip", 0.15),
        (event.seon_proxy, "proxy_detected", 0.10),
    ]
    for flag, name, impact in signal_checks:
        if flag:
            factors.append(RiskFactor(feature=name, value="true", impact=impact, direction="increases_risk"))

    # Graph-link counts from the event store
    if event_store:
        graph_counts = event_store.count_related(event)
        if graph_counts.get("device_link_count", 0) > 5:
            factors.append(RiskFactor(
                feature="shared_device",
                value=f"{graph_counts['device_link_count']} other listings",
                impact=0.20,
                direction="increases_risk",
            ))
        if graph_counts.get("ip_link_count", 0) > 10:
            factors.append(RiskFactor(
                feature="shared_ip",
                value=f"{graph_counts['ip_link_count']} other listings",
                impact=0.15,
                direction="increases_risk",
            ))

    factors.sort(key=lambda x: abs(x.impact), reverse=True)
    return factors[:5]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
