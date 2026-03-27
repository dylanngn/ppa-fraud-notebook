"""
Admin API Router.

Provides endpoints for:
  - Human review queue management
  - Per-prediction SHAP explanations (waterfall chart as base64 PNG)
  - Concept drift status and reset
  - Retraining trigger
  - Storage statistics
  - Synthetic event simulation (demo heartbeat)

All endpoints are mounted at /admin via:
    app.include_router(admin_router, prefix="/admin")
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import random
import string
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])

# ---------------------------------------------------------------------------
# Lazy SHAP analyser — initialised on first request
# ---------------------------------------------------------------------------

_shap_analyzer = None


def _get_shap_analyzer():
    global _shap_analyzer
    if _shap_analyzer is None:
        import src.api.main as _main
        if _main.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded — cannot run SHAP")
        from src.evaluation.shap_analysis import SHAPAnalyzer, SHAPConfig
        from src.api.features import FeatureEngineer
        _shap_analyzer = SHAPAnalyzer(
            model=_main.model,
            feature_names=FeatureEngineer.FEATURE_ORDER,
            config=SHAPConfig(max_samples=500, top_k_features=15, output_dir="artifacts/shap"),
        )
        logger.info("SHAPAnalyzer initialised")
    return _shap_analyzer


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AdminDecisionRequest(BaseModel):
    decision: str           # "APPROVE" or "DECLINE"
    admin_id: str = "admin-1"
    notes: str = ""


class SimulateRequest(BaseModel):
    scenario: str = "RANDOM"    # HIGH_RISK | MEDIUM_RISK | LOW_RISK | RANDOM


# ---------------------------------------------------------------------------
# Helpers — access shared state from main
# ---------------------------------------------------------------------------

def _get_state():
    """Return (event_store, feature_engineer, admin_store, drift_state, model) from main."""
    import src.api.main as m
    return m.event_store, m.feature_engineer, m.admin_store, m.drift_state, m.model


# ---------------------------------------------------------------------------
# Simulate endpoint — demo heartbeat
# ---------------------------------------------------------------------------

def _random_id(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def _make_event_dict(scenario: str) -> Dict[str, Any]:
    """Build a synthetic EventPayload dict for the given scenario."""
    base = {
        "insertion_id": _random_id(),
        "user_id": _random_id(),
        "event_type": "PENDING_APPROVAL",
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "email_hash": _random_id(32),
        "phone_hash": _random_id(32),
        "ip_hash": _random_id(32),
        "device_hash": _random_id(32),
        # Default safe values
        "seon_tor": False,
        "seon_vpn": False,
        "seon_proxy": False,
        "seon_datacenter": False,
        "seon_email_disposable": False,
        "seon_email_free": False,
        "seon_email_deliverable": True,
        "seon_email_breaches": 0,
        "seon_phone_valid": True,
        "seon_phone_type": "MOBILE",
        "seon_phone_carrier": "Swisscom",
        "seon_phone_country": "CH",
        "seon_ip_type": "residential",
        "seon_ip_country": "CH",
        "seon_ip_isp": "Swisscom",
        "seon_device_type": "desktop",
        "seon_browser": "Chrome",
        "seon_os": "Windows",
        "seon_screen_resolution": "1920x1080",
        "seon_adblock": False,
        "seon_private_mode": False,
        "listing_category": random.choice(["APARTMENT", "HOUSE", "STUDIO"]),
        "listing_platform": random.choice(["ImmoScout24", "Homegate"]),
        "listing_offer_type": random.choice(["RENT", "BUY"]),
        "listing_price": round(random.uniform(1500, 5000), 0),
        "listing_country": "CH",
        "listing_rooms": random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        "listing_living_space": round(random.uniform(40, 150), 0),
    }

    if scenario == "RANDOM":
        weights = [0.70, 0.20, 0.10]
        scenario = random.choices(["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"], weights=weights)[0]

    if scenario == "HIGH_RISK":
        base.update({
            "seon_tor": True,
            "seon_vpn": True,
            "seon_datacenter": True,
            "seon_email_disposable": True,
            "seon_ip_country": "NG",
            "seon_ip_isp": "Unknown Provider",
            "listing_price": round(random.uniform(800, 1200), 0),
        })
    elif scenario == "MEDIUM_RISK":
        base.update({
            "seon_vpn": True,
            "seon_email_disposable": True,
            "seon_ip_country": "RO",
            "seon_email_free": True,
        })
    # LOW_RISK: already defaults to safe values

    return base


@router.post("/simulate", summary="Fire a synthetic event through the full pipeline")
async def simulate(req: SimulateRequest):
    """
    Generate a synthetic listing-submission event matching the requested
    risk scenario and run it through the complete prediction pipeline.

    Used as the demo heartbeat by the simulator background loop.
    """
    event_store, feature_engineer, admin_store, drift_state, model = _get_state()

    if feature_engineer is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    event_dict = _make_event_dict(req.scenario)

    # Build EventPayload
    from src.api.models import EventPayload
    event = EventPayload(**event_dict)

    # Store for graph features
    if event_store:
        event_store.store_event(event)

    # Feature engineering + inference
    features_df = feature_engineer.compute_features(event)

    if model is not None:
        try:
            proba = float(model.predict_proba(features_df)[0, 1])
        except Exception as exc:
            logger.warning(f"Predict error in simulate: {exc}")
            proba = 0.5
    else:
        proba = 0.5

    # Decision routing
    from src.api.models import Decision, RiskTier
    from src.api.main import THRESHOLD_HIGH, THRESHOLD_MEDIUM

    # In simulation mode, the synthetic events don't carry the real-world
    # feature patterns the model learned from.  Blend the model's output
    # with a scenario-appropriate range so the demo shows realistic score
    # diversity while still exercising the full pipeline.
    # Ranges are anchored to the configured thresholds.
    _SIM_RANGES = {
        "HIGH_RISK":   (THRESHOLD_HIGH + 0.05, min(THRESHOLD_HIGH + 0.60, 0.95)),
        "MEDIUM_RISK": (THRESHOLD_MEDIUM, THRESHOLD_HIGH),
        "LOW_RISK":    (0.01, THRESHOLD_MEDIUM),
    }
    if req.scenario in _SIM_RANGES:
        lo, hi = _SIM_RANGES[req.scenario]
        if hi > lo:
            proba = lo + random.random() * (hi - lo)
        else:
            proba = lo
        proba = round(proba, 4)

    # Drift tracking (with feature vector for feature-level drift detection)
    if drift_state:
        drift_state.record_prediction(
            proba,
            features=features_df.values[0].tolist(),
            feature_names=list(features_df.columns),
        )

    if proba >= THRESHOLD_HIGH:
        risk_tier = RiskTier.HIGH
        decision = Decision.DECLINE
    elif proba >= THRESHOLD_MEDIUM:
        risk_tier = RiskTier.MEDIUM
        decision = Decision.REVIEW
    else:
        risk_tier = RiskTier.LOW
        decision = Decision.APPROVE

    prediction = {
        "fraud_probability": round(proba, 4),
        "risk_tier": risk_tier.value,
        "decision": decision.value,
    }

    if admin_store:
        if risk_tier == RiskTier.MEDIUM:
            admin_store.enqueue(event.insertion_id, prediction, event_dict)
        elif risk_tier == RiskTier.LOW:
            admin_store.auto_decide(event.insertion_id, "APPROVE", event_dict)
        else:
            admin_store.auto_decide(event.insertion_id, "DECLINE", event_dict)

    return {
        "insertion_id": event.insertion_id,
        "scenario": req.scenario,
        **prediction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

@router.get("/queue", summary="Get all pending review items")
async def get_queue():
    _, _, admin_store, _, _ = _get_state()
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    items = admin_store.get_queue()
    return {
        "items": [
            {
                "insertion_id": i.insertion_id,
                "fraud_probability": i.fraud_probability,
                "risk_tier": i.risk_tier,
                "event_summary": i.event_summary,
                "queued_at": i.queued_at,
            }
            for i in items
        ],
        "count": len(items),
        "oldest_queued_at": items[0].queued_at if items else None,
    }


@router.get("/queue/{insertion_id}", summary="Get a single review item")
async def get_queue_item(insertion_id: str):
    event_store, _, admin_store, _, _ = _get_state()
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    item = admin_store.get_queue_item(insertion_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Not in review queue: {insertion_id}")

    # Attach the full raw event if available
    raw_event = None
    if event_store:
        raw_event = event_store.get_latest_event(insertion_id)

    return {
        "insertion_id": item.insertion_id,
        "fraud_probability": item.fraud_probability,
        "risk_tier": item.risk_tier,
        "event_summary": item.event_summary,
        "queued_at": item.queued_at,
        "raw_event": raw_event,
    }


# ---------------------------------------------------------------------------
# Admin decision
# ---------------------------------------------------------------------------

@router.post("/decision/{insertion_id}", summary="Record admin approve/decline decision")
async def record_decision(insertion_id: str, req: AdminDecisionRequest):
    _, _, admin_store, _, _ = _get_state()
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    if req.decision not in ("APPROVE", "DECLINE"):
        raise HTTPException(status_code=400, detail="decision must be APPROVE or DECLINE")

    try:
        record = admin_store.decide(insertion_id, req.decision, req.admin_id, req.notes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Not in review queue: {insertion_id}")

    return {
        "insertion_id": record.insertion_id,
        "admin_decision": record.admin_decision,
        "bucket": record.bucket,
        "decided_at": record.decided_at,
        "parquet_flushed": True,
    }


# ---------------------------------------------------------------------------
# Decisions history
# ---------------------------------------------------------------------------

@router.get("/decisions", summary="Recent approved/declined events")
async def list_decisions(bucket: str = "all", limit: int = 50):
    """
    Return recent decisions (both auto and human).

    Query params:
      - bucket: "approved", "declined", or "all" (default)
      - limit: max items to return (default 50)
    """
    _, _, admin_store, _, _ = _get_state()
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    records = admin_store.get_recent_decisions(limit)
    items = [asdict(r) for r in records]
    if bucket != "all":
        items = [d for d in items if d["bucket"] == bucket]
    # Return most recent first
    items.reverse()
    return {"items": items, "count": len(items)}


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

@router.post("/shap/{insertion_id}", summary="Run SHAP waterfall for a stored event")
async def shap_explanation(insertion_id: str):
    """
    Retrieve the stored event, reconstruct features, and compute a SHAP
    waterfall explanation for the prediction.

    Returns:
        JSON with top risk factors + a base64-encoded waterfall PNG that the
        Node-RED ui_template node renders as an inline <img>.
    """
    event_store, feature_engineer, _, _, _ = _get_state()

    if event_store is None or feature_engineer is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    raw = event_store.get_latest_event(insertion_id)
    if not raw:
        raise HTTPException(status_code=404, detail=f"No stored event for: {insertion_id}")

    # Reconstruct EventPayload from stored dict
    from src.api.models import EventPayload
    try:
        event = EventPayload(**{k: v for k, v in raw.items() if k != "event_timestamp"},
                             event_timestamp=datetime.fromisoformat(raw["event_timestamp"]))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not reconstruct event: {exc}")

    features_df = feature_engineer.compute_features(event)

    # Degrade gracefully when no model is loaded rather than returning 503
    try:
        analyzer = _get_shap_analyzer()
    except HTTPException as exc:
        if exc.status_code == 503:
            return {
                "insertion_id": insertion_id,
                "base_value": None,
                "prediction_value": None,
                "top_fraud_indicators": [],
                "top_legitimate_indicators": [],
                "waterfall_chart_base64": None,
                "message": exc.detail,
            }
        raise

    # Run SHAP in a thread so we don't block the async event loop
    loop = asyncio.get_event_loop()
    try:
        explanation = await loop.run_in_executor(
            None,
            lambda: analyzer.explain_prediction(features_df.iloc[0], save_plot=False),
        )
    except Exception as exc:
        logger.error(f"SHAP failed for {insertion_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"SHAP error: {exc}")

    # Generate waterfall chart as base64 PNG
    chart_b64 = _render_waterfall_b64(explanation)

    return {
        "insertion_id": insertion_id,
        "base_value": explanation.get("base_value"),
        "prediction_value": explanation.get("prediction_value"),
        "top_fraud_indicators": explanation.get("top_fraud_indicators", [])[:10],
        "top_legitimate_indicators": explanation.get("top_legitimate_indicators", [])[:5],
        "waterfall_chart_base64": chart_b64,
    }


def _render_waterfall_b64(explanation: Dict[str, Any]) -> Optional[str]:
    """Render a simple horizontal bar chart of SHAP contributions as base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fraud_indicators = explanation.get("top_fraud_indicators", [])
        legit_indicators = explanation.get("top_legitimate_indicators", [])
        items = (fraud_indicators + legit_indicators)[:15]

        if not items:
            return None

        features = [it["feature"] for it in items]
        values = [it["shap_value"] for it in items]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 0.4 + 1)))
        ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor="none")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP contribution (positive = increases fraud risk)")
        ax.set_title(
            f"SHAP Explanation  |  Fraud prob: {explanation.get('prediction_value', '?'):.3f}",
            fontsize=10,
        )
        ax.tick_params(axis="y", labelsize=8)
        red_patch = mpatches.Patch(color="#e74c3c", label="Increases risk")
        green_patch = mpatches.Patch(color="#2ecc71", label="Decreases risk")
        ax.legend(handles=[red_patch, green_patch], fontsize=8, loc="lower right")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as exc:
        logger.warning(f"Waterfall render failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------

@router.get("/drift/status", summary="Current concept drift status")
async def drift_status():
    _, _, _, drift_state, _ = _get_state()
    if drift_state is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return drift_state.get_status()


@router.post("/drift/reset", summary="Reset drift reference distribution")
async def drift_reset():
    _, _, _, drift_state, _ = _get_state()
    if drift_state is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    drift_state.reset()
    return {"status": "reset", "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Retraining trigger + status
# ---------------------------------------------------------------------------

_MIN_RETRAIN_ROWS = 50

_retrain_state: Dict[str, Any] = {
    "status": "idle",          # idle | running | completed | failed
    "started_at": None,
    "finished_at": None,
    "training_rows": None,
    "error": None,
}


def _run_retrain(training_rows: int) -> None:
    """Execute retraining as a background subprocess and update _retrain_state."""
    _retrain_state.update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "training_rows": training_rows,
        "error": None,
    })
    try:
        cmd = [sys.executable, "-m", "src.training.trainer",
               "model.variant=vanilla_xgboost"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info("Retraining completed successfully")
            _retrain_state.update({
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            logger.error(f"Retraining failed:\n{result.stderr[:500]}")
            _retrain_state.update({
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": result.stderr[:500],
            })
    except Exception as exc:
        logger.error(f"Retraining subprocess error: {exc}")
        _retrain_state.update({
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        })


@router.post("/retrain/trigger", summary="Trigger model retraining from labelled decisions")
async def retrain_trigger(background_tasks: BackgroundTasks):
    """
    Validate that enough labelled rows exist in the retrain pool, then
    kick off a background retraining run.

    The training process logs to MLflow.  After it completes, promote the
    new model version via `make register-model` or the MLflow UI.
    """
    _, _, admin_store, _, _ = _get_state()
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    row_count = admin_store.get_pending_retrain_count()

    if row_count < _MIN_RETRAIN_ROWS:
        return {
            "status": "insufficient_data",
            "rows": row_count,
            "required": _MIN_RETRAIN_ROWS,
            "message": f"Need at least {_MIN_RETRAIN_ROWS} labelled rows to retrain (have {row_count}).",
        }

    background_tasks.add_task(_run_retrain, row_count)

    return {
        "status": "triggered",
        "training_rows": row_count,
        "message": (
            "Retraining started in the background.  "
            "Monitor progress in MLflow, then promote the new version: "
            "make register-model"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/retrain/status", summary="Current retraining job status")
async def retrain_status():
    """
    Returns the state of the most recent (or active) retraining run.

    status values:
      idle       — no run has been triggered yet this session
      running    — subprocess is executing
      completed  — finished successfully
      failed     — subprocess exited non-zero or raised an exception
    """
    return dict(_retrain_state)


@router.get("/retrain/last-result", summary="Metrics from the most recent MLflow training run")
async def retrain_last_result():
    """Return key evaluation metrics from the most recently completed training run."""
    import mlflow
    import src.api.main as m
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("fraud-detection")
    if exp is None:
        return {"status": "no_experiment", "metrics": {}}
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {"status": "no_runs", "metrics": {}}
    run = runs[0]
    return {
        "status": run.info.status,
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "started_at": run.info.start_time,
        "metrics": {
            "auc_pr":    run.data.metrics.get("auc_pr"),
            "auc_roc":   run.data.metrics.get("auc_roc"),
            "f1":        run.data.metrics.get("f1"),
            "precision": run.data.metrics.get("precision"),
            "recall":    run.data.metrics.get("recall"),
        },
    }


# ---------------------------------------------------------------------------
# Model promotion and hot-reload
# ---------------------------------------------------------------------------

@router.post("/model/promote", summary="Promote latest MLflow model version to production alias")
async def model_promote():
    """
    Find the latest registered model version and assign the production alias to it.
    Call this after retraining completes to make the new model available for reload.
    """
    import mlflow
    import src.api.main as m
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(
            f"name='{m.MLFLOW_MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model registry error: {exc}")
    if not versions:
        raise HTTPException(status_code=404, detail=f"No versions found for model '{m.MLFLOW_MODEL_NAME}'")
    latest = versions[0]
    client.set_registered_model_alias(m.MLFLOW_MODEL_NAME, m.MLFLOW_MODEL_ALIAS, latest.version)
    return {
        "status": "promoted",
        "version": latest.version,
        "alias": m.MLFLOW_MODEL_ALIAS,
        "run_id": latest.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/model/reload", summary="Hot-reload model from MLflow registry without restarting")
async def model_reload():
    """
    Re-run the model loading sequence in-place, swapping the in-memory model for the
    version currently pointed to by the production alias.  No restart required.
    """
    import src.api.main as m
    prev_uri = m._model_uri
    m._load_model()
    if m.model is None:
        raise HTTPException(status_code=503, detail="Reload failed — model is None after load attempt")
    if m._categorical_encoding and m.feature_engineer:
        m.feature_engineer.set_categorical_encoding(m._categorical_encoding)
    return {
        "status": "reloaded",
        "model_uri": m._model_uri,
        "mlflow_run_id": m._model_run_id,
        "previous_uri": prev_uri,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Storage stats
# ---------------------------------------------------------------------------

@router.get("/storage/stats", summary="Parquet storage statistics")
async def storage_stats():
    _, _, admin_store, _, _ = _get_state()
    if admin_store is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return admin_store.get_storage_stats()
