"""
Fraud Detection API Module.

A self-contained microservice for real-time fraud detection.

Components:
- main.py: FastAPI application
- models.py: Request/response models
- store.py: Internal event store
- features.py: Real-time feature engineering

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from src.api.main import app
from src.api.models import EventPayload, PredictResponse
from src.api.store import EventStore
from src.api.features import FeatureEngineer

__all__ = [
    "app",
    "EventPayload",
    "PredictResponse",
    "EventStore",
    "FeatureEngineer",
]

