"""
FastAPI Application - Fraud Detection Microservice

Endpoints:
- POST /predict: Get fraud prediction
- POST /explain: Get SHAP explanation
- GET /health: Service health check
"""
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

from src.api.prediction_service import PredictionService
from src.explainability import SHAPService
from src.features.store import FeatureStore


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    listing_id: int = Field(..., description="Listing ID to predict")
    as_of_time: Optional[str] = Field(None, description="ISO timestamp for temporal consistency")


class PredictionResponse(BaseModel):
    listing_id: int
    fraud_score: float
    fraud_probability: float
    is_cold_start: bool
    confidence: float
    model_version: str
    timestamp: str


class ExplanationRequest(BaseModel):
    listing_id: int = Field(..., description="Listing ID to explain")
    generate_plot: bool = Field(True, description="Generate waterfall plot")


class ExplanationResponse(BaseModel):
    listing_id: int
    fraud_score: float
    base_value: float
    top_features: List[Dict[str, float]]
    waterfall_plot_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    feature_count: int
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production fraud detection service with SHAP explainability",
    version="1.0.0"
)

# Initialize services (lazy loading on first request)
prediction_service: Optional[PredictionService] = None
shap_service: Optional[SHAPService] = None


def get_prediction_service() -> PredictionService:
    """Lazy load prediction service."""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service


def get_shap_service() -> SHAPService:
    """Lazy load SHAP service."""
    global shap_service
    if shap_service is None:
        pred_svc = get_prediction_service()
        shap_service = SHAPService(
            model=pred_svc.model,
            feature_names=pred_svc.feature_names
        )
    return shap_service


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud score for a listing.
    
    - **listing_id**: Listing to predict
    - **as_of_time**: Optional timestamp for temporal consistency (ISO format)
    """
    try:
        # Parse as_of_time if provided
        as_of_time = None
        if request.as_of_time:
            as_of_time = datetime.fromisoformat(request.as_of_time)
            
        # Get prediction
        pred_svc = get_prediction_service()
        result = pred_svc.predict(request.listing_id, as_of_time)
        
        return PredictionResponse(
            listing_id=result.listing_id,
            fraud_score=result.fraud_score,
            fraud_probability=result.fraud_probability,
            is_cold_start=result.is_cold_start,
            confidence=result.confidence,
            model_version=result.model_version,
            timestamp=result.timestamp.isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    """
    Generate SHAP explanation for a listing prediction.
    
    - **listing_id**: Listing to explain
    - **generate_plot**: Whether to generate waterfall plot (base64 encoded)
    """
    try:
        # Get feature store and prepare features
        pred_svc = get_prediction_service()
        feature_snapshot = pred_svc.feature_store.get_features(request.listing_id)
        
        # Prepare feature vector
        features = pred_svc._prepare_features(feature_snapshot.features)[0]
        
        # Get prediction
        result = pred_svc.predict(request.listing_id)
        
        # Generate SHAP explanation
        shap_svc = get_shap_service()
        explanation = shap_svc.explain(
            features=features,
            listing_id=request.listing_id,
            fraud_score=result.fraud_score,
            generate_plot=request.generate_plot
        )
        
        # Format top features for response
        top_features_formatted = [
            {feat: val} for feat, val in explanation.top_features
        ]
        
        return ExplanationResponse(
            listing_id=explanation.listing_id,
            fraud_score=explanation.fraud_score,
            base_value=explanation.base_value,
            top_features=top_features_formatted,
            waterfall_plot_base64=explanation.waterfall_plot_base64
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    
    Returns service status, model version, and feature count.
    """
    try:
        pred_svc = get_prediction_service()
        model_info = pred_svc.get_model_info()
        
        return HealthResponse(
            status="healthy",
            model_version=model_info["model_version"],
            feature_count=model_info["feature_count"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            model_version="unknown",
            feature_count=0,
            timestamp=datetime.now().isoformat()
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "explain": "/explain",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
