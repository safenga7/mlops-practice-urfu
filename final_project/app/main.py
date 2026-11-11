"""
Production ML Inference Service (FastAPI) for Final Project
Serves predictions from a pre-trained Iris classification model with full input validation and telemetry.
"""

import os
import time
import pickle
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Constants & Model Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model", "iris_classifier.pkl")
TARGET_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Global state
model_state: Dict[str, Any] = {
    "model": None,
    "start_time": time.time(),
    "prediction_count": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pre-trained model on startup and clean up on shutdown."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Pre-trained model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model_state["model"] = pickle.load(f)
    print(f"ML Model successfully loaded from {MODEL_PATH}")
    yield
    model_state["model"] = None

app = FastAPI(
    title="Final MLOps ML Inference Service",
    description="Production-grade ML prediction microservice for automated classification",
    version="1.0.0",
    lifespan=lifespan
)

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0.0, le=15.0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0.0, le=15.0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0.0, le=15.0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0.0, le=15.0, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    }

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probabilities: Dict[str, float]

class BatchIrisFeatures(BaseModel):
    samples: List[IrisFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int

@app.get("/", tags=["General"])
def read_root():
    """Returns general microservice information and operational status."""
    return {
        "service": "MLOps Final Project ML Service",
        "version": "1.0.0",
        "status": "online",
        "endpoints": ["/health", "/predict", "/predict/batch", "/metrics", "/docs"]
    }

@app.get("/health", tags=["Monitoring"])
def healthcheck():
    """Healthcheck endpoint for orchestrators (Docker, Kubernetes, Jenkins)."""
    is_loaded = model_state["model"] is not None
    if not is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded"
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "uptime_seconds": round(time.time() - model_state["start_time"], 2)
    }

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Operational telemetry endpoint."""
    return {
        "predictions_served": model_state["prediction_count"],
        "uptime_seconds": round(time.time() - model_state["start_time"], 2)
    }

def _predict_sample(features: IrisFeatures) -> PredictionResponse:
    clf = model_state["model"]
    x = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    class_id = int(clf.predict(x)[0])
    probs = clf.predict_proba(x)[0]
    
    prob_dict = {
        TARGET_NAMES[i]: round(float(prob), 4)
        for i, prob in enumerate(probs)
    }
    return PredictionResponse(
        class_id=class_id,
        class_name=TARGET_NAMES[class_id],
        probabilities=prob_dict
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: IrisFeatures):
    """Predict flower category from four numerical features."""
    if model_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready"
        )
    result = _predict_sample(features)
    model_state["prediction_count"] += 1
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchIrisFeatures):
    """Batch prediction endpoint for multiple samples."""
    if model_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready"
        )
    if not batch.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty"
        )
    
    results = [_predict_sample(sample) for sample in batch.samples]
    model_state["prediction_count"] += len(results)
    return BatchPredictionResponse(predictions=results, count=len(results))
