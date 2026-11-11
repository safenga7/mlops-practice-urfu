"""
FastAPI ML Inference Microservice for Iris Species Classification (Lab 3).
Provides /health and /predict endpoints with input validation and preprocessing.
"""

import os
import pickle
import logging
from contextlib import asynccontextmanager
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml-microservice")

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# Global model state
artifacts = {
    "model": None,
    "scaler": None,
}


def load_artifacts():
    """Loads trained model and scaler artifacts."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warning("Artifacts not found in %s, attempting to train model on-the-fly...", models_dir)
        from train_model import train_and_save
        train_and_save(models_dir)

    with open(model_path, "rb") as f:
        artifacts["model"] = pickle.load(f)
    with open(scaler_path, "rb") as f:
        artifacts["scaler"] = pickle.load(f)

    logger.info("Successfully loaded ML model and StandardScaler.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    load_artifacts()
    yield
    artifacts.clear()


app = FastAPI(
    title="Iris Classification Microservice",
    description="MLOps Lab 3: Production-ready ML inference API using FastAPI and Docker",
    version="1.0.0",
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0.0, lt=25.0, description="Sepal length in cm", example=5.1)
    sepal_width: float = Field(..., gt=0.0, lt=25.0, description="Sepal width in cm", example=3.5)
    petal_length: float = Field(..., gt=0.0, lt=25.0, description="Petal length in cm", example=1.4)
    petal_width: float = Field(..., gt=0.0, lt=25.0, description="Petal width in cm", example=0.2)


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Target class index (0, 1, or 2)")
    species: str = Field(..., description="Human-readable species name")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


class BatchPredictRequest(BaseModel):
    items: List[IrisFeatures]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


@app.get("/", tags=["Info"])
def root():
    """Service metadata and documentation link."""
    return {
        "service": "Iris Classification Microservice",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint confirming service and model readiness."""
    model_ready = artifacts["model"] is not None and artifacts["scaler"] is not None
    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": TARGET_NAMES,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(features: IrisFeatures):
    """
    Predict Iris species for a single sample.
    Input features are validated and normalized using StandardScaler.
    """
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")

    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference",
        )

    try:
        raw_features = np.array(
            [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]],
            dtype=float,
        )
        scaled_features = scaler.transform(raw_features)

        pred_class = int(model.predict(scaled_features)[0])
        pred_probs = model.predict_proba(scaled_features)[0]

        prob_dict = {
            name: round(float(prob), 4)
            for name, prob in zip(TARGET_NAMES, pred_probs)
        }

        species_name = TARGET_NAMES[pred_class] if pred_class < len(TARGET_NAMES) else "unknown"

        return PredictResponse(
            prediction=pred_class,
            species=species_name,
            probabilities=prob_dict,
        )
    except Exception as e:
        logger.error("Inference error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictRequest):
    """Batch prediction endpoint for multiple samples."""
    results = [predict(item) for item in batch.items]
    return BatchPredictResponse(predictions=results)
