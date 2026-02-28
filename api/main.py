"""
============================================================
AI-Based Fraud Detection System - FastAPI REST Service
============================================================
Endpoints:
  POST /predict   → Accept transaction features, return fraud probability
  GET  /health    → Health check (useful for load balancers / k8s probes)
  GET  /model-info → Return model metadata (threshold, version, metrics)

Security Notes (see comments throughout):
  - API key authentication via X-API-Key header
  - Rate limiting recommendation included
  - Input validation via Pydantic models
  - Structured JSON logging for observability
============================================================
"""

import os
import json
import time
import logging
import traceback
from typing import Optional, Tuple

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from pythonjsonlogger import jsonlogger

# ── Structured JSON Logging (production-ready) ──────────────────────────────
logger = logging.getLogger("fraud_api")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
SCALER_PATH= os.path.join(BASE_DIR, "model", "scaler.joblib")
META_PATH  = os.path.join(BASE_DIR, "model", "metadata.json")

# ── API Key Security ─────────────────────────────────────────────────────────
# In production: store this in an environment variable / secrets manager
# (AWS Secrets Manager, HashiCorp Vault, etc.)
API_KEY         = os.getenv("FRAUD_API_KEY", "demo-api-key-change-in-production")
api_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: Optional[str] = Depends(api_key_header)):
    """
    Simple API-key gate.  In a real production setup you would:
      1. Store keys hashed in a database (bcrypt / argon2)
      2. Implement per-key rate limiting (Redis + sliding window)
      3. Rotate keys periodically and log key usage
      4. Consider OAuth2 / JWT for multi-tenant scenarios
    """
    if key != API_KEY:
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.  Pass X-API-Key header.",
        )
    return key


# ════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time credit card fraud detection powered by Random Forest",
    version     = "1.0.0",
    docs_url    = "/docs",   # Swagger UI at /docs
    redoc_url   = "/redoc",
)

# CORS – restrict origins in production (currently open for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # ← change to your Streamlit URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load model at startup (once, not per-request) ────────────────────────────
model, scaler, metadata = None, None, {}

@app.on_event("startup")
async def load_model():
    global model, scaler, metadata
    try:
        model   = joblib.load(MODEL_PATH)
        scaler  = joblib.load(SCALER_PATH)
        with open(META_PATH) as f:
            metadata = json.load(f)
        logger.info("Model loaded successfully", extra={"model_type": metadata.get("model_type")})
    except FileNotFoundError:
        # Allow the API to start even without a model (useful for health checks)
        logger.error("Model files not found. Run model/train.py first!")


# ════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class TransactionRequest(BaseModel):
    """
    Mirrors the Kaggle dataset features after preprocessing.
    V1-V28 are PCA features; Amount and Time are raw and will be scaled
    server-side exactly as they were during training.
    """
    Time   : float = Field(..., description="Seconds since first transaction in dataset")
    Amount : float = Field(..., ge=0, description="Transaction amount in EUR (must be ≥ 0)")

    # PCA-transformed features (anonymised)
    V1  : float;  V2  : float;  V3  : float;  V4  : float
    V5  : float;  V6  : float;  V7  : float;  V8  : float
    V9  : float;  V10 : float;  V11 : float;  V12 : float
    V13 : float;  V14 : float;  V15 : float;  V16 : float
    V17 : float;  V18 : float;  V19 : float;  V20 : float
    V21 : float;  V22 : float;  V23 : float;  V24 : float
    V25 : float;  V26 : float;  V27 : float;  V28 : float

    @validator("Amount")
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

    class Config:
        schema_extra = {
            "example": {
                "Time": 406, "Amount": 149.62,
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            }
        }


class PredictionResponse(BaseModel):
    """What the API returns for each transaction."""
    prediction       : int    # 0 = Legit, 1 = Fraud
    fraud_probability: float  # Raw model probability [0.0 – 1.0]
    risk_level       : str    # "Low" | "Medium" | "High"
    risk_score       : int    # Numeric score 0–100 for UI gauges
    threshold_used   : float
    inference_time_ms: float
    model_version    : str


# ════════════════════════════════════════════════════════════════════════════
# RISK SCORING HELPER
# ════════════════════════════════════════════════════════════════════════════

def compute_risk(fraud_prob: float) -> Tuple[str, int]:
    """
    Map raw fraud probability to a human-readable risk tier and
    a 0-100 numeric score for visualisation in the frontend.

      Probability Range → Risk Level  → Score Range
      0.00 – 0.30       → Low         → 0  – 29
      0.30 – 0.70       → Medium      → 30 – 69
      0.70 – 1.00       → High        → 70 – 100
    """
    score = int(fraud_prob * 100)
    if fraud_prob < 0.30:
        level = "Low"
    elif fraud_prob < 0.70:
        level = "Medium"
    else:
        level = "High"
    return level, score


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """
    Lightweight health probe.
    Returns 200 if the API is running and the model is loaded.
    Kubernetes liveness/readiness probes should hit this endpoint.
    """
    model_loaded = model is not None
    return {
        "status"      : "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type"  : metadata.get("model_type", "unknown"),
    }


@app.get("/model-info", tags=["System"])
async def model_info():
    """
    Return model metadata for the frontend dashboard.
    Useful for showing the user what model version is running.
    """
    if not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Strip y_proba from the response (it's training-time data, not needed here)
    safe_meta = {k: v for k, v in metadata.items() if k not in ("y_proba",)}
    return safe_meta


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    transaction : TransactionRequest,
    request     : Request,
    api_key     : str = Depends(verify_api_key),
):
    """
    Main inference endpoint.

    Accepts a JSON body with all transaction features and returns:
      - prediction       : 0 (Legit) or 1 (Fraud)
      - fraud_probability: model confidence that this is fraud
      - risk_level       : Low / Medium / High
      - risk_score       : 0–100 numeric score
      - inference_time_ms: how long the model took to respond
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model/train.py first.")

    # ── Build feature vector in the exact same order as training ─
    feature_dict = transaction.dict()
    amount = feature_dict.pop("Amount")
    time_  = feature_dict.pop("Time")

    # Scale Amount and Time (same scaler used during training)
    # Note: we use transform (not fit_transform) – scaler is already fitted
    scaled_values = scaler.transform([[amount, time_]])
    scaled_amount = scaled_values[0][0]
    scaled_time   = scaled_values[0][1]

    # Build ordered feature array matching metadata["feature_names"]
    feature_names = metadata.get("feature_names", [])
    feature_map   = {**feature_dict, "scaled_amount": scaled_amount, "scaled_time": scaled_time}

    try:
        features = np.array([[feature_map[f] for f in feature_names]])
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature in input: {e}")

    # ── Inference ────────────────────────────────────────────────
    t_start      = time.perf_counter()
    fraud_proba  = float(model.predict_proba(features)[0][1])
    inference_ms = (time.perf_counter() - t_start) * 1000

    threshold    = metadata.get("threshold", 0.5)
    prediction   = int(fraud_proba >= threshold)
    risk_level, risk_score = compute_risk(fraud_proba)

    # ── Structured Logging (feeds into Datadog / ELK / CloudWatch) ──
    logger.info(
        "Prediction completed",
        extra={
            "prediction"       : prediction,
            "fraud_probability": round(fraud_proba, 4),
            "risk_level"       : risk_level,
            "inference_ms"     : round(inference_ms, 3),
            "amount"           : amount,
            "client_ip"        : request.client.host if request.client else "unknown",
        },
    )

    return PredictionResponse(
        prediction        = prediction,
        fraud_probability = round(fraud_proba, 4),
        risk_level        = risk_level,
        risk_score        = risk_score,
        threshold_used    = threshold,
        inference_time_ms = round(inference_ms, 3),
        model_version     = metadata.get("model_type", "RandomForest") + "-v1.0",
    )


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,   # Auto-reload on code changes (dev only – disable in prod)
        workers = 1,      # Increase to CPU cores in production
    )
