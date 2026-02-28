"""
============================================================
Enhanced Fraud Detection API - With Anomaly Detection
============================================================
This enhanced version includes:
  1. Original supervised model endpoint (Random Forest)
  2. Anomaly detection endpoint (Isolation Forest)
  3. Hybrid prediction combining both models
  4. Anomaly visualization data endpoint

Endpoints:
  POST /predict              → Supervised classification
  POST /predict-anomaly      → Unsupervised anomaly detection
  POST /predict-hybrid       → Combined prediction
  GET  /anomaly-stats        → Anomaly detection statistics
  GET  /health               → Health check
  GET  /model-info           → Model metadata
============================================================
"""

import os
import json
import time
import logging
import traceback
from typing import Optional, Tuple, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from pythonjsonlogger import jsonlogger

# ── Structured JSON Logging ──────────────────────────────────────────────────
logger = logging.getLogger("fraud_api")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
SCALER_PATH     = os.path.join(BASE_DIR, "model", "scaler.joblib")
META_PATH       = os.path.join(BASE_DIR, "model", "metadata.json")

ANOMALY_MODEL   = os.path.join(BASE_DIR, "model", "anomaly", "isolation_forest.joblib")
ANOMALY_SCALER  = os.path.join(BASE_DIR, "model", "anomaly", "anomaly_scaler.joblib")
ANOMALY_META    = os.path.join(BASE_DIR, "model", "anomaly", "anomaly_metadata.json")

# ── API Key Security ─────────────────────────────────────────────────────────
API_KEY        = os.getenv("FRAUD_API_KEY", "demo-api-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: Optional[str] = Depends(api_key_header)):
    if key != API_KEY:
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return key


# ════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Enhanced Fraud Detection API",
    description = "Real-time fraud detection with supervised + unsupervised learning",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load models at startup ────────────────────────────────────────────────────
rf_model, rf_scaler, rf_metadata = None, None, {}
anomaly_model, anomaly_scaler, anomaly_metadata = None, None, {}

@app.on_event("startup")
async def load_models():
    global rf_model, rf_scaler, rf_metadata
    global anomaly_model, anomaly_scaler, anomaly_metadata
    
    # Load supervised model
    try:
        rf_model   = joblib.load(MODEL_PATH)
        rf_scaler  = joblib.load(SCALER_PATH)
        with open(META_PATH) as f:
            rf_metadata = json.load(f)
        logger.info("Random Forest model loaded", 
                   extra={"model_type": rf_metadata.get("model_type")})
    except FileNotFoundError:
        logger.error("Random Forest model not found. Run model/train.py first!")
    
    # Load anomaly detection model
    try:
        anomaly_model  = joblib.load(ANOMALY_MODEL)
        anomaly_scaler = joblib.load(ANOMALY_SCALER)
        with open(ANOMALY_META) as f:
            anomaly_metadata = json.load(f)
        logger.info("Anomaly detector loaded", 
                   extra={"model_type": anomaly_metadata.get("model_type")})
    except FileNotFoundError:
        logger.warning("Anomaly detector not found. Run train_anomaly_detector.py")


# ════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class TransactionRequest(BaseModel):
    """Standard transaction features."""
    Time   : float = Field(..., description="Seconds since first transaction")
    Amount : float = Field(..., ge=0, description="Transaction amount (EUR)")
    
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


class PredictionResponse(BaseModel):
    """Standard supervised prediction response."""
    prediction       : int
    fraud_probability: float
    risk_level       : str
    risk_score       : int
    threshold_used   : float
    inference_time_ms: float
    model_version    : str


class AnomalyResponse(BaseModel):
    """Anomaly detection response."""
    is_anomaly       : bool
    anomaly_score    : float  # 0-100, higher = more anomalous
    anomaly_level    : str    # "Normal" | "Suspicious" | "Highly Anomalous"
    threshold_used   : float
    inference_time_ms: float
    behavioral_flags : Dict[str, bool]


class HybridResponse(BaseModel):
    """Combined prediction from both models."""
    final_prediction      : int
    confidence           : float
    supervised_pred      : int
    supervised_prob      : float
    anomaly_score        : float
    risk_assessment      : str
    model_agreement      : bool
    inference_time_ms    : float


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def compute_risk(fraud_prob: float) -> Tuple[str, int]:
    """Map probability to risk level."""
    score = int(fraud_prob * 100)
    if fraud_prob < 0.30:
        level = "Low"
    elif fraud_prob < 0.70:
        level = "Medium"
    else:
        level = "High"
    return level, score


def compute_anomaly_level(anomaly_score: float) -> str:
    """Map anomaly score to category."""
    if anomaly_score < 40:
        return "Normal"
    elif anomaly_score < 70:
        return "Suspicious"
    else:
        return "Highly Anomalous"


def extract_behavioral_flags(features: np.ndarray, anomaly_score: float) -> Dict[str, bool]:
    """
    Analyze transaction features for behavioral anomalies.
    Returns flags for different types of suspicious patterns.
    """
    flags = {
        'unusual_amount': False,
        'unusual_timing': False,
        'unusual_pattern': False,
        'high_risk_features': False
    }
    
    # Amount analysis (assume index 28 is scaled_amount)
    if len(features[0]) > 28:
        amount_val = abs(features[0][28])
        if amount_val > 3.0:  # More than 3 std devs
            flags['unusual_amount'] = True
    
    # Timing analysis (assume index 29 is scaled_time)
    if len(features[0]) > 29:
        time_val = abs(features[0][29])
        if time_val > 2.5:
            flags['unusual_timing'] = True
    
    # Pattern analysis based on V features
    v_features = features[0][:28]
    extreme_count = np.sum(np.abs(v_features) > 3.0)
    if extreme_count >= 3:
        flags['unusual_pattern'] = True
    
    # Overall risk
    if anomaly_score > 70:
        flags['high_risk_features'] = True
    
    return flags


def prepare_features_for_rf(transaction: TransactionRequest) -> np.ndarray:
    """Prepare features for Random Forest (supervised model)."""
    feature_dict = transaction.dict()
    amount = feature_dict.pop("Amount")
    time_  = feature_dict.pop("Time")
    
    # Scale Amount and Time
    scaled_values = rf_scaler.transform([[amount, time_]])
    scaled_amount = scaled_values[0][0]
    scaled_time   = scaled_values[0][1]
    
    # Build feature array
    feature_names = rf_metadata.get("feature_names", [])
    feature_map = {**feature_dict, "scaled_amount": scaled_amount, "scaled_time": scaled_time}
    
    features = np.array([[feature_map[f] for f in feature_names]])
    return features


def prepare_features_for_anomaly(transaction: TransactionRequest) -> np.ndarray:
    """Prepare features for Isolation Forest (anomaly detector)."""
    feature_dict = transaction.dict()
    
    # Build feature array in correct order
    features = []
    for i in range(1, 29):
        features.append(feature_dict[f'V{i}'])
    features.append(feature_dict['Amount'])
    features.append(feature_dict['Time'])
    
    X = np.array([features])
    
    # Scale Amount and Time columns (indices 28, 29)
    X[:, [28, 29]] = anomaly_scaler.transform(X[:, [28, 29]])
    
    return X


def compute_anomaly_score_normalized(model, X: np.ndarray) -> float:
    """
    Convert Isolation Forest decision scores to 0-100 scale.
    """
    decision_score = model.decision_function(X)[0]
    
    # Empirical normalization based on training distribution
    # Typical range: [-0.5, 0.5], but can vary
    # Negative = anomaly, Positive = normal
    
    # Simple linear mapping: normalize to 0-100
    # More negative = higher anomaly score
    normalized = 50 - (decision_score * 100)
    normalized = np.clip(normalized, 0, 100)
    
    return float(normalized)


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    rf_loaded = rf_model is not None
    anomaly_loaded = anomaly_model is not None
    
    return {
        "status": "healthy" if rf_loaded else "degraded",
        "supervised_model_loaded": rf_loaded,
        "anomaly_model_loaded": anomaly_loaded,
        "model_type": rf_metadata.get("model_type", "unknown"),
    }


@app.get("/model-info", tags=["System"])
async def model_info():
    """Return combined model metadata."""
    if not rf_metadata:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "supervised": {k: v for k, v in rf_metadata.items() if k != "y_proba"},
        "anomaly": {k: v for k, v in anomaly_metadata.items() if k != "anomaly_scores"}
    }


@app.get("/anomaly-stats", tags=["Anomaly Detection"])
async def anomaly_stats():
    """Return anomaly detection statistics."""
    if not anomaly_metadata:
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded")
    
    return {
        "model_type": anomaly_metadata.get("model_type"),
        "threshold": anomaly_metadata.get("threshold"),
        "contamination": anomaly_metadata.get("contamination"),
        "metrics": anomaly_metadata.get("metrics", {}),
        "trained_at": anomaly_metadata.get("trained_at")
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_supervised(
    transaction: TransactionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Supervised learning prediction using Random Forest.
    Original endpoint - maintains backward compatibility.
    """
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Supervised model not loaded")
    
    # Prepare features
    features = prepare_features_for_rf(transaction)
    
    # Inference
    t_start = time.perf_counter()
    fraud_proba = float(rf_model.predict_proba(features)[0][1])
    inference_ms = (time.perf_counter() - t_start) * 1000
    
    threshold = rf_metadata.get("threshold", 0.5)
    prediction = int(fraud_proba >= threshold)
    risk_level, risk_score = compute_risk(fraud_proba)
    
    logger.info(
        "Supervised prediction",
        extra={
            "prediction": prediction,
            "fraud_probability": round(fraud_proba, 4),
            "risk_level": risk_level,
            "inference_ms": round(inference_ms, 3),
        },
    )
    
    return PredictionResponse(
        prediction=prediction,
        fraud_probability=round(fraud_proba, 4),
        risk_level=risk_level,
        risk_score=risk_score,
        threshold_used=threshold,
        inference_time_ms=round(inference_ms, 3),
        model_version=rf_metadata.get("model_type", "RandomForest") + "-v1.0",
    )


@app.post("/predict-anomaly", response_model=AnomalyResponse, tags=["Anomaly Detection"])
async def predict_anomaly(
    transaction: TransactionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Unsupervised anomaly detection using Isolation Forest.
    Detects transactions that deviate from normal behavioral patterns.
    """
    if anomaly_model is None:
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded")
    
    # Prepare features
    features = prepare_features_for_anomaly(transaction)
    
    # Inference
    t_start = time.perf_counter()
    
    # Get anomaly score (0-100)
    anomaly_score = compute_anomaly_score_normalized(anomaly_model, features)
    
    # Prediction
    threshold = anomaly_metadata.get("threshold", 50.0)
    is_anomaly = anomaly_score >= threshold
    anomaly_level = compute_anomaly_level(anomaly_score)
    
    # Behavioral analysis
    behavioral_flags = extract_behavioral_flags(features, anomaly_score)
    
    inference_ms = (time.perf_counter() - t_start) * 1000
    
    logger.info(
        "Anomaly prediction",
        extra={
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 2),
            "anomaly_level": anomaly_level,
            "inference_ms": round(inference_ms, 3),
        },
    )
    
    return AnomalyResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 2),
        anomaly_level=anomaly_level,
        threshold_used=threshold,
        inference_time_ms=round(inference_ms, 3),
        behavioral_flags=behavioral_flags,
    )


@app.post("/predict-hybrid", response_model=HybridResponse, tags=["Prediction"])
async def predict_hybrid(
    transaction: TransactionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Hybrid prediction combining supervised and unsupervised models.
    Provides the most robust fraud detection by leveraging both approaches.
    
    Decision Logic:
      - If BOTH models flag as fraud → High confidence fraud
      - If ONE model flags → Medium confidence fraud (manual review)
      - If NEITHER flags → Legitimate transaction
    """
    if rf_model is None or anomaly_model is None:
        raise HTTPException(status_code=503, detail="Models not fully loaded")
    
    t_start = time.perf_counter()
    
    # Get supervised prediction
    rf_features = prepare_features_for_rf(transaction)
    supervised_prob = float(rf_model.predict_proba(rf_features)[0][1])
    supervised_threshold = rf_metadata.get("threshold", 0.5)
    supervised_pred = int(supervised_prob >= supervised_threshold)
    
    # Get anomaly prediction
    anomaly_features = prepare_features_for_anomaly(transaction)
    anomaly_score = compute_anomaly_score_normalized(anomaly_model, anomaly_features)
    anomaly_threshold = anomaly_metadata.get("threshold", 50.0)
    anomaly_pred = int(anomaly_score >= anomaly_threshold)
    
    # Combine predictions
    model_agreement = (supervised_pred == anomaly_pred)
    
    if supervised_pred == 1 and anomaly_pred == 1:
        final_prediction = 1
        confidence = 0.95
        risk_assessment = "CRITICAL: Both models detected fraud"
    elif supervised_pred == 1 or anomaly_pred == 1:
        final_prediction = 1
        confidence = 0.75
        risk_assessment = "HIGH: One model detected fraud - Recommend manual review"
    else:
        final_prediction = 0
        confidence = 0.90
        risk_assessment = "LOW: Transaction appears legitimate"
    
    inference_ms = (time.perf_counter() - t_start) * 1000
    
    logger.info(
        "Hybrid prediction",
        extra={
            "final_prediction": final_prediction,
            "supervised_pred": supervised_pred,
            "anomaly_pred": anomaly_pred,
            "model_agreement": model_agreement,
            "confidence": confidence,
            "inference_ms": round(inference_ms, 3),
        },
    )
    
    return HybridResponse(
        final_prediction=final_prediction,
        confidence=round(confidence, 3),
        supervised_pred=supervised_pred,
        supervised_prob=round(supervised_prob, 4),
        anomaly_score=round(anomaly_score, 2),
        risk_assessment=risk_assessment,
        model_agreement=model_agreement,
        inference_time_ms=round(inference_ms, 3),
    )


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
