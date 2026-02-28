"""
============================================================
Enhanced Fraud Detection API - Sparkov Behavioral Edition
============================================================
This enhanced version includes:
  1. Real-time Feature Engineering (Distance, Age, Time)
  2. Supervised model endpoint (Random Forest)
  3. Anomaly detection endpoint (Isolation Forest)
  4. Hybrid prediction combining both models
============================================================
"""

import os
import json
import time
import logging
from datetime import datetime
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
ENCODER_PATH    = os.path.join(BASE_DIR, "model", "category_encoder.joblib")
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
    title       = "Behavioral Fraud Detection API",
    description = "Real-time fraud detection using geographic and temporal feature engineering",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Models ────────────────────────────────────────────────────────────
models = {}

@app.on_event("startup")
async def load_models():
    # Load supervised model & feature transformers
    try:
        models['rf'] = joblib.load(MODEL_PATH)
        models['rf_scaler'] = joblib.load(SCALER_PATH)
        models['category_encoder'] = joblib.load(ENCODER_PATH)
        with open(META_PATH) as f:
            models['rf_meta'] = json.load(f)
        logger.info("Random Forest & Encoders loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load supervised models: {e}")
    
    # Load anomaly detection model
    try:
        models['iso'] = joblib.load(ANOMALY_MODEL)
        models['iso_scaler'] = joblib.load(ANOMALY_SCALER)
        with open(ANOMALY_META) as f:
            models['iso_meta'] = json.load(f)
        logger.info("Isolation Forest loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load anomaly detector: {e}")


# ════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class TransactionRequest(BaseModel):
    """Raw Sparkov transaction features from the frontend."""
    trans_date_trans_time : str = Field(..., description="Format: YYYY-MM-DD HH:MM:SS")
    amt                   : float = Field(..., ge=0)
    category              : str
    dob                   : str = Field(..., description="Format: YYYY-MM-DD")
    lat                   : float
    long                  : float
    merch_lat             : float
    merch_long            : float

class PredictionResponse(BaseModel):
    prediction       : int
    fraud_probability: float
    risk_level       : str
    threshold_used   : float
    inference_time_ms: float
    distance_km_proxy: float

class AnomalyResponse(BaseModel):
    is_anomaly       : bool
    anomaly_score    : float
    anomaly_level    : str
    inference_time_ms: float
    distance_km_proxy: float

class HybridResponse(BaseModel):
    final_prediction : int
    confidence       : float
    supervised_pred  : int
    supervised_prob  : float
    anomaly_score    : float
    risk_assessment  : str
    model_agreement  : bool
    inference_time_ms: float
    distance_km_proxy: float


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (FEATURE ENGINEERING)
# ════════════════════════════════════════════════════════════════════════════

def extract_features(req: TransactionRequest) -> Tuple[np.ndarray, float]:
    """
    Translates raw human data into the exact mathematical features the models expect.
    Expected Order: ['Amount', 'Hour_of_Day', 'Day_of_Week', 'Distance_Proxy', 'Age', 'category_encoded']
    """
    # 1. Temporal Math
    trans_dt = datetime.strptime(req.trans_date_trans_time, "%Y-%m-%d %H:%M:%S")
    dob_dt = datetime.strptime(req.dob, "%Y-%m-%d")
    
    hour = trans_dt.hour
    day_of_week = trans_dt.weekday()
    
    # 2. Demographic Math
    age = (trans_dt - dob_dt).days // 365
    
    # 3. Geographic Math
    distance_proxy = np.sqrt((req.lat - req.merch_lat)**2 + (req.long - req.merch_long)**2)
    distance_km = distance_proxy * 111  # Rough conversion proxy for the UI
    
    # 4. Categorical Encoding
    try:
        cat_enc = models['category_encoder'].transform([req.category])
    except Exception:
        cat_enc = -1 # Handle unseen categories securely

    # Build final array
    features = np.array([[req.amt, hour, day_of_week, distance_proxy, age, cat_enc]])
    return features, distance_km

def compute_anomaly_score_normalized(decision_score: float) -> float:
    """Normalize the raw isolation forest score to a 0-100 gauge."""
    meta = models.get('iso_meta', {})
    min_score = meta.get('min_score', -0.15)
    max_score = meta.get('max_score', 0.05)
    
    normalized = 100 * (max_score - decision_score) / (max_score - min_score + 1e-6)
    return float(np.clip(normalized, 0, 100))


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if 'rf' in models else "degraded",
        "supervised_loaded": 'rf' in models,
        "anomaly_loaded": 'iso' in models
    }

@app.get("/anomaly-stats", tags=["System"])
async def anomaly_stats():
    if 'iso_meta' not in models: return {}
    return models['iso_meta']

@app.get("/model-info", tags=["System"])
async def model_info():
    return {
        "supervised": {k: v for k, v in models.get('rf_meta', {}).items() if k != "metrics"},
        "anomaly": {k: v for k, v in models.get('iso_meta', {}).items() if k != "metrics"}
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_supervised(req: TransactionRequest, api_key: str = Depends(verify_api_key)):
    t_start = time.perf_counter()
    raw_features, distance_km = extract_features(req)
    
    # Scale & Predict
    scaled_features = models['rf_scaler'].transform(raw_features)
    fraud_proba = float(models['rf'].predict_proba(scaled_features))
    
    threshold = models['rf_meta'].get("threshold", 0.5)
    prediction = int(fraud_proba >= threshold)
    
    risk_level = "High" if fraud_proba > 0.7 else ("Medium" if fraud_proba > 0.3 else "Low")
    inference_ms = (time.perf_counter() - t_start) * 1000
    
    return PredictionResponse(
        prediction=prediction, fraud_probability=round(fraud_proba, 4),
        risk_level=risk_level, threshold_used=threshold,
        inference_time_ms=round(inference_ms, 3), distance_km_proxy=round(distance_km, 1)
    )

@app.post("/predict-anomaly", response_model=AnomalyResponse, tags=["Prediction"])
async def predict_anomaly(req: TransactionRequest, api_key: str = Depends(verify_api_key)):
    t_start = time.perf_counter()
    raw_features, distance_km = extract_features(req)
    
    # Scale & Predict
    scaled_features = models['iso_scaler'].transform(raw_features)
    decision = float(models['iso'].decision_function(scaled_features))
    anomaly_score = compute_anomaly_score_normalized(decision)
    
    threshold = models['iso_meta'].get("threshold", 50.0)
    is_anomaly = anomaly_score >= threshold
    anomaly_level = "Highly Anomalous" if anomaly_score > 70 else ("Suspicious" if anomaly_score > 40 else "Normal")
    
    inference_ms = (time.perf_counter() - t_start) * 1000
    
    return AnomalyResponse(
        is_anomaly=is_anomaly, anomaly_score=round(anomaly_score, 2),
        anomaly_level=anomaly_level, inference_time_ms=round(inference_ms, 3),
        distance_km_proxy=round(distance_km, 1)
    )

@app.post("/predict-hybrid", response_model=HybridResponse, tags=["Prediction"])
async def predict_hybrid(req: TransactionRequest, api_key: str = Depends(verify_api_key)):
    t_start = time.perf_counter()
    raw_features, distance_km = extract_features(req)
    
    # 1. Supervised Pass
    rf_scaled = models['rf_scaler'].transform(raw_features)
    supervised_prob = float(models['rf'].predict_proba(rf_scaled))
    supervised_pred = int(supervised_prob >= models['rf_meta'].get("threshold", 0.5))
    
    # 2. Anomaly Pass
    iso_scaled = models['iso_scaler'].transform(raw_features)
    decision = float(models['iso'].decision_function(iso_scaled))
    anomaly_score = compute_anomaly_score_normalized(decision)
    anomaly_pred = int(anomaly_score >= models['iso_meta'].get("threshold", 50.0))
    
    # 3. Hybrid Logic
    model_agreement = (supervised_pred == anomaly_pred)
    if supervised_pred and anomaly_pred:
        final_pred, confidence, assessment = 1, 0.95, "CRITICAL: Both models detected fraud"
    elif supervised_pred or anomaly_pred:
        final_pred, confidence, assessment = 1, 0.75, "HIGH: One model flagged issue - Review required"
    else:
        final_pred, confidence, assessment = 0, 0.90, "LOW: Transaction appears legitimate"
    
    inference_ms = (time.perf_counter() - t_start) * 1000
    
    return HybridResponse(
        final_prediction=final_pred, confidence=confidence,
        supervised_pred=supervised_pred, supervised_prob=round(supervised_prob, 4),
        anomaly_score=round(anomaly_score, 2), risk_assessment=assessment,
        model_agreement=model_agreement, inference_time_ms=round(inference_ms, 3),
        distance_km_proxy=round(distance_km, 1)
    )

if __name__ == "__main__":
    import uvicorn
    # Fixed the module name to point directly to main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)