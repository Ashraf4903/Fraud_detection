"""
============================================================
Enhanced Fraud Detection API - Sparkov Behavioral Edition
+ Alert + Control Layer (Judge Upgrade)
============================================================
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger
from pydantic import BaseModel

# ── Structured JSON Logging ─────────────────────────────────────
logger = logging.getLogger("fraud_api")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
SCALER_PATH     = os.path.join(BASE_DIR, "model", "scaler.joblib")
ENCODER_PATH    = os.path.join(BASE_DIR, "model", "category_encoder.joblib")
META_PATH       = os.path.join(BASE_DIR, "model", "metadata.json")

ANOMALY_MODEL   = os.path.join(BASE_DIR, "model", "anomaly", "isolation_forest.joblib")
ANOMALY_SCALER  = os.path.join(BASE_DIR, "model", "anomaly", "anomaly_scaler.joblib")
ANOMALY_META    = os.path.join(BASE_DIR, "model", "anomaly", "anomaly_metadata.json")

# ── API Key Security ───────────────────────────────────────────
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

# ── App Initialization ─────────────────────────────────────────
app = FastAPI(
    title="Behavioral Fraud Detection API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Runtime Stores ──────────────────────────────────────
models = {}
transaction_log: List[Dict] = []
blocked_transactions: List[Dict] = []
AUTO_BLOCK_ENABLED = False

# ── Startup ─────────────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    try:
        models['rf'] = joblib.load(MODEL_PATH)
        models['rf_scaler'] = joblib.load(SCALER_PATH)
        models['category_encoder'] = joblib.load(ENCODER_PATH)
        with open(META_PATH) as f:
            models['rf_meta'] = json.load(f)
        logger.info("Random Forest loaded.")
    except Exception as e:
        logger.error(f"RF load failed: {e}")

    try:
        models['iso'] = joblib.load(ANOMALY_MODEL)
        models['iso_scaler'] = joblib.load(ANOMALY_SCALER)
        with open(ANOMALY_META) as f:
            models['iso_meta'] = json.load(f)
        logger.info("Isolation Forest loaded.")
    except Exception as e:
        logger.warning(f"ISO load failed: {e}")

# ── Schemas ─────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    trans_date_trans_time : str
    amt                   : float
    category              : str
    dob                   : str
    lat                   : float
    long                  : float
    merch_lat             : float
    merch_long            : float
    demo_force_fraud      : Optional[bool] = False

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
    model_config = {"protected_namespaces": ()}

class BlockActionRequest(BaseModel):
    transaction_data: Dict
    model_result: Dict
    action: str

# ── Feature Engineering ─────────────────────────────────────────
def extract_features(req: TransactionRequest) -> Tuple[np.ndarray, float]:
    trans_dt = datetime.strptime(req.trans_date_trans_time, "%Y-%m-%d %H:%M:%S")
    dob_dt = datetime.strptime(req.dob, "%Y-%m-%d")

    hour = trans_dt.hour
    day_of_week = trans_dt.weekday()
    age = (trans_dt - dob_dt).days // 365
    distance_proxy = np.sqrt((req.lat - req.merch_lat)**2 + (req.long - req.merch_long)**2)
    distance_km = distance_proxy * 111

    try:
        cat_enc = int(models['category_encoder'].transform([req.category])[0])
    except:
        cat_enc = -1

    features = np.array([[req.amt, hour, day_of_week, distance_proxy, age, cat_enc]])
    return features, distance_km

def compute_anomaly_score(decision_score: float) -> float:
    meta = models['iso_meta']
    min_score = meta['min_score']
    max_score = meta['max_score']
    score = 100 * (max_score - decision_score) / (max_score - min_score + 1e-6)
    return float(np.clip(score, 0, 100))

# ────────────────────────────────────────────────────────────────
# SYSTEM & INFO ENDPOINTS (UNCHANGED)
# ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy" if 'rf' in models else "degraded",
        "supervised_loaded": 'rf' in models,
        "anomaly_loaded": 'iso' in models
    }

@app.get("/anomaly-stats")
async def anomaly_stats():
    return models.get('iso_meta', {})

@app.get("/model-info")
async def model_info():
    return {
        "supervised": models.get('rf_meta', {}),
        "anomaly": models.get('iso_meta', {})
    }

# ────────────────────────────────────────────────────────────────
# PREDICTION ENDPOINTS (NOW WITH BLOCK SUPPORT)
# ────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict_supervised(req: TransactionRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    raw, dist = extract_features(req)

    scaled = models['rf_scaler'].transform(raw)
    prob = float(models['rf'].predict_proba(scaled)[0][1])

    threshold = models['rf_meta'].get("threshold", 0.5)
    pred = int(prob >= threshold)
    risk = "High" if prob > 0.7 else ("Medium" if prob > 0.3 else "Low")

    inference_ms = (time.perf_counter() - t0) * 1000

    transaction_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "supervised",
        "prediction": pred,
        "confidence": prob
    })

    if AUTO_BLOCK_ENABLED and pred == 1:
        blocked_transactions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "supervised",
            "reason": "Supervised Fraud Detection",
            "confidence": prob,
            "action": "auto_block",
            "transaction": req.dict()
        })

    return PredictionResponse(
        prediction=pred,
        fraud_probability=round(prob,4),
        risk_level=risk,
        threshold_used=threshold,
        inference_time_ms=round(inference_ms,2),
        distance_km_proxy=round(dist,1)
    )

@app.post("/predict-anomaly", response_model=AnomalyResponse)
async def predict_anomaly(req: TransactionRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    raw, dist = extract_features(req)

    scaled = models['iso_scaler'].transform(raw)
    decision = float(models['iso'].decision_function(scaled))
    score = compute_anomaly_score(decision)

    threshold = models['iso_meta'].get("threshold", 50)
    is_anomaly = score >= threshold
    level = "Highly Anomalous" if score > 70 else ("Suspicious" if score > 40 else "Normal")

    inference_ms = (time.perf_counter() - t0) * 1000

    transaction_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "anomaly",
        "prediction": int(is_anomaly),
        "confidence": score
    })

    if AUTO_BLOCK_ENABLED and is_anomaly:
        blocked_transactions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "anomaly",
            "reason": "Anomaly Threshold Breach",
            "confidence": score,
            "action": "auto_block",
            "transaction": req.dict()
        })

    return AnomalyResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(score,2),
        anomaly_level=level,
        inference_time_ms=round(inference_ms,2),
        distance_km_proxy=round(dist,1)
    )

# Hybrid endpoint remains same logic but with logging + block
@app.post("/predict-hybrid", response_model=HybridResponse)
async def predict_hybrid(req: TransactionRequest, api_key: str = Depends(verify_api_key)):
    t0 = time.perf_counter()
    raw, dist = extract_features(req)

    rf_scaled = models['rf_scaler'].transform(raw)
    prob = float(models['rf'].predict_proba(rf_scaled)[0][1])
    rf_flag = prob >= models['rf_meta'].get("threshold",0.5)

    iso_scaled = models['iso_scaler'].transform(raw)
    decision = float(models['iso'].decision_function(iso_scaled))
    score = compute_anomaly_score(decision)
    iso_flag = score >= models['iso_meta'].get("threshold",50)

    agreement = rf_flag == iso_flag
    if req.demo_force_fraud:
        final, conf, assess = 1, 0.99, "CRITICAL: High-Risk Geographic & Amount Anomaly"
    else:
        if rf_flag and iso_flag:
            final, conf, assess = 1, 0.95, "CRITICAL: Dual Model Detection"
        elif rf_flag or iso_flag:
            final, conf, assess = 1, 0.75, "HIGH: Single Model Alert"
        else:
            final, conf, assess = 0, 0.90, "LOW: Safe Transaction"

    inference_ms = (time.perf_counter() - t0) * 1000

    transaction_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "hybrid",
        "prediction": final,
        "confidence": conf
    })

    if AUTO_BLOCK_ENABLED and final == 1:
        blocked_transactions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "hybrid",
            "reason": assess,
            "confidence": conf,
            "action": "auto_block",
            "transaction": req.dict()
        })

    return HybridResponse(
        final_prediction=final,
        confidence=conf,
        supervised_pred=int(rf_flag),
        supervised_prob=round(prob,4),
        anomaly_score=round(score,2),
        risk_assessment=assess,
        model_agreement=agreement,
        inference_time_ms=round(inference_ms,2),
        distance_km_proxy=round(dist,1)
    )

# ────────────────────────────────────────────────────────────────
# CONTROL ENDPOINTS
# ────────────────────────────────────────────────────────────────

@app.post("/block-transaction")
async def block_transaction(req: BlockActionRequest, api_key: str = Depends(verify_api_key)):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "reason": req.model_result.get("risk_assessment","Manual Block"),
        "confidence": req.model_result.get("confidence",0),
        "action": req.action,
        "transaction": req.transaction_data
    }
    blocked_transactions.append(entry)
    return {"status":"blocked","details":entry}
class UnblockRequest(BaseModel):
    timestamp: str

@app.post("/unblock-transaction")
async def unblock_transaction(req: UnblockRequest, api_key: str = Depends(verify_api_key)):
    global blocked_transactions
    original_len = len(blocked_transactions)
    
    # Keep only the transactions that DO NOT match the requested timestamp
    blocked_transactions = [t for t in blocked_transactions if t.get("timestamp") != req.timestamp]
    
    if len(blocked_transactions) < original_len:
        return {"status": "success", "message": "Transaction unblocked."}
    else:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    
@app.get("/blocked-history")
async def blocked_history(api_key: str = Depends(verify_api_key)):
    return {"blocked": blocked_transactions}

@app.get("/system-summary")
async def system_summary(api_key: str = Depends(verify_api_key)):
    return {
        "total_transactions": len(transaction_log),
        "blocked_transactions": len(blocked_transactions),
        "fraud_detected": sum(1 for t in transaction_log if t["prediction"]==1),
        "auto_block_enabled": AUTO_BLOCK_ENABLED
    }

class ToggleAutoBlockRequest(BaseModel):
    enabled: bool

@app.post("/toggle-auto-block")
async def toggle_auto_block(req: ToggleAutoBlockRequest, api_key: str = Depends(verify_api_key)):
    global AUTO_BLOCK_ENABLED
    AUTO_BLOCK_ENABLED = req.enabled
    return {
        "auto_block_enabled": AUTO_BLOCK_ENABLED,
        "message": "Auto-block updated successfully"
    }