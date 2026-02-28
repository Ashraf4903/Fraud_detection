# 🛡️ AI-Based Fraud Detection System

> Real-time credit card fraud detection powered by **Random Forest**, **SMOTE**, **FastAPI**, and **Streamlit**.  
> Built for hackathon presentation — clean, modular, and production-ready.

---

## 📐 Project Architecture

```
fraud_detection/
├── data/
│   └── creditcard.csv          ← Kaggle dataset (download separately)
│
├── model/
│   ├── train.py                ← Full ML pipeline (preprocessing → training → evaluation → save)
│   ├── fraud_model.joblib      ← Trained Random Forest model (generated)
│   ├── scaler.joblib           ← Fitted StandardScaler (generated)
│   ├── metadata.json           ← Threshold, feature names, metrics (generated)
│   └── plots/                  ← Confusion matrix, ROC curve, feature importance (generated)
│
├── api/
│   └── main.py                 ← FastAPI REST service (POST /predict, GET /health)
│
├── frontend/
│   └── app.py                  ← Streamlit dashboard (form, gauge, history)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd fraud_detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the Dataset

1. Go to → https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it inside `data/creditcard.csv`

### 3. Train the Model

```bash
python model/train.py
```

This will:
- Load and preprocess the data
- Apply SMOTE oversampling
- Train Logistic Regression + Random Forest
- Print a model comparison results table
- Run threshold tuning
- Save `fraud_model.joblib`, `scaler.joblib`, `metadata.json`
- Generate diagnostic plots in `model/plots/`

Expected output (last lines):
```
MODEL COMPARISON RESULTS TABLE
======================================================================
Model                     Precision     Recall         F1    ROC-AUC
----------------------------------------------------------------------
Logistic Regression          0.0678     0.9184     0.1264     0.9693
Random Forest                0.8372     0.8367     0.8369     0.9753
======================================================================
```

### 4. Start the API

```bash
# From the project root
uvicorn api.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Model info:  http://localhost:8000/model-info

### 5. Start the Frontend

```bash
streamlit run frontend/app.py
```

Open → http://localhost:8501

---

## 🔐 API Usage

### Authentication

All `/predict` requests require an `X-API-Key` header:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key-change-in-production" \
  -d '{
    "Time": 406, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46,  "V7": 0.24,  "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21,  "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28,  "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13,  "V28": -0.02
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "fraud_probability": 0.0034,
  "risk_level": "Low",
  "risk_score": 0,
  "threshold_used": 0.47,
  "inference_time_ms": 2.14,
  "model_version": "RandomForest-v1.0"
}
```

---

## 📊 Model Pipeline Explained

### Data Preprocessing
- **StandardScaler** applied to `Amount` and `Time` (V1–V28 already PCA-scaled)
- **Stratified train/test split** (80/20) preserving class ratio

### Handling Class Imbalance
The dataset is severely imbalanced (0.17% fraud).  
We use **SMOTE** (Synthetic Minority Over-sampling TEchnique):
- Generates synthetic fraud samples in feature space
- Balances the training set to 50/50
- Applied **only to training data** (no leakage)

### Models Trained
| Model | Role | Notes |
|-------|------|-------|
| Logistic Regression | Baseline | Fast, interpretable, lower recall |
| Random Forest | Main | High recall, robust to outliers |

### Threshold Tuning
Default classification threshold = 0.5.  
We sweep thresholds 0.10–0.90 and pick the value that maximises F1 (or recall).  
Lower threshold → higher recall → fewer missed frauds.

### Risk Scoring System
| Fraud Probability | Risk Level | Score |
|-------------------|------------|-------|
| 0% – 30% | 🟢 Low | 0–29 |
| 30% – 70% | 🟡 Medium | 30–69 |
| 70% – 100% | 🔴 High | 70–100 |

---

## ⏱️ Inference Time

Random Forest inference on a single transaction: **~2–5 ms** (measured on CPU).

The model is loaded once at API startup and held in memory — no disk I/O per request.  
This translates to **>500 predictions/second** on a single CPU core.

---

## 🔒 Security Recommendations

| Layer | Recommendation |
|-------|----------------|
| API Key | Store in env var / secrets manager (never hardcode) |
| Rate Limiting | Redis + sliding window (e.g. `slowapi` for FastAPI) |
| HTTPS | Terminate TLS at load balancer (nginx / AWS ALB) |
| Auth | Upgrade to OAuth2 + JWT for multi-tenant |
| Input Validation | Pydantic enforces types & ranges (already implemented) |
| Logging | Structured JSON logs → ship to ELK / Datadog / CloudWatch |
| CORS | Restrict `allow_origins` to your frontend URL in production |

---

## 🚀 Future Enhancements

### 1. Cloud Deployment
```
AWS Lambda + API Gateway          # Serverless, auto-scaling
Google Cloud Run                  # Container-based, per-request billing
Azure App Service                 # Managed PaaS option
```

### 2. Docker / Containerization

```dockerfile
# Example Dockerfile (api)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - FRAUD_API_KEY=${FRAUD_API_KEY}
  frontend:
    build: ./frontend
    ports: ["8501:8501"]
    environment:
      - API_URL=http://api:8000
```

### 3. Model Improvements
- **XGBoost / LightGBM** for faster inference + higher accuracy
- **Isolation Forest** for unsupervised anomaly detection
- **AutoML** (H2O, AutoGluon) for automated hyperparameter tuning
- **Online learning** to retrain on new fraud patterns in real-time
- **SHAP values** for per-prediction explainability

### 4. Production Scaling
- **Model serving**: MLflow + Seldon Core / BentoML
- **Feature store**: Feast for real-time feature retrieval
- **Kafka** for streaming transaction ingestion
- **A/B testing**: Shadow mode to compare models before promotion
- **Drift detection**: Evidently AI to alert on data distribution shifts

### 5. Monitoring
- **Grafana + Prometheus** dashboards for prediction volume, latency, fraud rate
- **MLflow** experiment tracking for model versions
- **PagerDuty** alerts when fraud rate spikes unexpectedly

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Model Persistence | joblib |
| REST API | FastAPI + Uvicorn |
| Input Validation | Pydantic v2 |
| Frontend | Streamlit |
| Visualisation | Plotly, Seaborn, Matplotlib |
| Logging | python-json-logger |
| Security | API Key header, Pydantic validation |

---

## 📄 License

MIT — free to use for hackathons, demos, and learning.

---

*Built with ❤️ for hackathon presentation.*
