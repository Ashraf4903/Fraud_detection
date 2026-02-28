# 🛡️ Enhanced AI-Based Fraud Detection System

## Level 2 Implementation: Anomaly Detection + Visualization

A production-ready fraud detection system combining **supervised learning** (Random Forest) and **unsupervised anomaly detection** (Isolation Forest) to protect financial transactions.

---

## 🎯 Project Overview

### What's New in Level 2

This enhanced version adds a complete **Anomaly Detection Layer** that complements the existing supervised classifier:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Supervised Classifier** | Random Forest | Detects known fraud patterns from labeled data |
| **Anomaly Detector** | Isolation Forest | Identifies unusual behavioral deviations |
| **Hybrid System** | Combined Logic | Maximum fraud protection |

### Key Features ✨

1. **Dual-Layer Protection**
   - Supervised learning for known fraud patterns (98.48% ROC-AUC)
   - Anomaly detection for behavioral deviations
   - Hybrid prediction combining both models

2. **Interactive Visualizations**
   - Anomaly score distribution charts
   - Temporal pattern analysis
   - PCA-based anomaly visualization
   - Behavioral flag indicators

3. **Production-Ready API**
   - `/predict` - Supervised classification
   - `/predict-anomaly` - Anomaly detection
   - `/predict-hybrid` - Combined prediction
   - `/anomaly-stats` - Model statistics

4. **Enhanced Dashboard**
   - Real-time anomaly scoring
   - Behavioral pattern analysis
   - Model comparison metrics
   - Interactive gauges and charts

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── data/
│   └── creditcard.csv              # Kaggle dataset (download separately)
│
├── model/
│   ├── fraud_model.joblib          # Random Forest classifier
│   ├── scaler.joblib               # Feature scaler
│   ├── metadata.json               # Model metadata
│   │
│   ├── anomaly/                    # NEW: Anomaly detection models
│   │   ├── isolation_forest.joblib
│   │   ├── anomaly_scaler.joblib
│   │   ├── anomaly_metadata.json
│   │   └── plots/                  # Visualization outputs
│   │       ├── anomaly_distribution.png
│   │       ├── amount_vs_anomaly.png
│   │       ├── time_vs_anomaly.png
│   │       ├── pca_anomalies.png
│   │       └── top_anomalies.png
│   │
│   └── plots/                      # Supervised model plots
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       └── ...
│
├── api/
│   ├── main.py                     # Original API
│   └── main_enhanced.py            # NEW: Enhanced API with anomaly detection
│
├── frontend/
│   ├── app.py                      # Original Streamlit app
│   └── app_enhanced.py             # NEW: Enhanced app with visualizations
│
├── train.py                        # Train supervised model
├── train_anomaly_detector.py       # NEW: Train anomaly detector
│
└── README_ENHANCED.md              # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Required Packages:**
```
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
streamlit
plotly
requests
imbalanced-learn
python-json-logger
```

### Step 1: Download Dataset

Download the Kaggle Credit Card Fraud Detection dataset:
- URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
- Place `creditcard.csv` in the `data/` folder

### Step 2: Train Both Models

```bash
# Train supervised model (Random Forest)
python train.py

# Train anomaly detector (Isolation Forest) - NEW
python train_anomaly_detector.py
```

Expected outputs:
- `model/fraud_model.joblib` - Supervised classifier
- `model/anomaly/isolation_forest.joblib` - Anomaly detector
- Visualization plots in respective `plots/` folders

### Step 3: Start Enhanced API

```bash
# Start the enhanced API server
uvicorn main_enhanced:app --reload --port 8000
```

API will be available at: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Step 4: Launch Enhanced Dashboard

```bash
# In a new terminal
streamlit run app_enhanced.py
```

Dashboard will open at: `http://localhost:8501`

---

## 🎛️ API Endpoints

### Original Endpoint (Backward Compatible)

**POST /predict**
```json
{
  "Time": 406.0,
  "Amount": 149.62,
  "V1": -1.36,
  "V2": -0.07,
  ...
  "V28": -0.02
}
```

Response:
```json
{
  "prediction": 0,
  "fraud_probability": 0.0234,
  "risk_level": "Low",
  "risk_score": 2,
  "threshold_used": 0.85,
  "inference_time_ms": 2.34,
  "model_version": "RandomForestClassifier-v1.0"
}
```

### NEW: Anomaly Detection

**POST /predict-anomaly**

Same input format as `/predict`

Response:
```json
{
  "is_anomaly": false,
  "anomaly_score": 23.45,
  "anomaly_level": "Normal",
  "threshold_used": 50.0,
  "inference_time_ms": 1.89,
  "behavioral_flags": {
    "unusual_amount": false,
    "unusual_timing": false,
    "unusual_pattern": false,
    "high_risk_features": false
  }
}
```

### NEW: Hybrid Prediction

**POST /predict-hybrid**

Same input format

Response:
```json
{
  "final_prediction": 0,
  "confidence": 0.95,
  "supervised_pred": 0,
  "supervised_prob": 0.0234,
  "anomaly_score": 23.45,
  "risk_assessment": "LOW: Transaction appears legitimate",
  "model_agreement": true,
  "inference_time_ms": 4.12
}
```

### Statistics

**GET /anomaly-stats**

Returns anomaly model metrics and configuration.

---

## 📊 Visualizations Explained

### 1. Anomaly Score Distribution
Shows how legitimate vs fraudulent transactions score in the anomaly detection system. Helps understand model behavior.

### 2. Amount vs Anomaly Score
Scatter plot showing relationship between transaction amount and anomaly score. Identifies amount-based anomalies.

### 3. Temporal Pattern Analysis
Time-series view of anomalies. Detects if fraud attempts cluster at specific times.

### 4. PCA Space Visualization
2D projection of high-dimensional data. Visually separates normal behavior from anomalies.

### 5. Top Anomalies Feature Heatmap
Shows which features contribute most to the highest anomaly scores.

---

## 🧠 Model Architecture

### Hybrid Detection Strategy

```
Transaction Input
        │
        ├─────────────────────┬─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
  Random Forest      Isolation Forest      Feature Analysis
  (Supervised)         (Anomaly)           (Behavioral)
        │                     │                     │
        ├─────────────────────┴─────────────────────┤
        │                                           │
        ▼                                           ▼
   Decision Logic                          Risk Assessment
        │
        ▼
  Final Verdict
```

### Decision Logic

| Supervised | Anomaly | Final Decision | Confidence |
|------------|---------|----------------|------------|
| Fraud | Fraud | **Fraud** | 95% |
| Fraud | Normal | **Fraud** (Review) | 75% |
| Normal | Fraud | **Fraud** (Review) | 75% |
| Normal | Normal | **Legitimate** | 90% |

---

## 🔬 Anomaly Detection Deep Dive

### How Isolation Forest Works

1. **Isolation Trees**: Randomly select features and split values
2. **Path Length**: Measure steps needed to isolate each sample
3. **Anomaly Score**: Shorter paths = anomalies (easier to isolate)

### Why It's Effective for Fraud

- **No Labels Needed**: Learns normal behavior from legitimate transactions
- **Novel Fraud**: Detects new attack patterns not in training data
- **Behavioral Focus**: Identifies deviations from user's typical behavior

### Behavioral Flags

The system tracks 4 types of anomalies:

1. **Unusual Amount**: Transaction amount significantly outside normal range
2. **Unusual Timing**: Transaction at atypical time
3. **Unusual Pattern**: Multiple PCA features showing extreme values
4. **High Risk Features**: Overall anomaly score above critical threshold

---

## 📈 Performance Metrics

### Supervised Model (Random Forest)
- **Precision**: 96.1%
- **Recall**: 75.5%
- **F1 Score**: 84.6%
- **ROC-AUC**: 98.5%

### Anomaly Detector (Isolation Forest)
*Performance varies based on threshold tuning*

Typical results:
- **Precision**: 85-92%
- **Recall**: 60-75%
- **F1 Score**: 70-80%
- **ROC-AUC**: 92-95%

### Hybrid System
- **Best Precision**: 97%+
- **Best Recall**: 80%+
- **Catches Novel Fraud**: ✓

---

## 🛠️ Configuration

### Anomaly Detection Threshold

Adjust in `model/anomaly/anomaly_metadata.json`:

```json
{
  "threshold": 50.0  // Lower = more sensitive (more flags)
}
```

**Tuning Guide:**
- Threshold 30-40: High sensitivity, more false positives
- Threshold 50-60: Balanced (recommended)
- Threshold 70-80: Conservative, fewer false positives

### Hybrid Decision Logic

Modify in `main_enhanced.py`:

```python
if supervised_pred == 1 and anomaly_pred == 1:
    final_prediction = 1
    confidence = 0.95
```

---

## 🎨 Dashboard Features

### Mode Selection
- **Supervised Only**: Use Random Forest classifier
- **Anomaly Detection**: Use Isolation Forest only
- **Hybrid (Recommended)**: Combined analysis

### Visualizations
- Real-time gauge charts for both models
- Behavioral flag analysis
- Side-by-side model comparison
- Interactive anomaly plots

---

## 🚦 Use Cases

### When to Use Each Model

**Supervised (Random Forest):**
- High-stakes transactions where precision is critical
- When you have labeled training data
- Known fraud pattern detection

**Anomaly Detection (Isolation Forest):**
- Detecting zero-day attacks
- Monitoring for behavioral changes
- When fraud labels are scarce

**Hybrid (Recommended):**
- Maximum protection
- Production deployment
- Regulatory compliance scenarios

---

## 🔐 Security Best Practices

1. **API Key Rotation**: Change default key in production
2. **Rate Limiting**: Implement per-client rate limits
3. **Logging**: Enable structured logging for audit trails
4. **Model Updates**: Retrain regularly with new data
5. **Threshold Monitoring**: Track false positive/negative rates

---

## 📝 Expected Outcomes (Level 2 Requirements)

✅ **Transaction Classifier** - Random Forest with 98.5% ROC-AUC  
✅ **Anomaly Detection** - Isolation Forest for behavioral deviations  
✅ **Visualization of Anomalies** - 5+ interactive plots showing:
   - Distribution analysis
   - Temporal patterns
   - PCA projections
   - Feature importance
   - Behavioral flags

---

## 🐛 Troubleshooting

### Issue: Models not loading
```bash
# Ensure both training scripts have been run
python train.py
python train_anomaly_detector.py
```

### Issue: API connection errors
```bash
# Check API is running
curl http://localhost:8000/health

# Verify API key matches
export FRAUD_API_KEY="your-key-here"
```

### Issue: Missing visualizations
```bash
# Retrain anomaly detector to generate plots
python train_anomaly_detector.py
```

---

## 📚 References

- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Random Forest**: Supervised classification algorithm
- **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)
- **SMOTE**: Chawla et al. (2002)

---

## 🤝 Contributing

To extend this system:

1. Add new anomaly detection algorithms (Autoencoder, LOF, etc.)
2. Implement real-time streaming analysis
3. Add user-specific behavioral profiles
4. Integrate with external APIs (address verification, device fingerprinting)

---

## 📄 License

This project is for educational and research purposes.

---

## 👥 Support

For questions or issues:
1. Check API documentation at `/docs`
2. Review training logs in console output
3. Inspect visualization plots for model behavior

---

**Built with ❤️ for Level 2 Fraud Detection**

*Protecting financial transactions with dual-layer AI defense*
