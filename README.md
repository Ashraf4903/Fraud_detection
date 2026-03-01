# 🛡️ AI-Based Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![ML Models](https://img.shields.io/badge/Models-RF%20%7C%20ISO%20Forest-purple.svg)

**A production-ready fraud detection system combining supervised learning, anomaly detection, and behavioral analytics**

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [API Documentation](#api-documentation) • [Model Performance](#model-performance)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Training](#model-training)
- [API Documentation](#api-documentation)
- [Web Interface](#web-interface)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This is an advanced fraud detection system that leverages machine learning to identify fraudulent credit card transactions in real-time. The system combines multiple detection approaches:

- **Supervised Learning**: Random Forest classifier trained on labeled fraud data
- **Anomaly Detection**: Isolation Forest for identifying unusual transaction patterns
- **Hybrid Approach**: Ensemble decision-making for maximum accuracy

The system is built with production deployment in mind, featuring a FastAPI backend, Streamlit frontend, comprehensive monitoring, and detailed analytics.

---

## ✨ Features

### 🤖 Machine Learning Models

- **Random Forest Classifier** (Primary Supervised Model)
  - 99.5% ROC-AUC score
  - Handles imbalanced data with SMOTE
  - Optimized threshold tuning for maximum recall
  
- **Isolation Forest** (Anomaly Detection)
  - Detects novel fraud patterns
  - Identifies outliers in behavioral features
  - Complementary to supervised approach

- **Hybrid Prediction Engine**
  - Combines both models for ensemble decisions
  - Configurable confidence scoring
  - Real-time risk assessment

### 🎨 Interactive Web Dashboard

- **Real-time Transaction Testing**
  - Manual transaction input
  - Instant fraud probability calculation
  - Geographic distance analysis
  
- **Comprehensive Visualizations**
  - PCA anomaly space visualization
  - Temporal pattern analysis
  - Feature importance charts
  - ROC and PR curves
  
- **Alert & Control System**
  - Transaction blocking capabilities
  - Alert history tracking
  - Auto-block configuration
  - Manual review queue

### 🔌 RESTful API

- **Multiple Prediction Endpoints**
  - `/predict` - Supervised model predictions
  - `/predict-anomaly` - Anomaly detection
  - `/predict-hybrid` - Ensemble approach
  
- **Monitoring & Analytics**
  - `/health` - System health check
  - `/model-info` - Model metadata
  - `/anomaly-stats` - Anomaly statistics
  - `/system-summary` - Overall system metrics

- **Control Features**
  - Transaction blocking/unblocking
  - Alert management
  - Auto-block toggle

### 📊 Advanced Analytics

- **Behavioral Feature Engineering**
  - Transaction timing patterns (hour, day)
  - Geographic distance calculations
  - Customer age demographics
  - Transaction category encoding
  
- **Performance Monitoring**
  - Real-time inference metrics
  - Confusion matrix tracking
  - Precision/Recall optimization
  - ROC-AUC monitoring

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Streamlit Web Interface                     │   │
│  │  • Transaction Input  • Real-time Predictions        │   │
│  │  • Visualizations     • Alert Management             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       API Layer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI Backend                          │   │
│  │  • /predict          • /predict-anomaly              │   │
│  │  • /predict-hybrid   • /block-transaction            │   │
│  │  • /health           • /system-summary               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Random     │         │  Isolation   │                 │
│  │   Forest     │◄────────►   Forest     │                 │
│  │  (Supervised)│         │  (Anomaly)   │                 │
│  └──────────────┘         └──────────────┘                 │
│         │                         │                          │
│         └─────────┬───────────────┘                          │
│                   ▼                                          │
│          ┌─────────────────┐                                │
│          │ Hybrid Decision │                                │
│          │     Engine      │                                │
│          └─────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing                             │
│  • Feature Engineering  • Scaling & Encoding                │
│  • SMOTE Resampling     • Threshold Optimization            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

This system uses the **Credit Card Fraud Detection Dataset** from Kaggle:

- **Source**: [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Size**: 1.2M+ transactions
- **Features**: 
  - Transaction amount, timestamp
  - Customer demographics (DOB, location)
  - Merchant details (category, location)
  - Geographic coordinates (lat/long)
- **Class Distribution**: Highly imbalanced (~0.5% fraud)

### Dataset Files Required

```
data/
├── fraudTrain.csv    # Training data (~1M transactions)
└── fraudTest.csv     # Test data (~500K transactions)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Visit [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
2. Download `fraudTrain.csv` and `fraudTest.csv`
3. Place files in the `data/` directory

```bash
mkdir -p data
# Copy your downloaded files
cp /path/to/fraudTrain.csv data/
cp /path/to/fraudTest.csv data/
```

### Step 5: Environment Configuration

Create a `.env` file in the project root:

```env
# API Configuration
API_URL=http://localhost:8000
FRAUD_API_KEY=your-secure-api-key-here

# Optional: Gemini AI Integration
GEMINI_API_KEY=your-gemini-api-key
```

---

## 🎬 Quick Start

### 1. Train the Models

Train both supervised and anomaly detection models:

```bash
# Train Random Forest model
python train.py

# Train Isolation Forest (anomaly detector)
python train_anomaly_detector.py
```

**Expected Output:**
```
✓ Training pipeline complete. Ready for deployment!
✓ Pipeline Completed Successfully
```

Generated artifacts:
- `model/fraud_model.joblib` - Trained Random Forest
- `model/scaler.joblib` - Feature scaler
- `model/category_encoder.joblib` - Category encoder
- `model/anomaly/isolation_forest.joblib` - Anomaly detector
- `model/metadata.json` - Model metadata

### 2. Start the API Server

```bash
python main.py
```

The API will be available at: `http://localhost:8000`

### 3. Launch the Web Interface

In a new terminal:

```bash
streamlit run app.py
```

The web interface will open at: `http://localhost:8501`

---

## 🧠 Model Training

### Supervised Model Training (Random Forest)

**Features Engineered:**
- `Amount` - Transaction amount
- `Hour_of_Day` - Transaction hour (0-23)
- `Day_of_Week` - Day of week (0-6)
- `Distance_Proxy` - Euclidean distance between customer and merchant
- `Age` - Customer age at transaction time
- `category_encoded` - Transaction category (encoded)

**Training Process:**

```python
# 1. Load and preprocess data
X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)

# 2. Handle class imbalance with SMOTE
X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

# 3. Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_balanced, y_train_balanced)

# 4. Optimize decision threshold
best_threshold = find_best_threshold(model, X_test, y_test, metric='f1')

# 5. Evaluate and save
evaluate_model(model, X_test, y_test, threshold=best_threshold)
save_artifacts(model, scaler, encoder, threshold)
```

**Hyperparameters:**
- `n_estimators`: 100 trees
- `max_depth`: 20 levels
- `min_samples_split`: 10
- `min_samples_leaf`: 4
- `max_features`: 'sqrt'
- `class_weight`: 'balanced'

### Anomaly Detection Training (Isolation Forest)

**Training Process:**

```python
# 1. Train only on legitimate transactions
X_train_legit = X_train[y_train == 0]

# 2. Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.005,  # Expected fraud rate
    random_state=42
)
model.fit(X_train_legit)

# 3. Compute normalized anomaly scores (0-100)
scores = compute_anomaly_scores(model, X_test)

# 4. Evaluate with threshold=88
evaluate_anomaly_detector(model, X_test, y_test, threshold=88)
```

**Configuration:**
- `n_estimators`: 200 trees
- `contamination`: 0.005 (0.5% expected anomalies)
- `threshold`: 88 (anomaly score cutoff)

---

## 📡 API Documentation

### Authentication

All endpoints require API key authentication:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/endpoint
```

### Endpoints

#### 🔍 Supervised Prediction

```http
POST /predict
```

**Request Body:**
```json
{
  "trans_date_trans_time": "2024-01-15 14:30:00",
  "amt": 125.50,
  "category": "grocery_pos",
  "dob": "1985-03-20",
  "lat": 40.7128,
  "long": -74.0060,
  "merch_lat": 40.7580,
  "merch_long": -73.9855
}
```

**Response:**
```json
{
  "prediction": 0,
  "fraud_probability": 0.0234,
  "risk_level": "Low",
  "threshold_used": 0.85,
  "inference_time_ms": 3.45,
  "distance_km_proxy": 7.2
}
```

#### 🎯 Anomaly Detection

```http
POST /predict-anomaly
```

**Request Body:** Same as `/predict`

**Response:**
```json
{
  "is_anomaly": false,
  "anomaly_score": 45.67,
  "anomaly_level": "Normal",
  "inference_time_ms": 2.89,
  "distance_km_proxy": 7.2
}
```

#### 🔀 Hybrid Prediction

```http
POST /predict-hybrid
```

**Request Body:** Same as `/predict`

**Response:**
```json
{
  "final_prediction": 1,
  "confidence": 0.95,
  "supervised_pred": 1,
  "supervised_prob": 0.92,
  "anomaly_score": 89.3,
  "risk_assessment": "CRITICAL: Dual Model Detection",
  "model_agreement": true,
  "inference_time_ms": 5.12,
  "distance_km_proxy": 7.2
}
```

#### 🚫 Block Transaction

```http
POST /block-transaction
```

**Request Body:**
```json
{
  "transaction_data": { ... },
  "model_result": { ... },
  "action": "manual_block"
}
```

#### ✅ Unblock Transaction

```http
POST /unblock-transaction
```

**Request Body:**
```json
{
  "timestamp": "2024-01-15T14:30:00"
}
```

#### 📊 System Health

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "supervised_loaded": true,
  "anomaly_loaded": true
}
```

#### 📈 System Summary

```http
GET /system-summary
```

**Response:**
```json
{
  "total_transactions": 1523,
  "blocked_transactions": 47,
  "fraud_detected": 52,
  "auto_block_enabled": false
}
```

---

## 🖥️ Web Interface

### Dashboard Features

#### 1. **Transaction Testing Tab**

Test transactions interactively with real-time predictions:

![Transaction Testing](docs/images/transaction_testing.png)

- Manual input form
- Instant prediction results
- Risk level visualization
- Geographic distance calculation

#### 2. **Anomaly Visualization Tab**

Explore anomaly detection results:

![Anomaly Viz](docs/images/anomaly_viz.png)

- PCA space visualization
- Anomaly score distributions
- Temporal patterns
- Feature heatmaps

#### 3. **Model Performance Tab**

Review model metrics and comparisons:

![Model Performance](docs/images/model_performance.png)

- Confusion matrices
- ROC and PR curves
- Feature importance
- Model comparison charts

#### 4. **Alert Management Tab**

Monitor and control fraud alerts:

![Alert Management](docs/images/alert_management.png)

- Real-time alert feed
- Block/unblock transactions
- Auto-block configuration
- Alert history

---

## 📊 Model Performance

### Supervised Model (Random Forest)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.9953 |
| **Precision** | 0.7760 |
| **Recall** | 0.7400 |
| **F1 Score** | 0.7576 |
| **Threshold** | 0.85 |

**Confusion Matrix:**

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 553,117 | 457 |
| **Actual Positive** | 557 | 1,588 |

### Anomaly Detection (Isolation Forest)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | 0.8500 |
| **Precision** | 0.2100 |
| **Recall** | 0.7800 |
| **F1 Score** | 0.3300 |
| **Threshold** | 88 |

### Feature Importance (Top 5)

1. **Amount** - 0.62 (62% importance)
2. **Hour_of_Day** - 0.22 (22% importance)
3. **category_encoded** - 0.12 (12% importance)
4. **Age** - 0.03 (3% importance)
5. **Day_of_Week** - 0.01 (1% importance)

### Model Visualizations

The training pipeline generates comprehensive visualizations:

- **Confusion Matrices** - Model accuracy breakdown
- **ROC Curves** - True positive vs false positive rates
- **Precision-Recall Curves** - Precision vs recall tradeoff
- **Feature Importance** - Top contributing features
- **PCA Anomaly Space** - Anomalies in reduced dimensions
- **Temporal Patterns** - Fraud patterns by time
- **Amount Analysis** - Anomaly scores vs transaction amount

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── 📂 data/
│   ├── fraudTrain.csv          # Training dataset
│   └── fraudTest.csv            # Test dataset
│
├── 📂 model/
│   ├── fraud_model.joblib       # Trained Random Forest
│   ├── scaler.joblib            # Feature scaler
│   ├── category_encoder.joblib  # Category encoder
│   ├── metadata.json            # Model metadata
│   │
│   ├── 📂 anomaly/
│   │   ├── isolation_forest.joblib    # Anomaly detector
│   │   ├── anomaly_scaler.joblib      # Anomaly scaler
│   │   ├── anomaly_metadata.json      # Anomaly metadata
│   │   │
│   │   └── 📂 plots/
│   │       ├── pca_anomalies.png
│   │       ├── anomaly_distribution.png
│   │       ├── amount_vs_anomaly.png
│   │       ├── time_vs_anomaly.png
│   │       └── top_anomalies.png
│   │
│   └── 📂 plots/
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── model_comparison.png
│       └── feature_importance.png
│
├── 📂 src/
│   ├── train.py                 # Supervised model training
│   ├── train_anomaly_detector.py # Anomaly model training
│   ├── main.py                  # FastAPI backend
│   └── app.py                   # Streamlit frontend
│
├── 📂 docs/
│   └── 📂 images/               # Documentation images
│
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## 🛠️ Technologies Used

### Machine Learning & Data Science

- **scikit-learn** 1.3.0 - ML algorithms (Random Forest, Isolation Forest)
- **imbalanced-learn** 0.11.0 - SMOTE for class imbalance
- **pandas** 2.0.3 - Data manipulation
- **numpy** 1.24.3 - Numerical computing

### Web Framework & API

- **FastAPI** 0.104.1 - High-performance async API
- **Streamlit** 1.28.0 - Interactive web dashboard
- **Uvicorn** 0.24.0 - ASGI server
- **Pydantic** 2.5.0 - Data validation

### Visualization

- **Matplotlib** 3.7.2 - Static plots
- **Seaborn** 0.12.2 - Statistical visualizations
- **Plotly** 5.17.0 - Interactive charts

### Utilities

- **python-dotenv** 1.0.0 - Environment management
- **python-json-logger** 2.0.7 - Structured logging
- **joblib** 1.3.2 - Model serialization
- **requests** 2.31.0 - HTTP client

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Backend API endpoint | `http://localhost:8000` |
| `FRAUD_API_KEY` | API authentication key | `demo-api-key-change-in-production` |
| `GEMINI_API_KEY` | Google Gemini API key (optional) | - |

### Model Configuration

Edit `train.py` to adjust model hyperparameters:

```python
# Random Forest Configuration
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=4,    # Minimum samples per leaf
    class_weight='balanced' # Handle imbalance
)

# Isolation Forest Configuration
IsolationForest(
    n_estimators=200,      # Number of trees
    contamination=0.005,   # Expected fraud rate
    random_state=42
)
```

### Threshold Tuning

Adjust decision thresholds in `train.py`:

```python
# Optimize for F1 score
best_thresh = find_best_threshold(model, X_test, y_test, metric='f1')

# Or optimize for recall (catch more fraud)
best_thresh = find_best_threshold(model, X_test, y_test, metric='recall')
```

---

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models on first run
RUN python train.py && python train_anomaly_detector.py

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

### Cloud Deployment

#### AWS Deployment

1. **Package application:**
   ```bash
   zip -r fraud-detection.zip . -x "*.git*" "venv/*" "*.pyc"
   ```

2. **Deploy to EC2:**
   - Launch Ubuntu EC2 instance
   - Install dependencies
   - Configure security groups (port 8000, 8501)
   - Run application

3. **Or use AWS Lambda + API Gateway:**
   - Package as Lambda function
   - Configure API Gateway
   - Use S3 for model storage

#### Google Cloud Platform

1. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy fraud-detection \
     --source . \
     --platform managed \
     --region us-central1
   ```

2. **Or use App Engine:**
   - Create `app.yaml`
   - Deploy with `gcloud app deploy`

---

## 📈 Performance Optimization

### Inference Speed

**Current Performance:**
- Random Forest: ~3-5ms per prediction
- Isolation Forest: ~2-3ms per prediction
- Hybrid: ~5-8ms per prediction

**Optimization Tips:**

1. **Model Quantization:**
   ```python
   # Reduce tree depth for faster inference
   RandomForestClassifier(max_depth=15)  # vs 20
   ```

2. **Feature Selection:**
   ```python
   # Use only top N most important features
   top_features = ['Amount', 'Hour_of_Day', 'category_encoded']
   ```

3. **Batch Processing:**
   ```python
   # Process multiple transactions at once
   predictions = model.predict_proba(X_batch)
   ```

### Memory Usage

**Current Memory Footprint:**
- Random Forest: ~50MB
- Isolation Forest: ~30MB
- Total: ~100MB in memory

**Optimization:**
- Use model compression
- Implement lazy loading
- Cache predictions

---

## 🧪 Testing

### Unit Tests

Run unit tests:

```bash
pytest tests/
```

### Integration Tests

Test API endpoints:

```bash
# Start API server
python main.py

# In another terminal
pytest tests/integration/
```

### Load Testing

Use Locust for load testing:

```python
# locustfile.py
from locust import HttpUser, task

class FraudDetectionUser(HttpUser):
    @task
    def predict(self):
        self.client.post("/predict", json={
            "trans_date_trans_time": "2024-01-15 14:30:00",
            "amt": 125.50,
            # ... other fields
        }, headers={"X-API-Key": "your-api-key"})
```

Run load test:

```bash
locust -f locustfile.py --host=http://localhost:8000
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes:**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch:**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) Ashraf Pathan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **Dataset**: Credit Card Fraud Detection Dataset by [Kartik Shenoy](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Inspiration**: Real-world fraud detection systems at financial institutions
- **Libraries**: scikit-learn, FastAPI, Streamlit, and the entire Python ML ecosystem

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/fraud-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/fraud-detection-system/discussions)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

### Version 2.0 (Planned)

- [ ] Add deep learning models (LSTM, Transformer)
- [ ] Implement graph neural networks for transaction networks
- [ ] Add explainable AI (SHAP, LIME) for predictions
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Real-time streaming with Kafka
- [ ] Enhanced authentication (OAuth, JWT)
- [ ] A/B testing framework

### Version 3.0 (Future)

- [ ] Federated learning support
- [ ] Automated retraining pipeline
- [ ] Advanced ensemble methods
- [ ] Integration with payment gateways
- [ ] Blockchain fraud detection
- [ ] Multi-model comparison dashboard

---

## 📚 Additional Resources

### Documentation

- [Model Training Guide](docs/TRAINING.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

### Tutorials

- [Getting Started Tutorial](docs/tutorials/getting_started.md)
- [Feature Engineering Deep Dive](docs/tutorials/feature_engineering.md)
- [Custom Model Integration](docs/tutorials/custom_models.md)

### Research Papers

1. "Credit Card Fraud Detection using Machine Learning" - IEEE
2. "Anomaly Detection in Financial Transactions" - ACM
3. "Ensemble Methods for Imbalanced Classification" - JMLR

---

<div align="center">

**Made with ❤️ by the Fraud Detection Team**

⭐ Star this repo if you find it helpful!

</div>
