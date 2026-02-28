"""
============================================================
AI-Based Fraud Detection System - Anomaly Detection Layer
============================================================
"""

import os
import json
import time
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(BASE_DIR, "data", "fraudTrain.csv")
TEST_PATH   = os.path.join(BASE_DIR, "data", "fraudTest.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
ANOMALY_DIR = os.path.join(MODEL_DIR, "anomaly")
os.makedirs(ANOMALY_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(train_path: str, test_path: str):
    log.info(f"Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    return df_train, df_test


def engineer_features(df: pd.DataFrame, is_training=True, label_encoders=None):
    df = df.copy()

    df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'])
    df['Hour_of_Day'] = df['trans_date'].dt.hour
    df['Day_of_Week'] = df['trans_date'].dt.dayofweek

    df['Distance_Proxy'] = np.sqrt(
        (df['lat'] - df['merch_lat'])**2 +
        (df['long'] - df['merch_long'])**2
    )

    df['dob_date'] = pd.to_datetime(df['dob'])
    df['Age'] = (df['trans_date'] - df['dob_date']).dt.days // 365

    if is_training:
        label_encoders = {}
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        label_encoders['category'] = le
    else:
        le = label_encoders['category']
        df['category_encoded'] = df['category'].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    df['Amount'] = df['amt']

    features = [
        'Amount',
        'Hour_of_Day',
        'Day_of_Week',
        'Distance_Proxy',
        'Age',
        'category_encoded'
    ]

    return df[features], df['is_fraud'], label_encoders, features


# ════════════════════════════════════════════════════════════════════════════
# 2. ISOLATION FOREST
# ════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(X_train, contamination=0.005):
    log.info("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    return model


# ════════════════════════════════════════════════════════════════════════════
# 3. SCORING
# ════════════════════════════════════════════════════════════════════════════

def compute_anomaly_scores(model, X, min_score=None, max_score=None):
    scores = model.decision_function(X)

    if min_score is None or max_score is None:
        min_score = np.percentile(scores, 1)
        max_score = np.percentile(scores, 99)

    normalized = 100 * (max_score - scores) / (max_score - min_score + 1e-6)
    normalized = np.clip(normalized, 0, 100)

    return normalized, min_score, max_score


# ════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_anomaly_detector(model, X_test, y_test, threshold, min_score, max_score):
    anomaly_scores, _, _ = compute_anomaly_scores(model, X_test, min_score, max_score)
    y_pred = (anomaly_scores >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, anomaly_scores / 100)

    cm = confusion_matrix(y_test, y_pred)

    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1 Score:  {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    log.info(f"Confusion Matrix:\n{cm.tolist()}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }, anomaly_scores


# ════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION (FIXED)
# ════════════════════════════════════════════════════════════════════════════

def visualize_anomalies(X_test_scaled, anomaly_scores, y_test, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    legit_mask = y_test == 0
    fraud_mask = y_test == 1

    # PCA FIXED
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test_scaled)

    plt.figure(figsize=(10, 8))

    plt.scatter(
        X_pca[legit_mask, 0],
        X_pca[legit_mask, 1],
        c=anomaly_scores[legit_mask],
        cmap='YlOrRd',
        alpha=0.4,
        s=20
    )

    plt.scatter(
        X_pca[fraud_mask, 0],
        X_pca[fraud_mask, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        label='Actual Fraud'
    )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Anomaly Detection in PCA Space')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'pca_anomalies.png'), dpi=150)
    plt.close()

    log.info("Saved: pca_anomalies.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():

    df_train, df_test = load_and_prepare_data(TRAIN_PATH, TEST_PATH)

    X_train_full, y_train_full, encoder, features = engineer_features(df_train)
    X_test_full, y_test_full, _, _ = engineer_features(
        df_test,
        is_training=False,
        label_encoders=encoder
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    X_train_legit = X_train_scaled[y_train_full == 0]

    model = train_isolation_forest(X_train_legit)

    _, min_score, max_score = compute_anomaly_scores(model, X_train_legit)

    threshold = 88

    metrics, anomaly_scores = evaluate_anomaly_detector(
        model,
        X_test_scaled,
        y_test_full,
        threshold,
        min_score,
        max_score
    )

    visualize_anomalies(
        X_test_scaled,
        anomaly_scores,
        y_test_full,
        os.path.join(ANOMALY_DIR, "plots")
    )

    log.info("✓ Pipeline Completed Successfully")


if __name__ == "__main__":
    main()