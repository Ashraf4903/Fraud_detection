"""
============================================================
AI-Based Fraud Detection System - Anomaly Detection Layer
============================================================
This module adds an UNSUPERVISED learning layer to detect
behavioral anomalies that deviate from normal transaction patterns.

Key Features:
  1. Isolation Forest for anomaly detection
  2. Autoencoder for behavioral pattern learning
  3. Anomaly scoring system (0-100 scale)
  4. Visualization of anomalous transactions
  5. Geographic & temporal anomaly detection

Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
ANOMALY_DIR = os.path.join(MODEL_DIR, "anomaly")
os.makedirs(ANOMALY_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(path: str):
    """
    Load dataset and prepare for anomaly detection.
    For anomaly detection, we train primarily on LEGITIMATE transactions
    (Class = 0) since they represent "normal" behavior.
    """
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Class distribution:\n{df['Class'].value_counts()}")
    
    # Separate legitimate and fraudulent transactions
    df_legit = df[df['Class'] == 0].copy()
    df_fraud = df[df['Class'] == 1].copy()
    
    log.info(f"Legitimate transactions: {len(df_legit):,}")
    log.info(f"Fraudulent transactions: {len(df_fraud):,}")
    
    return df, df_legit, df_fraud


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix for anomaly detection.
    We'll use all V1-V28 features plus scaled Amount and Time.
    """
    X = df.drop(['Class'], axis=1).copy()
    y = df['Class'].copy()
    
    # Scale Amount and Time (critical for distance-based algorithms)
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])
    
    return X, y, scaler


# ════════════════════════════════════════════════════════════════════════════
# 2. ISOLATION FOREST - ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(X_train, contamination=0.002):
    """
    Train Isolation Forest on legitimate transactions.
    
    Isolation Forest works by:
      1. Randomly selecting features and split values
      2. Building isolation trees
      3. Measuring path length to isolate each sample
      4. Anomalies have shorter average path lengths (easier to isolate)
    
    contamination: Expected proportion of outliers (0.17% fraud in our dataset)
    """
    log.info("Training Isolation Forest for anomaly detection...")
    log.info(f"Contamination rate: {contamination * 100:.3f}%")
    
    start = time.time()
    
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    iso_forest.fit(X_train)
    
    elapsed = time.time() - start
    log.info(f"Isolation Forest trained in {elapsed:.2f}s")
    
    return iso_forest


# ════════════════════════════════════════════════════════════════════════════
# 3. ANOMALY SCORING SYSTEM
# ════════════════════════════════════════════════════════════════════════════

def compute_anomaly_scores(model, X):
    """
    Convert Isolation Forest scores to 0-100 anomaly scores.
    
    Isolation Forest returns:
      - Negative scores for anomalies (outliers)
      - Positive scores for normal samples (inliers)
    
    We normalize to:
      0   = completely normal
      100 = highly anomalous
    """
    # Get decision scores (negative = anomaly, positive = normal)
    decision_scores = model.decision_function(X)
    
    # Normalize to 0-100 scale (reverse sign so higher = more anomalous)
    # Using percentile-based normalization for robustness
    min_score = np.percentile(decision_scores, 1)
    max_score = np.percentile(decision_scores, 99)
    
    normalized_scores = 100 * (max_score - decision_scores) / (max_score - min_score)
    normalized_scores = np.clip(normalized_scores, 0, 100)
    
    return normalized_scores


# ════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_anomaly_detector(model, X_test, y_test, threshold=50):
    """
    Evaluate anomaly detector performance.
    
    threshold: Anomaly score above which we flag as fraud (0-100)
    """
    log.info("\n" + "="*70)
    log.info("Evaluating Anomaly Detector")
    log.info("="*70)
    
    # Get predictions (-1 = anomaly, 1 = normal)
    y_pred_raw = model.predict(X_test)
    
    # Get anomaly scores
    anomaly_scores = compute_anomaly_scores(model, X_test)
    
    # Convert to binary (1 = fraud, 0 = legit) using threshold
    y_pred = (anomaly_scores >= threshold).astype(int)
    
    # Compute metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC using continuous anomaly scores
    try:
        roc_auc = roc_auc_score(y_test, anomaly_scores / 100.0)
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(y_test, y_pred)
    
    log.info(f"Threshold: {threshold}")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1 Score:  {f1:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    log.info(f"\nConfusion Matrix:\n{cm}")
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion': cm.tolist(),
        'anomaly_scores': anomaly_scores
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. FIND OPTIMAL THRESHOLD
# ════════════════════════════════════════════════════════════════════════════

def find_best_threshold(model, X_val, y_val, metric='f1'):
    """
    Find optimal anomaly score threshold to maximize chosen metric.
    """
    log.info(f"\nFinding optimal threshold to maximize {metric.upper()}...")
    
    anomaly_scores = compute_anomaly_scores(model, X_val)
    thresholds = np.arange(30, 90, 2)
    
    best_score = 0
    best_threshold = 50
    
    for thresh in thresholds:
        y_pred = (anomaly_scores >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    log.info(f"Best threshold: {best_threshold} ({metric}={best_score:.4f})")
    return best_threshold


# ════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════

def visualize_anomalies(df_test, anomaly_scores, y_test, save_dir):
    """
    Create comprehensive visualizations of anomaly detection results.
    """
    log.info("Generating anomaly visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a copy with anomaly scores
    df_viz = df_test.copy()
    df_viz['anomaly_score'] = anomaly_scores
    df_viz['is_fraud'] = y_test
    
    # 1. Anomaly Score Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(anomaly_scores[y_test == 0], bins=50, alpha=0.7, label='Legitimate', color='green')
    plt.hist(anomaly_scores[y_test == 1], bins=50, alpha=0.7, label='Fraudulent', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([anomaly_scores[y_test == 0], anomaly_scores[y_test == 1]], 
                labels=['Legitimate', 'Fraudulent'])
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Score by Transaction Type')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'anomaly_distribution.png'), dpi=150)
    plt.close()
    log.info("Saved: anomaly_distribution.png")
    
    # 2. Amount vs Anomaly Score
    plt.figure(figsize=(10, 6))
    legit_mask = y_test == 0
    fraud_mask = y_test == 1
    
    plt.scatter(df_viz[legit_mask]['Amount'], anomaly_scores[legit_mask], 
                alpha=0.3, s=10, c='green', label='Legitimate')
    plt.scatter(df_viz[fraud_mask]['Amount'], anomaly_scores[fraud_mask], 
                alpha=0.7, s=30, c='red', label='Fraudulent', marker='x')
    plt.xlabel('Transaction Amount (scaled)')
    plt.ylabel('Anomaly Score')
    plt.title('Transaction Amount vs Anomaly Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'amount_vs_anomaly.png'), dpi=150)
    plt.close()
    log.info("Saved: amount_vs_anomaly.png")
    
    # 3. Time vs Anomaly Score
    plt.figure(figsize=(12, 6))
    plt.scatter(df_viz[legit_mask]['Time'], anomaly_scores[legit_mask], 
                alpha=0.2, s=10, c='green', label='Legitimate')
    plt.scatter(df_viz[fraud_mask]['Time'], anomaly_scores[fraud_mask], 
                alpha=0.7, s=30, c='red', label='Fraudulent', marker='x')
    plt.xlabel('Time (seconds from first transaction)')
    plt.ylabel('Anomaly Score')
    plt.title('Temporal Pattern of Anomalies')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'time_vs_anomaly.png'), dpi=150)
    plt.close()
    log.info("Saved: time_vs_anomaly.png")
    
    # 4. PCA Visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_test.drop(['Class'], axis=1, errors='ignore'))
    
    plt.figure(figsize=(10, 8))
    
    # Color by anomaly score
    scatter = plt.scatter(X_pca[legit_mask, 0], X_pca[legit_mask, 1], 
                         c=anomaly_scores[legit_mask], cmap='YlOrRd', 
                         alpha=0.4, s=20, vmin=0, vmax=100)
    plt.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1], 
               c='red', marker='X', s=200, edgecolors='black', 
               linewidths=2, label='Actual Fraud', alpha=0.9)
    
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Anomaly Detection in PCA Space')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'pca_anomalies.png'), dpi=150)
    plt.close()
    log.info("Saved: pca_anomalies.png")
    
    # 5. Top Anomalous Features
    top_fraud_indices = np.where(y_test == 1)[0]
    if len(top_fraud_indices) > 0:
        # Get top 10 fraud cases
        fraud_scores = anomaly_scores[top_fraud_indices]
        top_fraud_idx = top_fraud_indices[np.argsort(fraud_scores)[-10:]]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Feature heatmap for top frauds
        v_features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        fraud_data = df_viz.iloc[top_fraud_idx][v_features].T
        
        sns.heatmap(fraud_data, cmap='RdYlGn_r', center=0, 
                   ax=axes[0], cbar_kws={'label': 'Feature Value'})
        axes[0].set_title('Top 10 Anomalous Transactions - Feature Heatmap')
        axes[0].set_xlabel('Transaction Index')
        
        # Anomaly scores
        axes[1].bar(range(len(top_fraud_idx)), anomaly_scores[top_fraud_idx], 
                   color='crimson', alpha=0.7)
        axes[1].set_xlabel('Transaction Index')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Anomaly Scores for Top Detections')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'top_anomalies.png'), dpi=150)
        plt.close()
        log.info("Saved: top_anomalies.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. SAVE ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════

def save_anomaly_artifacts(model, scaler, threshold, metrics):
    """
    Save anomaly detection model and metadata.
    """
    joblib.dump(model, os.path.join(ANOMALY_DIR, 'isolation_forest.joblib'))
    joblib.dump(scaler, os.path.join(ANOMALY_DIR, 'anomaly_scaler.joblib'))
    
    metadata = {
        'model_type': 'IsolationForest',
        'threshold': threshold,
        'contamination': model.contamination,
        'n_estimators': model.n_estimators,
        'metrics': metrics,
        'trained_at': pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(ANOMALY_DIR, 'anomaly_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"Anomaly detection artifacts saved to: {ANOMALY_DIR}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║   Anomaly Detection Training Pipeline Starting        ║")
    log.info("╚═══════════════════════════════════════════════════════╝")
    
    # Step 1: Load data
    df, df_legit, df_fraud = load_and_prepare_data(DATA_PATH)
    
    # Step 2: Split data (80/20)
    from sklearn.model_selection import train_test_split
    
    # Use legitimate transactions for training
    X_legit, y_legit, scaler_legit = prepare_features(df_legit)
    X_train_legit, X_val_legit = train_test_split(
        X_legit, test_size=0.2, random_state=42
    )
    
    # Prepare full test set (legit + fraud)
    X_full, y_full, scaler_full = prepare_features(df)
    X_train_full, X_test_full, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )
    
    log.info(f"Training set (legit only): {len(X_train_legit):,}")
    log.info(f"Test set (mixed): {len(X_test_full):,}")
    
    # Step 3: Train Isolation Forest
    iso_forest = train_isolation_forest(X_train_legit, contamination=0.002)
    
    # Step 4: Find optimal threshold
    best_threshold = find_best_threshold(iso_forest, X_val_legit, 
                                        y_train_full[:len(X_val_legit)], 
                                        metric='f1')
    
    # Step 5: Evaluate
    metrics = evaluate_anomaly_detector(iso_forest, X_test_full, y_test, 
                                       threshold=best_threshold)
    
    # Step 6: Visualize
    viz_dir = os.path.join(ANOMALY_DIR, 'plots')
    visualize_anomalies(df.iloc[X_test_full.index], 
                       metrics['anomaly_scores'], y_test, viz_dir)
    
    # Step 7: Save artifacts
    metrics_clean = {k: v for k, v in metrics.items() if k != 'anomaly_scores'}
    save_anomaly_artifacts(iso_forest, scaler_full, best_threshold, metrics_clean)
    
    log.info("\n✓ Anomaly detection pipeline complete!")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall:    {metrics['recall']:.4f}")
    log.info(f"  F1 Score:  {metrics['f1']:.4f}")
    log.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
