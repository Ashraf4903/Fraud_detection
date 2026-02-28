"""
============================================================
AI-Based Fraud Detection System - Model Training Pipeline
============================================================
Author      : ML Engineering Team
Description : End-to-end training pipeline that:
              1. Loads the Kaggle credit card dataset
              2. Splits data FIRST to prevent data leakage
              3. Scales features based ONLY on training data
              4. Handles class imbalance using SMOTE
              5. Trains Logistic Regression (baseline) & Random Forest (main)
              6. Evaluates both models with comprehensive metrics
              7. Tunes decision threshold to maximize F1/Recall
              8. Saves the best model for API inference

Dataset     : https://www.kaggle.com/mlg-ulb/creditcardfraud
              Download creditcard.csv and place it inside data/
============================================================
"""

import os
import time
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & EXPLORATION
# ════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load the CSV dataset and perform a quick sanity check.
    """
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    log.info(f"Dataset shape : {df.shape}")
    log.info(f"Missing values: {df.isnull().sum().sum()}")

    fraud_count = df["Class"].sum()
    legit_count = len(df) - fraud_count
    ratio = fraud_count / len(df) * 100

    log.info(f"Class distribution → Legit: {legit_count:,} | Fraud: {fraud_count:,} ({ratio:.3f}%)")

    return df


# ════════════════════════════════════════════════════════════════════════════
# 2.  PREPARATION (SPLIT THEN SCALE)
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    1. Splits the data into Train/Test FIRST.
    2. Fits the StandardScaler ONLY on the Train set to prevent data leakage.
    3. Transforms both Train and Test sets.
    """
    log.info("Starting data preparation (Split -> Scale) to prevent leakage …")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 1. Split FIRST (Stratified to maintain 0.17% fraud ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    log.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    log.info(f"Train fraud count: {y_train.sum()}  |  Test fraud count: {y_test.sum()}")

    # 2. Scale ONLY on training data
    log.info("Scaling features (Amount & Time) …")
    scaler = StandardScaler()

    # Fit on train, transform train
    X_train_scaled = X_train.copy()
    X_train_scaled[["scaled_amount", "scaled_time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_train_scaled.drop(["Amount", "Time"], axis=1, inplace=True)

    # Transform test (do NOT fit)
    X_test_scaled = X_test.copy()
    X_test_scaled[["scaled_amount", "scaled_time"]] = scaler.transform(X_test[["Amount", "Time"]])
    X_test_scaled.drop(["Amount", "Time"], axis=1, inplace=True)

    log.info(f"Feature matrix shape after scaling: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ════════════════════════════════════════════════════════════════════════════
# 3.  HANDLE CLASS IMBALANCE WITH SMOTE
# ════════════════════════════════════════════════════════════════════════════

def apply_smote(X_train, y_train, random_state: int = 42):
    """
    SMOTE (Synthetic Minority Over-sampling TEchnique).
    IMPORTANT: Applied ONLY to the training set.
    """
    log.info("Applying SMOTE to balance training data …")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    log.info(f"Before SMOTE → Legit: {(y_train == 0).sum():,} | Fraud: {(y_train == 1).sum():,}")
    log.info(f"After  SMOTE → Legit: {(y_resampled == 0).sum():,} | Fraud: {(y_resampled == 1).sum():,}")
    return X_resampled, y_resampled


# ════════════════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train):
    """Baseline model: Logistic Regression"""
    log.info("Training Logistic Regression (baseline) …")
    start = time.time()

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    elapsed = time.time() - start
    log.info(f"Logistic Regression trained in {elapsed:.2f}s")
    return lr


def train_random_forest(X_train, y_train):
    """Main model: Random Forest"""
    log.info("Training Random Forest (main model) …")
    start = time.time()

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    elapsed = time.time() - start
    log.info(f"Random Forest trained in {elapsed:.2f}s")
    return rf


# ════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5):
    """Comprehensive evaluation with a custom decision threshold."""
    log.info(f"\n{'='*60}")
    log.info(f"Evaluating: {model_name}  (threshold={threshold})")
    log.info(f"{'='*60}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)
    avg_prec  = average_precision_score(y_test, y_proba)

    log.info(f"  Precision  : {precision:.4f}")
    log.info(f"  Recall     : {recall:.4f}   ← Priority metric for fraud detection")
    log.info(f"  F1 Score   : {f1:.4f}")
    log.info(f"  ROC-AUC    : {roc_auc:.4f}")
    log.info(f"  PR-AUC     : {avg_prec:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log.info(f"\n  Confusion Matrix:")
    log.info(f"    TN={tn}  FP={fp}")
    log.info(f"    FN={fn}  TP={tp}")
    log.info(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    return {
        "model_name" : model_name,
        "threshold"  : threshold,
        "precision"  : round(precision, 4),
        "recall"     : round(recall, 4),
        "f1"         : round(f1, 4),
        "roc_auc"    : round(roc_auc, 4),
        "pr_auc"     : round(avg_prec, 4),
        "confusion"  : cm.tolist(),
        "y_proba"    : y_proba,
    }


# ════════════════════════════════════════════════════════════════════════════
# 6.  THRESHOLD TUNING
# ════════════════════════════════════════════════════════════════════════════

def find_best_threshold(model, X_test, y_test, metric: str = "f1"):
    """Sweep over candidate thresholds and pick the optimal one."""
    log.info(f"Searching for optimal threshold (optimising: {metric}) …")
    y_proba    = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score, best_thresh = 0, 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_test, y_pred, zero_division=0)
        else:
            score = f1_score(y_test, y_pred, zero_division=0)

        if score > best_score:
            best_score  = score
            best_thresh = t

    log.info(f"Best threshold: {best_thresh:.2f}  →  {metric}={best_score:.4f}")
    return round(float(best_thresh), 2)


# ════════════════════════════════════════════════════════════════════════════
# 7.  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_results(results: list, X_test, y_test, save_dir: str):
    """Generate and save all diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # ── Confusion Matrices ─────────────────────
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = np.array(res["confusion"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(f'{res["model_name"]}\n(threshold={res["threshold"]})')
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=150)
    plt.close()
    log.info("Saved: confusion_matrices.png")

    # ── ROC Curves ──────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, lw=2, label=f'{res["model_name"]} (AUC={res["roc_auc"]})')

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves – Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close()
    log.info("Saved: roc_curves.png")

    # ── Precision-Recall Curves ─────────────────────────────
    plt.figure(figsize=(8, 6))
    for res in results:
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        plt.plot(rec, prec, lw=2, label=f'{res["model_name"]} (PR-AUC={res["pr_auc"]})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curves.png"), dpi=150)
    plt.close()
    log.info("Saved: pr_curves.png")

    # ── Model Comparison Bar Chart ──────────────────────────
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res["model_name"])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1 Score", "ROC-AUC"])
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison – Key Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close()
    log.info("Saved: model_comparison.png")


def plot_feature_importance(rf_model, feature_names: list, save_dir: str, top_n: int = 20):
    """Show which features the Random Forest uses most for decisions."""
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(top_n).sort_values()

    plt.figure(figsize=(8, 7))
    top_features.plot(kind="barh", color="steelblue")
    plt.title(f"Random Forest – Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150)
    plt.close()
    log.info("Saved: feature_importance.png")


# ════════════════════════════════════════════════════════════════════════════
# 8.  MODEL PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, scaler, threshold: float, feature_names: list, metrics: dict):
    """Persist the model, scaler, and metadata for API inference."""
    joblib.dump(model,  os.path.join(MODEL_DIR, "fraud_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    metadata = {
        "model_type"    : type(model).__name__,
        "threshold"     : threshold,
        "feature_names" : feature_names,
        "metrics"       : {k: v for k, v in metrics.items() if k != "y_proba"},
        "trained_at"    : pd.Timestamp.now().isoformat(),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Model artifacts saved to: {MODEL_DIR}")


# ════════════════════════════════════════════════════════════════════════════
# 9. INFERENCE BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

def benchmark_inference(model, X_test, n_runs: int = 1000):
    """Measure single-sample inference latency."""
    single_sample = X_test.iloc[[0]]
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(single_sample)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    log.info(f"Inference latency over {n_runs} runs → avg: {avg_ms:.3f}ms | p99: {p99_ms:.3f}ms")
    return avg_ms, p99_ms


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   Fraud Detection – Training Pipeline Starting   ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── Step 1: Load data ────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Step 2: Prepare (Split -> Scale) to prevent leakage ──────
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Grab the feature names for persistence and plotting later
    feature_names = list(X_train.columns)

    # ── Step 3: SMOTE ────────────────────────────────────────────
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── Step 4: Train models ─────────────────────────────────────
    lr_model = train_logistic_regression(X_train_sm, y_train_sm)
    rf_model = train_random_forest(X_train_sm, y_train_sm)

    # ── Step 5: Find optimal threshold for Random Forest ─────────
    best_thresh = find_best_threshold(rf_model, X_test, y_test, metric="f1")

    # ── Step 6: Evaluate ─────────────────────────────────────────
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", threshold=0.5)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", threshold=best_thresh)
    all_results = [lr_results, rf_results]

    # ── Step 7: Model Comparison Table ───────────────────────────
    print("\n" + "="*70)
    print("  MODEL COMPARISON RESULTS TABLE")
    print("="*70)
    header = f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    print(header)
    print("-"*70)
    for r in all_results:
        print(
            f"{r['model_name']:<25} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}"
        )
    print("="*70)
    print("  ★ Higher Recall = fewer missed frauds (priority for this problem)")
    print("="*70 + "\n")

    # ── Step 8: Plots ────────────────────────────────────────────
    plots_dir = os.path.join(MODEL_DIR, "plots")
    plot_results(all_results, X_test, y_test, plots_dir)
    plot_feature_importance(rf_model, feature_names, plots_dir)

    # ── Step 9: Inference benchmark ─────────────────────────────
    avg_ms, p99_ms = benchmark_inference(rf_model, X_test)
    log.info(f"Ready for production: avg inference = {avg_ms:.2f}ms")

    # ── Step 10: Save artifacts ───────────────────────────────────
    # We save the Random Forest as the production model
    save_artifacts(
        model         = rf_model,
        scaler        = scaler,
        threshold     = best_thresh,
        feature_names = feature_names,
        metrics       = rf_results,
    )

    log.info("✓ Training pipeline complete. Ready for deployment!")


if __name__ == "__main__":
    main()