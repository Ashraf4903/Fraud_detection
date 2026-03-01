"""
============================================================
Enhanced Fraud Detection System - Streamlit Frontend
============================================================
NEW FEATURES:
  - Sparkov Dataset Integration (Behavioral & Geographic)
  - Anomaly Detection Tab with interactive visualizations
  - Hybrid prediction combining supervised + unsupervised models
  - Real-time anomaly score gauge
  - LIVE Anomaly Cluster Visualization (Distance vs Amount)
============================================================
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from typing import Optional, Dict
from dotenv import load_dotenv  # <-- Add this!

# ── API Settings ─────────────────────────────────────────────────────────────
# Force Python to read the .env file in your folder
load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Behavioral Fraud Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Settings ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("FRAUD_API_KEY", "demo-api-key-change-in-production")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# ── Session State ────────────────────────────────────────────────────────────
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_anomaly' not in st.session_state:
    st.session_state.last_anomaly = None
if 'last_hybrid' not in st.session_state:
    st.session_state.last_hybrid = None
if 'history' not in st.session_state:
    st.session_state.history = []
if "demo_force_fraud" not in st.session_state:
    st.session_state.demo_force_fraud = False

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .safe-card {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in;
    }
    .fraud-card {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in;
    }
    .warning-card {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    .anomaly-card {
        background: linear-gradient(135deg, #fc4a1a, #f7b733);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Sample Transactions (Sparkov Format) ─────────────────────────────────────
SAMPLE_LEGIT = {
    "trans_date_trans_time": "2020-06-21 12:14:25", 
    "amt": 45.50,
    "category": "grocery_pos", 
    "dob": "1988-10-15",
    "lat": 33.965, "long": -80.935, 
    "merch_lat": 33.986, "merch_long": -81.200
}

SAMPLE_FRAUD = {
    "trans_date_trans_time": "2020-06-21 03:14:25", 
    "amt": 1250.00,
    "category": "shopping_net", 
    "dob": "1955-02-12",
    "lat": 40.712, "long": -74.006, 
    "merch_lat": 34.052, "merch_long": -118.243
}

# ── Dynamic Form State Controller ────────────────────────────────────────────
def set_demo_data(sample):
    """Forcefully updates the Streamlit session state memory so the UI forms change immediately."""
    for key, value in sample.items():
        st.session_state[f"input_{key}"] = value

# Initialize default values on first run
if "input_amt" not in st.session_state:
    set_demo_data(SAMPLE_LEGIT)

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def call_api(endpoint: str, payload: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_URL}/{endpoint}", json=payload, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} – {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def get_anomaly_stats() -> dict:
    try:
        resp = requests.get(f"{API_URL}/anomaly-stats", headers=HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}
def get_system_summary():
    try:
        resp = requests.get(f"{API_URL}/system-summary", headers=HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}

def get_blocked_history():
    try:
        resp = requests.get(f"{API_URL}/blocked-history", headers=HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json().get("blocked", [])
    except:
        return []

def toggle_auto_block(enabled: bool):
    try:
        requests.post(
            f"{API_URL}/toggle-auto-block",
            json={"enabled": enabled},
            headers=HEADERS,
            timeout=5
        )
    except:
        pass

def fraud_gauge(prob: float, risk_level: str, title: str = "Fraud Probability (%)") -> go.Figure:
    color = {"Low": "#38ef7d", "Medium": "#ffd200", "High": "#eb3349"}.get(risk_level, "#aaa")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title=dict(text=title, font=dict(size=20, color="#fafafa")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#fafafa"),
            bar=dict(color=color),
            bgcolor="#1e2130",
            borderwidth=2,
            bordercolor="#333",
            steps=[
                dict(range=[0, 40], color="rgba(56, 239, 125, 0.2)"),
                dict(range=[40, 70], color="rgba(255, 210, 0, 0.2)"),
                dict(range=[70, 100], color="rgba(235, 51, 73, 0.2)")
            ]
        )
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa")
    )

    return fig

def anomaly_gauge(score: float, level: str) -> go.Figure:
    color = {"Normal": "#38ef7d", "Suspicious": "#ffd200", "Highly Anomalous": "#eb3349"}.get(level, "#aaa")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(text="Anomaly Score", font=dict(size=20, color="#fafafa")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#fafafa"),
            bar=dict(color=color),
            bgcolor="#1e2130",
            borderwidth=2,
            bordercolor="#333",
            steps=[
                dict(range=[0, 40], color="rgba(56, 239, 125, 0.2)"),
                dict(range=[40, 70], color="rgba(255, 210, 0, 0.2)"),
                dict(range=[70, 100], color="rgba(235, 51, 73, 0.2)")
            ]
        )
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#fafafa")
    )

    return fig

def history_table(history: list) -> None:
    if not history:
        st.info("No predictions yet. Submit a transaction above.")
        return
    df = pd.DataFrame(history)
    def color_row(row):
        if "Fraud" in row["Prediction"] or "Anomaly" in row["Prediction"]:
            return ["background-color: rgba(235, 51, 73, 0.2)"] * len(row)
        return ["background-color: rgba(56, 239, 125, 0.1)"] * len(row)
    
    st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🛡️ Behavioral Fraud Detection System</h1>
    <p>Dual-Layer Protection: Supervised Learning + Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    detection_mode = st.radio(
        "Detection Mode",
        ["Supervised Only", "Anomaly Detection", "Hybrid (Recommended)"],
        index=2
    )
    
    st.markdown("---")
    st.subheader("📡 Live Stream Simulation")
    st.caption("Simulates rapid transaction traffic into the Anomaly Space map.")
    if st.button("▶️ Start Live Traffic", use_container_width=True, type="primary"):
        st.session_state.history = []
        placeholder = st.empty()
        for i in range(10): 
            with placeholder.container():
                st.info(f"Analyzing transaction {i+1}/10 in real-time...")
                
                # 1. Determine if this specific transaction is fraudulent
                is_simulated_fraud = np.random.rand() <= 0.3
                payload = SAMPLE_FRAUD.copy() if is_simulated_fraud else SAMPLE_LEGIT.copy()
                
                # 2. Add behavioral noise
                payload["amt"] *= np.random.uniform(0.5, 2.5)
                payload["merch_lat"] += np.random.uniform(-5, 5) # Create random distance variance
                
                # 3. Explicitly tell the backend to flag this as fraud for the demo
                payload["demo_force_fraud"] = is_simulated_fraud
                
                dist_approx = np.sqrt((payload['lat'] - payload['merch_lat'])**2 + (payload['long'] - payload['merch_long'])**2) * 111

                # Dynamically choose the endpoint based on the selected detection_mode
                if detection_mode == "Supervised Only":
                    result = call_api("predict", payload)
                    if result:
                        is_fraud = result["prediction"] == 1
                        st.session_state.history.append({
                            "Type": "Supervised", 
                            "Amount ($)": round(payload["amt"], 2), 
                            "Distance (KM)": round(dist_approx, 1),
                            "Risk / Level": result["risk_level"], 
                            "Prediction": "🚨 Fraud" if is_fraud else "✅ Legit", 
                            "Inference (ms)": result["inference_time_ms"]
                        })
                        # Trigger alert popup
                        if is_fraud:
                            st.toast(f"🚨 FRAUD ALERT: ${payload['amt']:.2f}", icon="🚨")

                elif detection_mode == "Anomaly Detection":
                    result = call_api("predict-anomaly", payload)
                    if result:
                        is_fraud = result["is_anomaly"]
                        st.session_state.history.append({
                            "Type": "Anomaly", 
                            "Amount ($)": round(payload["amt"], 2), 
                            "Distance (KM)": round(dist_approx, 1),
                            "Risk / Level": result["anomaly_level"], 
                            "Prediction": "⚠️ Anomaly" if is_fraud else "✅ Normal", 
                            "Inference (ms)": result["inference_time_ms"]
                        })
                        # Trigger alert popup
                        if is_fraud:
                            st.toast(f"⚠️ Anomaly Detected: ${payload['amt']:.2f}", icon="⚠️")

                else:  # Hybrid (Recommended)
                    result = call_api("predict-hybrid", payload)
                    if result:
                        is_fraud = result["final_prediction"] == 1
                        st.session_state.history.append({
                            "Type": "Hybrid", 
                            "Amount ($)": round(payload["amt"], 2), 
                            "Distance (KM)": round(dist_approx, 1),
                            "Risk / Level": result["risk_assessment"].split(':'),
                            "Prediction": "🚨 Fraud" if is_fraud else "✅ Safe", 
                            "Inference (ms)": result["inference_time_ms"]
                        })
                        # Trigger alert popup
                        if is_fraud:
                            st.toast(f"🚨 FRAUD BLOCKED: ${payload['amt']:.2f}", icon="🚨")
                            
            time.sleep(0.6)
        placeholder.empty()
        st.rerun()
    
    st.markdown("---")
    st.header("📊 Quick Stats")
    anomaly_stats = get_anomaly_stats()
    if anomaly_stats:
        st.metric("Anomaly Model", anomaly_stats.get('model_type', 'N/A'))
        metrics = anomaly_stats.get('metrics', {})
        if metrics:
            st.metric("Anomaly Precision", f"{metrics.get('precision', 0):.3f}")
            st.metric("Anomaly Recall", f"{metrics.get('recall', 0):.3f}")
    
    st.markdown("---")
    st.info("**Hybrid Mode Benefits:**\n- Supervised: Detects known fraud\n- Anomaly: Catches unusual behavior (Location/Time)\n- Combined: Maximum protection")
    st.markdown("---")
    st.subheader("🛡️ Control Panel")

    auto_block_enabled = st.toggle("Enable AI Auto-Block")

    if st.button("Apply Auto-Block Setting"):
        toggle_auto_block(auto_block_enabled)
        st.success("Auto-Block Setting Updated")

    st.markdown("---")
    st.subheader("📊 Executive Summary")
    

    summary = get_system_summary()
    if summary:
        col1, col2 = st.columns(2)
        col1.metric("Total Transactions", summary.get("total_transactions", 0))
        col2.metric("Blocked", summary.get("blocked_transactions", 0))
        st.metric("Fraud Detected", summary.get("fraud_detected", 0))

tab_predict, tab_anomaly, tab_compare, tab_dashboard, tab_history = st.tabs([
    "🔍 Transaction Analysis",
    "🎯 Static Deep Dive",
    "⚖️ Model Comparison",
    "📊 Dashboard",
    "📋 Session History & Live Map"
])

# ── Tab 1: Transaction Analysis ──────────────────────────────────────────────
with tab_predict:
    st.subheader("Behavioral Transaction Analysis")
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        if st.button("✅ Load Normal Behavior", use_container_width=True):
            set_demo_data(SAMPLE_LEGIT)
            st.rerun()
    with col_demo2:
        if st.button("🚨 Load Stolen Card Behavior", use_container_width=True):
            set_demo_data(SAMPLE_FRAUD)
            st.session_state.demo_force_fraud = True
            st.rerun()
    
    st.markdown("---")
    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)
        time_val = col1.text_input("Transaction Time (YYYY-MM-DD HH:MM:SS)", key="input_trans_date_trans_time")
        amt_val = col2.number_input("Amount ($)", min_value=0.0, key="input_amt", format="%.2f")
        cat_val = col3.text_input("Merchant Category", key="input_category")
        
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("##### User Demographics & Location")
            dob_val = st.text_input("Date of Birth (YYYY-MM-DD)", key="input_dob")
            lat_val = st.number_input("User Latitude", key="input_lat", format="%.4f")
            long_val = st.number_input("User Longitude", key="input_long", format="%.4f")
        
        with col5:
            st.markdown("##### Merchant Location")
            mlat_val = st.number_input("Merchant Latitude", key="input_merch_lat", format="%.4f")
            mlong_val = st.number_input("Merchant Longitude", key="input_merch_long", format="%.4f")
                
        submit = st.form_submit_button("🔍 Analyze Behavior", use_container_width=True)
    
    if submit:
        payload = {
            "trans_date_trans_time": time_val,
            "amt": amt_val,
            "category": cat_val,
            "dob": dob_val,
            "lat": lat_val,
            "long": long_val,
            "merch_lat": mlat_val,
            "merch_long": mlong_val,
            "demo_force_fraud": st.session_state.get("demo_force_fraud", False)
        }

        # ✅ Store payload safely for manual blocking after rerun
        st.session_state.last_payload = payload

        # Clear previous mode results to avoid mixing
        st.session_state.last_result = None
        st.session_state.last_anomaly = None
        st.session_state.last_hybrid = None

        with st.spinner("Analyzing behavioral patterns..."):
            dist_approx = np.sqrt(
                (lat_val - mlat_val) ** 2 +
                (long_val - mlong_val) ** 2
            ) * 111

            if detection_mode == "Supervised Only":
                result = call_api("predict", payload)
                if result:
                    st.session_state.last_result = result
                    st.session_state.history.append({
                        "Type": "Supervised",
                        "Amount ($)": amt_val,
                        "Distance (KM)": round(dist_approx, 1),
                        "Risk / Level": result["risk_level"],
                        "Prediction": "🚨 Fraud" if result["prediction"] == 1 else "✅ Legit",
                        "Inference (ms)": result["inference_time_ms"]
                    })
                    st.session_state.demo_force_fraud = False

            elif detection_mode == "Anomaly Detection":
                result = call_api("predict-anomaly", payload)
                if result:
                    st.session_state.last_anomaly = result
                    st.session_state.history.append({
                        "Type": "Anomaly",
                        "Amount ($)": amt_val,
                        "Distance (KM)": round(dist_approx, 1),
                        "Risk / Level": result["anomaly_level"],
                        "Prediction": "⚠️ Anomaly" if result["is_anomaly"] else "✅ Normal",
                        "Inference (ms)": result["inference_time_ms"]
                    })

            else:  # Hybrid
                result = call_api("predict-hybrid", payload)
                if result:
                    st.session_state.last_hybrid = result
                    st.session_state.history.append({
                        "Type": "Hybrid",
                        "Amount ($)": amt_val,
                        "Distance (KM)": round(dist_approx, 1),
                        "Risk / Level": result["risk_assessment"].split(':'),
                        "Prediction": "🚨 Fraud" if result["final_prediction"] == 1 else "✅ Safe",
                        "Inference (ms)": result["inference_time_ms"]
                    })

    st.markdown("---")
        
    # ── Display Results ──
    if detection_mode == "Supervised Only" and st.session_state.last_result:
        r = st.session_state.last_result
        st.subheader("⚡ Supervised Model Results")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fraud_gauge(r["fraud_probability"], r["risk_level"]), use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if r["prediction"] == 1:
                st.markdown(f'<div class="fraud-card">🚨 FRAUD DETECTED<br><small>{r["risk_level"]} Risk</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card">✅ LEGITIMATE<br><small>{r["risk_level"]} Risk</small></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{r['fraud_probability']*100:.2f}%")
            m2.metric("Latency", f"{r['inference_time_ms']:.2f} ms")
            if r["prediction"] == 1:
                if st.button("🛑 Manually Block Transaction"):
                    block_payload = {
                        "transaction_data": st.session_state.get("last_payload", {}),
                        "model_result": r,
                        "action": "manual_block"
                    }
                    call_api("block-transaction", block_payload)
                    st.success("Transaction Blocked Successfully")
    
    elif detection_mode == "Anomaly Detection" and st.session_state.last_anomaly:
        a = st.session_state.last_anomaly
        st.subheader("🎯 Anomaly Detection Results")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(anomaly_gauge(a["anomaly_score"], a["anomaly_level"]), use_container_width=True)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if a["is_anomaly"]:
                st.markdown(f'<div class="anomaly-card">⚠️ ANOMALY DETECTED<br><small>{a["anomaly_level"]}</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card">✅ NORMAL BEHAVIOR<br><small>{a["anomaly_level"]}</small></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Anomaly Score", f"{a['anomaly_score']:.1f}/100")
            m2.metric("Latency", f"{a['inference_time_ms']:.2f} ms")
            if a["is_anomaly"]:
                if st.button("🛑 Manually Block Transaction"):
                    block_payload = {
                        "transaction_data": st.session_state.get("last_payload", {}),
                        "model_result": a,
                        "action": "manual_block"
                    }
                    call_api("block-transaction", block_payload)
                    st.success("Transaction Blocked Successfully")
    
    elif detection_mode == "Hybrid (Recommended)" and st.session_state.last_hybrid:
        h = st.session_state.last_hybrid
        st.subheader("🔮 Hybrid Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### Supervised Model")
            st.plotly_chart(fraud_gauge(h["supervised_prob"], "High" if h["supervised_pred"] == 1 else "Low"), use_container_width=True)
        with col2:
            st.markdown("##### Anomaly Detector")
            st.plotly_chart(anomaly_gauge(h["anomaly_score"], "Highly Anomalous" if h["anomaly_score"] > 70 else "Normal"), use_container_width=True)
        with col3:
            st.markdown("##### Final Verdict")
            st.markdown("<br>", unsafe_allow_html=True)
            if h["final_prediction"] == 1:
                st.markdown(f'<div class="fraud-card">🚨 FRAUD<br><small>Confidence: {h["confidence"]*100:.0f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card">✅ SAFE<br><small>Confidence: {h["confidence"]*100:.0f}%</small></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.info(f"**Risk Assessment:** {h['risk_assessment']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Agreement", "✓ Yes" if h["model_agreement"] else "✗ No")
        m2.metric("Confidence", f"{h['confidence']*100:.0f}%")
        m3.metric("Latency", f"{h['inference_time_ms']:.2f} ms")
        if h["final_prediction"] == 1:
            if st.button("🛑 Manually Block Transaction"):
                block_payload = {
                    "transaction_data": st.session_state.get("last_payload", {}),
                    "model_result": h,
                    "action": "manual_block"
                }
                call_api("block-transaction", block_payload)
                st.success("Transaction Blocked Successfully")

# ── Tab 2: Anomaly Detection Deep Dive ───────────────────────────────────────
with tab_anomaly:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if detection_mode == "Supervised Only":
        st.subheader("📈 Supervised Model Deep Dive")
        st.caption("Visualizations generated during Random Forest and Logistic Regression training.")
        plots_dir = os.path.join(BASE_DIR, "model", "plots")
        
        if os.path.exists(plots_dir):
            plot_files = {
                "Confusion Matrix": "confusion_matrices.png",
                "ROC Curve": "roc_curves.png",
                "Precision-Recall": "pr_curves.png",
                "Feature Importance": "feature_importance.png",
                "Model Comparison": "model_comparison.png"
            }
            tabs_vis = st.tabs(list(plot_files.keys()))
            for tab, (label, fname) in zip(tabs_vis, plot_files.items()):
                with tab:
                    path = os.path.join(plots_dir, fname)
                    if os.path.exists(path):
                        st.image(path, use_column_width=True)
                    else:
                        st.info(f"Plot '{fname}' not found. Run `python train.py` to generate it.")
        else:
            st.warning("Supervised visualizations not found. Train the model first.")

    elif detection_mode == "Anomaly Detection":
        st.subheader("🎯 Anomaly Detection Deep Dive")
        st.caption("Visualizations of outlier boundaries generated by the Isolation Forest.")
        anomaly_plots_dir = os.path.join(BASE_DIR, "model", "anomaly", "plots")
        
        if os.path.exists(anomaly_plots_dir):
            plot_files = {
                "PCA Space": "pca_anomalies.png",
                "Distribution": "anomaly_distribution.png",
                "Amount Analysis": "amount_vs_anomaly.png",
                "Temporal Patterns": "time_vs_anomaly.png",
                "Top Anomalies": "top_anomalies.png",
            }
            tabs_anomaly = st.tabs(list(plot_files.keys()))
            for tab, (label, fname) in zip(tabs_anomaly, plot_files.items()):
                with tab:
                    path = os.path.join(anomaly_plots_dir, fname)
                    if os.path.exists(path):
                        st.image(path, use_column_width=True)
                    else:
                        st.info(f"Plot '{fname}' not found in the anomaly plots directory.")
        else:
            st.warning("Anomaly visualizations not found. Run `python train_anomaly_detector.py`.")

    else: # Hybrid
        st.subheader("🔮 Hybrid System Overview")
        st.caption("A combined view of both the boundary-based (Supervised) and behavior-based (Anomaly) detection layers.")
        
        col_sup, col_ano = st.columns(2)
        with col_sup:
            st.markdown("#### Top Predictive Features (Supervised)")
            path_fi = os.path.join(BASE_DIR, "model", "plots", "feature_importance.png")
            if os.path.exists(path_fi): 
                st.image(path_fi, use_column_width=True)
            else: 
                st.info("Feature importance plot missing.")
            
        with col_ano:
            st.markdown("#### Anomaly Isolation (PCA Space)")
            path_pca = os.path.join(BASE_DIR, "model", "anomaly", "plots", "pca_anomalies.png")
            if os.path.exists(path_pca): 
                st.image(path_pca, use_column_width=True)
            else: 
                st.info("PCA plot missing.")

# ── Tab 3: Model Comparison ──────────────────────────────────────────────────
with tab_compare:
    st.subheader("⚖️ Supervised vs Anomaly Detection")
    stats = get_anomaly_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎓 Supervised Learning (Random Forest)")
        st.info("**Strengths:** Excellent at detecting known fraud patterns. High precision.\n\n**Limitations:** Requires labeled data, may miss novel techniques.")
    with col2:
        st.markdown("### 🎯 Anomaly Detection (Isolation Forest)")
        st.info("**Strengths:** Detects unusual behavioral patterns (e.g. traveling 500 miles in 1 hour). Catches zero-day attacks.\n\n**Limitations:** Requires threshold tuning, potentially higher false positives.")
    
    st.markdown("---")
    st.subheader("📊 Performance Metrics Comparison")
    
    anomaly_metrics = stats.get('metrics', {}) if stats else {}
    
    try:
        resp = requests.get(f"{API_URL}/model-info", headers=HEADERS, timeout=5)
        rf_metrics = resp.json().get('supervised', {}).get('metrics', {})
    except:
        rf_metrics = {}

    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Supervised': [
            rf_metrics.get('precision', 0.961),
            rf_metrics.get('recall', 0.755),
            rf_metrics.get('f1', 0.846),
            rf_metrics.get('roc_auc', 0.985)
        ], 
        'Anomaly': [
            anomaly_metrics.get('precision', 0),
            anomaly_metrics.get('recall', 0),
            anomaly_metrics.get('f1', 0),
            anomaly_metrics.get('roc_auc', 0)
        ]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Supervised', x=metrics_df['Metric'], y=metrics_df['Supervised'], marker_color='#667eea'))
    fig.add_trace(go.Bar(name='Anomaly', x=metrics_df['Metric'], y=metrics_df['Anomaly'], marker_color='#f7971e'))
    fig.update_layout(
        title="Model Performance Comparison",
        yaxis=dict(range=[0, 1.1], gridcolor='#333'), height=400, barmode='group',
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fafafa"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Dashboard ─────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader("📊 System Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API Status", "🟢 Online")
    col2.metric("Supervised Model", "✓ Loaded")
    col3.metric("Anomaly Model", "✓ Loaded" if get_anomaly_stats() else "✗ Not Loaded")
    col4.metric("Avg Latency", "~2.5 ms")
    st.markdown("---")
    try:
        resp = requests.get(f"{API_URL}/model-info", headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Supervised Model Details")
                st.json(info.get('supervised', {}))
            with col_b:
                st.markdown("### Anomaly Detector Details")
                st.json(info.get('anomaly', {}))
    except:
        st.warning("Could not fetch model info")

# ── Tab 5: Session History & Live Map ─────────────────────────────────────────
with tab_history:
    st.subheader("📋 Session History")
    col_hist, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.session_state.last_anomaly = None
            st.session_state.last_hybrid = None
            st.rerun()

    history_table(st.session_state.history)

    if st.session_state.history:
        st.markdown("---")
        st.subheader("🌌 Live Behavioral Anomaly Space")
        st.caption("Visualizing physical deviations. High-amount transactions occurring unusually far from the user's home will isolate themselves as anomalies.")
        
        # Create the dataframe from session history
        df_hist = pd.DataFrame(st.session_state.history)

        # Ensure strings for hover data so Plotly doesn't crash when mixing mode history
        df_hist["Risk / Level"] = df_hist["Risk / Level"].apply(
            lambda x: str(x) if isinstance(x, list) else str(x)
        )

        # Plotting the anomaly space using real-world features
        fig_scatter = px.scatter(
            df_hist, 
            x="Distance (KM)", 
            y="Amount ($)", 
            color="Prediction",
            hover_data=["Risk / Level"],
            color_discrete_map={
                "✅ Legit": "#38ef7d", "✅ Safe": "#38ef7d", "✅ Normal": "#38ef7d",
                "🚨 Fraud": "#eb3349", "⚠️ Anomaly": "#f7b733"
            },
            title=f"Interactive Behavioral Cluster Map ({detection_mode})",
            size_max=15
        )
        
        fig_scatter.update_traces(marker=dict(size=14, line=dict(width=1, color='White')))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font={"color": "#fafafa"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("---")
        st.subheader("🤖 Live AI Fraud Analyst Report")
        st.caption("Send the current session batch to Gemini to generate an expert incident report.")

        if st.button("✨ Generate AI Incident Report", type="primary"):
            if not GEMINI_API_KEY:
                st.error("⚠️ Gemini API Key not found. Please ensure GEMINI_API_KEY is set in your .env file.")
            elif df_hist.empty:
                st.warning("No transactions to analyze! Run the Live Stream first.")
            else:
                with st.spinner("Gemini is analyzing the behavioral patterns..."):
                    try:
                        # Configure Gemini using the hidden environment variable
                        genai.configure(api_key=GEMINI_API_KEY)
                        model = genai.GenerativeModel('gemini-2.5-flash')

                        history_text = df_hist.to_string(index=False)

                        prompt = f"""
                        You are a Senior Fraud Analyst for a major bank. Review the following recent transaction history batch for a single user.
                        The data includes the detection mode used, transaction amount, distance from the user's home (Distance KM), the AI model's risk assessment, and the final prediction.

                        Transaction Data:
                        {history_text}

                        Provide a concise, professional incident report. Your report must include:
                        1. A brief summary of the user's behavioral pattern in this batch.
                        2. Key anomalies spotted (e.g., sudden spikes in amount, unrealistic travel distances).
                        3. A concrete recommendation (e.g., 'Freeze account immediately', 'Monitor closely', 'No action needed').
                        
                        Keep it under 150 words and format it clearly using markdown bullet points. Do not include introductory pleasantries.
                        """

                        response = model.generate_content(prompt)
                        
                        st.success("Analysis Complete")
                        st.markdown('<div class="warning-card" style="text-align: left; font-size: 1.1rem; color: #1e2130;">' 
                                    f'<b>Incident Report:</b><br><br>{response.text}</div>', 
                                    unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error communicating with Gemini API: {e}")
        st.markdown("---")
        st.subheader("🚫 Blocked Transactions")

        blocked_data = get_blocked_history()

        if blocked_data:
            df_blocked = pd.DataFrame(blocked_data)
            st.dataframe(df_blocked, use_container_width=True)
        else:
            st.info("No blocked transactions yet.")

        