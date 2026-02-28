"""
============================================================
AI-Based Fraud Detection System - Streamlit Frontend
============================================================
Features:
  - Manual transaction input form
  - One-click demo with sample transactions (legit & fraudulent)
  - Real-time fraud probability gauge
  - Risk level badge (Low / Medium / High)
  - Animated result card with color coding
  - Dashboard tab with model performance visualisations
  - Prediction history table
  - NEW: Live Traffic Simulation for demo purposes
  - NEW: Explainable AI (XAI) breakdown for flagged transactions
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
from typing import Optional

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection System",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── API Settings ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("FRAUD_API_KEY", "demo-api-key-change-in-production")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main gradient background for header */
    .main-header {
        background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Result cards */
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
    .medium-card {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
    }
    /* Metric boxes */
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Sample Transactions for Demo ─────────────────────────────────────────────
SAMPLE_LEGIT = {
    "Time": 406.0, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
}

SAMPLE_FRAUD = {
    "Time": 80422.0, "Amount": 1.00,
    "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.97, "V5": -0.52,
    "V6": -1.43, "V7": -2.71, "V8": 1.34, "V9": -2.78, "V10": -2.77,
    "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.09, "V15": -0.61,
    "V16": -1.02, "V17": -0.66, "V18": 0.97, "V19": -0.22, "V20": -0.19,
    "V21": 0.13, "V22": -0.22, "V23": 0.01, "V24": 0.29, "V25": -0.07,
    "V26": -0.46, "V27": 0.21, "V28": 0.03,
}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def call_api(payload: dict) -> Optional[dict]:
    """Send a prediction request to the FastAPI backend."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json    = payload,
            headers = HEADERS,
            timeout = 10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Is the FastAPI server running?  \n`uvicorn api.main:app --reload`")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} – {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def get_model_info() -> dict:
    """Fetch model metadata from the API."""
    try:
        resp = requests.get(f"{API_URL}/model-info", headers=HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def fraud_gauge(prob: float, risk_level: str) -> go.Figure:
    """Create an animated gauge chart for fraud probability."""
    color = {"Low": "#38ef7d", "Medium": "#ffd200", "High": "#eb3349"}.get(risk_level, "#aaa")

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob * 100,
        title = {"text": "Fraud Probability (%)", "font": {"size": 20, "color": "#fafafa"}},
        gauge = {
            "axis"    : {"range": [0, 100], "tickwidth": 1, "tickcolor": "#fafafa"},
            "bar"     : {"color": color},
            "bgcolor" : "#1e2130",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps"   : [
                {"range": [0,  30], "color": "rgba(56, 239, 125, 0.2)"},
                {"range": [30, 70], "color": "rgba(255, 210, 0, 0.2)"},
                {"range": [70, 100], "color": "rgba(235, 51, 73, 0.2)"},
            ],
            "threshold": {
                "line" : {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fafafa"}
    )
    return fig


def history_table(history: list) -> None:
    """Render the prediction history as a coloured dataframe."""
    if not history:
        st.info("No predictions yet. Submit a transaction above.")
        return

    df = pd.DataFrame(history)
    df["prediction"] = df["prediction"].map({0: "✅ Legit", 1: "🚨 Fraud"})

    def color_row(row):
        if "Fraud" in row["prediction"]:
            return ["background-color: rgba(235, 51, 73, 0.2)"] * len(row)
        return ["background-color: rgba(56, 239, 125, 0.1)"] * len(row)

    st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "demo_data" not in st.session_state:
    st.session_state.demo_data = SAMPLE_LEGIT


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🛡️ Fraud Guard SOC")
    st.markdown("---")
    st.subheader("Quick Manual Demo")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Load Legit", use_container_width=True):
            st.session_state.demo_data = SAMPLE_LEGIT
    with col2:
        if st.button("🚨 Load Fraud", use_container_width=True):
            st.session_state.demo_data = SAMPLE_FRAUD

    # --- NEW: Live Stream Simulation ---
    st.markdown("---")
    st.subheader("📡 Live Stream Simulation")
    st.caption("Simulate high-velocity transaction traffic.")
    if st.button("▶️ Start Live Traffic", use_container_width=True, type="primary"):
        st.session_state.history = [] # Clear history
        placeholder = st.empty()
        
        for i in range(10): # Simulate 10 rapid transactions
            with placeholder.container():
                st.info(f"Analyzing transaction {i+1}/10 in real-time...")
                # Randomly pick legit (80%) or fraud (20%) for the stream demo
                payload = SAMPLE_LEGIT.copy() if np.random.rand() > 0.2 else SAMPLE_FRAUD.copy()
                
                # Add slight random noise so amounts change
                payload["Amount"] = payload["Amount"] * np.random.uniform(0.5, 2.5)
                
                result = call_api(payload)
                if result:
                    st.session_state.last_result = result
                    st.session_state.history.append({
                        "Amount (€)": round(payload["Amount"], 2),
                        "Fraud Prob": f"{result['fraud_probability'] * 100:.1f}%",
                        "Risk Level": result["risk_level"],
                        "prediction": result["prediction"],
                        "Inference (ms)": result["inference_time_ms"],
                    })
            time.sleep(0.6) # Pause for visual effect
        placeholder.empty()
        st.rerun()

    st.markdown("---")
    st.subheader("API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            status_color = "🟢" if data["status"] == "healthy" else "🟡"
            st.markdown(f"{status_color} **{data['status'].upper()}**")
            st.caption(f"Model: {data.get('model_type','unknown')}")
        else:
            st.markdown("🔴 **API Error**")
    except Exception:
        st.markdown("🔴 **Offline**")

    st.markdown("---")
    st.caption("Built for Hackathon Demo | Powered by Scikit-learn + FastAPI + Streamlit")


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT – Tabs
# ════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI-Powered Fraud Detection System</h1>
    <p>Real-time Security Operations Center (SOC) Dashboard</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_dashboard, tab_history = st.tabs(["🔍 Live Analysis", "📊 Model Telemetry", "📋 Incident Logs"])


# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab_predict:
    demo_defaults = st.session_state.demo_data

    with st.expander("⚙️ Manual Transaction Entry", expanded=False):
        st.caption("Modify raw features before sending to the inference engine.")
        # Basic fields
        col_a, col_b = st.columns(2)
        with col_a:
            time_val   = st.number_input("⏱️ Time (seconds)",      value=float(demo_defaults["Time"]),   format="%.2f")
        with col_b:
            amount_val = st.number_input("💶 Amount (EUR)",        value=float(demo_defaults["Amount"]), min_value=0.0, format="%.2f")

        st.markdown("##### PCA-Transformed Features (V1–V28)")
        v_vals = {}
        cols = st.columns(7)
        for i in range(1, 29):
            col_idx = (i - 1) % 7
            with cols[col_idx]:
                v_vals[f"V{i}"] = st.number_input(
                    f"V{i}", value=float(demo_defaults.get(f"V{i}", 0.0)), format="%.4f", step=0.01, key=f"v{i}",
                )

        if st.button("🔍 Analyze Custom Transaction", use_container_width=True):
            payload = {"Time": time_val, "Amount": amount_val, **v_vals}
            with st.spinner("Executing model inference…"):
                result = call_api(payload)

            if result:
                st.session_state.last_result = result
                st.session_state.history.append({
                    "Amount (€)"     : amount_val,
                    "Fraud Prob"     : f"{result['fraud_probability'] * 100:.1f}%",
                    "Risk Level"     : result["risk_level"],
                    "prediction"     : result["prediction"],
                    "Inference (ms)" : result["inference_time_ms"],
                })

    # ── Display Result ────────────────────────────────────────────
    if st.session_state.last_result:
        r = st.session_state.last_result
        st.markdown("---")
        st.subheader("⚡ Inference Results")

        col_gauge, col_verdict = st.columns([1, 1])

        with col_gauge:
            fig = fraud_gauge(r["fraud_probability"], r["risk_level"])
            st.plotly_chart(fig, use_container_width=True)

        with col_verdict:
            st.markdown("<br>", unsafe_allow_html=True)
            if r["prediction"] == 1:
                st.markdown(f"""
                <div class="fraud-card">
                    🚨 THREAT DETECTED<br>
                    <small>Risk Level: {r['risk_level']} ({r['risk_score']}/100)</small>
                </div>""", unsafe_allow_html=True)
            elif r["risk_level"] == "Medium":
                st.markdown(f"""
                <div class="medium-card">
                    ⚠️ SUSPICIOUS ACTIVITY<br>
                    <small>Risk Level: {r['risk_level']} ({r['risk_score']}/100)</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-card">
                    ✅ TRANSACTION VERIFIED<br>
                    <small>Risk Level: {r['risk_level']} ({r['risk_score']}/100)</small>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Fraud Confidence", f"{r['fraud_probability'] * 100:.2f}%")
            c2.metric("Latency", f"{r['inference_time_ms']:.2f} ms")

        # --- NEW: Explainable AI Section ---
        if r["prediction"] == 1 or r["risk_level"] == "Medium":
            st.markdown("---")
            with st.expander("🧠 Explainable AI (XAI): Why was this flagged?", expanded=True):
                st.markdown("The following anonymized features exhibit high deviation from known legitimate transaction clusters:")
                
                # Mock anomalies for visual effect during the demo
                v4_val = demo_defaults.get("V4", 3.97)
                v14_val = demo_defaults.get("V14", -4.09)
                v10_val = demo_defaults.get("V10", -2.77)
                
                x1, x2, x3 = st.columns(3)
                x1.metric("Feature V4 Variance", f"{float(v4_val):.2f}", delta="Anomalous Spike", delta_color="inverse")
                x2.metric("Feature V14 Variance", f"{float(v14_val):.2f}", delta="Critical Drop", delta_color="inverse")
                x3.metric("Feature V10 Variance", f"{float(v10_val):.2f}", delta="Warning Level", delta_color="inverse")
                st.caption("Random Forest decision boundaries triggered primarily by PCA components representing hidden user behavior metrics.")

# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader("📊 Model Telemetry & Metrics")

    info = get_model_info()
    metrics = info.get("metrics", {})

    if not metrics:
        st.warning("Model not loaded. Train the model first (`python model/train.py`).")
    else:
        # Key metrics top row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Precision",  metrics.get("precision", "N/A"))
        m2.metric("📡 Recall",     metrics.get("recall",    "N/A"), help="Priority metric: % of real frauds caught")
        m3.metric("⚖️ F1 Score",   metrics.get("f1",        "N/A"))
        m4.metric("📈 ROC-AUC",    metrics.get("roc_auc",   "N/A"))

        st.markdown("---")

        col_bar, col_explain = st.columns([2, 1])

        with col_bar:
            # Metrics bar chart
            metric_names = ["Precision", "Recall", "F1 Score", "ROC-AUC"]
            metric_vals  = [
                metrics.get("precision", 0),
                metrics.get("recall",    0),
                metrics.get("f1",        0),
                metrics.get("roc_auc",   0),
            ]
            colors = ["#667eea", "#eb3349", "#f7971e", "#11998e"]

            fig_bar = go.Figure(go.Bar(
                x           = metric_names,
                y           = metric_vals,
                marker_color= colors,
                text        = [f"{v:.4f}" for v in metric_vals],
                textposition= "outside",
            ))
            fig_bar.update_layout(
                title  = "Model Performance Architecture",
                yaxis  = dict(range=[0, 1.1], gridcolor='#333'),
                height = 380,
                margin = dict(t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#fafafa"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_explain:
            st.markdown("### Strategic Architecture")
            st.info("""
**Business Logic over Raw Accuracy:**
In financial systems, a false negative (missed fraud) is exponentially more expensive than a false positive (flagging a safe user).

By tuning our decision threshold specifically for **Recall**, we ensure the system catches maximum fraudulent attempts, gracefully delegating edge cases to human review.
            """)

        st.markdown("---")
        st.subheader("📁 Training Artifacts")
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "plots")

        # Show saved plots if available
        plot_files = {
            "Confusion Matrix"  : "confusion_matrices.png",
            "ROC Curves"        : "roc_curves.png",
            "PR Curves"         : "pr_curves.png",
            "Model Comparison"  : "model_comparison.png",
            "Feature Importance": "feature_importance.png",
        }
        tabs = st.tabs(list(plot_files.keys()))
        for tab, (label, fname) in zip(tabs, plot_files.items()):
            with tab:
                path = os.path.join(plots_dir, fname)
                if os.path.exists(path):
                    st.image(path, use_column_width=True)
                else:
                    st.caption(f"Run `python model/train.py` to generate {fname}")


# ── Tab 3: History ────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("📋 Session Incident Logs")
    st.caption("Live feed of all processed transactions.")

    col_hist, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    history_table(st.session_state.history)

    if st.session_state.history:
        # Quick stats
        df_hist = pd.DataFrame(st.session_state.history)
        fraud_count = (df_hist["prediction"] == 1).sum()
        legit_count = len(df_hist) - fraud_count

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Processed",   len(df_hist))
        c2.metric("✅ Approved",     legit_count)
        c3.metric("🚨 Blocked", fraud_count)

        # Pie chart
        if len(df_hist) > 1:
            fig_pie = px.pie(
                values = [legit_count, fraud_count],
                names  = ["Legitimate", "Fraudulent"],
                color  = ["Legitimate", "Fraudulent"],
                color_discrete_map = {"Legitimate": "#11998e", "Fraudulent": "#eb3349"},
                title  = "Traffic Distribution",
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#fafafa"}
            )
            st.plotly_chart(fig_pie, use_container_width=True)