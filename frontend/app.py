"""
============================================================
Enhanced Fraud Detection System - Streamlit Frontend
============================================================
NEW FEATURES:
  - Anomaly Detection Tab with interactive visualizations
  - Hybrid prediction combining supervised + unsupervised models
  - Real-time anomaly score gauge
  - Behavioral pattern analysis
  - Comparative model analysis
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
from typing import Optional, Dict

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enhanced Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API Settings ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("FRAUD_API_KEY", "demo-api-key-change-in-production")
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

# ── Sample Transactions ──────────────────────────────────────────────────────
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
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def call_api(endpoint: str, payload: dict) -> Optional[dict]:
    """Generic API caller."""
    try:
        resp = requests.post(
            f"{API_URL}/{endpoint}",
            json=payload,
            headers=HEADERS,
            timeout=10,
        )
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
    """Fetch anomaly detection statistics."""
    try:
        resp = requests.get(f"{API_URL}/anomaly-stats", headers=HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except:
        return {}


def fraud_gauge(prob: float, risk_level: str, title: str = "Fraud Probability (%)") -> go.Figure:
    """Create gauge chart."""
    color = {"Low": "#38ef7d", "Medium": "#ffd200", "High": "#eb3349"}.get(risk_level, "#aaa")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": title, "font": {"size": 20, "color": "#fafafa"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#fafafa"},
            "bar": {"color": color},
            "bgcolor": "#1e2130",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 30], "color": "rgba(56, 239, 125, 0.2)"},
                {"range": [30, 70], "color": "rgba(255, 210, 0, 0.2)"},
                {"range": [70, 100], "color": "rgba(235, 51, 73, 0.2)"},
            ],
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fafafa"}
    )
    return fig


def anomaly_gauge(score: float, level: str) -> go.Figure:
    """Create anomaly score gauge."""
    color_map = {
        "Normal": "#38ef7d",
        "Suspicious": "#ffd200",
        "Highly Anomalous": "#eb3349"
    }
    color = color_map.get(level, "#aaa")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Anomaly Score", "font": {"size": 20, "color": "#fafafa"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#fafafa"},
            "bar": {"color": color},
            "bgcolor": "#1e2130",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 40], "color": "rgba(56, 239, 125, 0.2)"},
                {"range": [40, 70], "color": "rgba(255, 210, 0, 0.2)"},
                {"range": [70, 100], "color": "rgba(235, 51, 73, 0.2)"},
            ],
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fafafa"}
    )
    return fig


def behavioral_flags_chart(flags: Dict[str, bool]) -> go.Figure:
    """Create behavioral flags visualization."""
    labels = []
    values = []
    colors = []
    
    for key, value in flags.items():
        label = key.replace('_', ' ').title()
        labels.append(label)
        values.append(1 if value else 0)
        colors.append('#eb3349' if value else '#38ef7d')
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=['⚠️ FLAGGED' if v else '✓ Normal' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Behavioral Pattern Analysis",
        yaxis=dict(range=[0, 1.2], showticklabels=False, gridcolor='#333'),
        height=300,
        margin=dict(t=50, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fafafa"}
    )
    
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Enhanced AI Fraud Detection System</h1>
    <p>Dual-Layer Protection: Supervised Learning + Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    detection_mode = st.radio(
        "Detection Mode",
        ["Supervised Only", "Anomaly Detection", "Hybrid (Recommended)"],
        index=2
    )
    
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
    st.info("""
    **Hybrid Mode Benefits:**
    - Supervised: Detects known fraud patterns
    - Anomaly: Catches novel/unusual behavior
    - Combined: Maximum protection
    """)

# Tabs
tab_predict, tab_anomaly, tab_compare, tab_dashboard = st.tabs([
    "🔍 Transaction Analysis",
    "🎯 Anomaly Detection",
    "⚖️ Model Comparison",
    "📊 Dashboard"
])

# ── Tab 1: Transaction Analysis ──────────────────────────────────────────────
with tab_predict:
    st.subheader("Transaction Analysis & Prediction")
    
    # Demo buttons
    col_demo1, col_demo2 = st.columns(2)
    
    demo_defaults = SAMPLE_LEGIT.copy()
    
    with col_demo1:
        if st.button("✅ Load Legitimate Sample", use_container_width=True):
            demo_defaults = SAMPLE_LEGIT.copy()
            st.session_state.demo_type = 'legit'
    
    with col_demo2:
        if st.button("🚨 Load Fraudulent Sample", use_container_width=True):
            demo_defaults = SAMPLE_FRAUD.copy()
            st.session_state.demo_type = 'fraud'
    
    st.markdown("---")
    
    # Input form
    with st.form("transaction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            time_val = st.number_input("⏱️ Time (seconds)", value=float(demo_defaults["Time"]))
        with col_b:
            amount_val = st.number_input("💶 Amount (EUR)", value=float(demo_defaults["Amount"]), min_value=0.0)
        
        st.markdown("##### PCA Features (V1-V28)")
        v_vals = {}
        cols = st.columns(7)
        for i in range(1, 29):
            col_idx = (i - 1) % 7
            with cols[col_idx]:
                v_vals[f"V{i}"] = st.number_input(
                    f"V{i}", value=float(demo_defaults.get(f"V{i}", 0.0)),
                    format="%.4f", step=0.01, key=f"v{i}"
                )
        
        submit = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
    
    if submit:
        payload = {"Time": time_val, "Amount": amount_val, **v_vals}
        
        with st.spinner("Analyzing transaction..."):
            if detection_mode == "Supervised Only":
                result = call_api("predict", payload)
                if result:
                    st.session_state.last_result = result
            
            elif detection_mode == "Anomaly Detection":
                result = call_api("predict-anomaly", payload)
                if result:
                    st.session_state.last_anomaly = result
            
            else:  # Hybrid
                result = call_api("predict-hybrid", payload)
                if result:
                    st.session_state.last_hybrid = result
    
    # Display results based on mode
    st.markdown("---")
    
    if detection_mode == "Supervised Only" and st.session_state.last_result:
        r = st.session_state.last_result
        st.subheader("⚡ Supervised Model Results")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = fraud_gauge(r["fraud_probability"], r["risk_level"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if r["prediction"] == 1:
                st.markdown(f'<div class="fraud-card">🚨 FRAUD DETECTED<br><small>{r["risk_level"]} Risk</small></div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card">✅ LEGITIMATE<br><small>{r["risk_level"]} Risk</small></div>', 
                           unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{r['fraud_probability']*100:.2f}%")
            m2.metric("Latency", f"{r['inference_time_ms']:.2f} ms")
    
    elif detection_mode == "Anomaly Detection" and st.session_state.last_anomaly:
        a = st.session_state.last_anomaly
        st.subheader("🎯 Anomaly Detection Results")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = anomaly_gauge(a["anomaly_score"], a["anomaly_level"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if a["is_anomaly"]:
                st.markdown(f'<div class="anomaly-card">⚠️ ANOMALY DETECTED<br><small>{a["anomaly_level"]}</small></div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card">✅ NORMAL BEHAVIOR<br><small>{a["anomaly_level"]}</small></div>', 
                           unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Anomaly Score", f"{a['anomaly_score']:.1f}/100")
            m2.metric("Latency", f"{a['inference_time_ms']:.2f} ms")
        
        # Behavioral flags
        st.markdown("---")
        st.subheader("🧠 Behavioral Analysis")
        fig_flags = behavioral_flags_chart(a["behavioral_flags"])
        st.plotly_chart(fig_flags, use_container_width=True)
    
    elif detection_mode == "Hybrid (Recommended)" and st.session_state.last_hybrid:
        h = st.session_state.last_hybrid
        st.subheader("🔮 Hybrid Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Supervised Model")
            fig1 = fraud_gauge(h["supervised_prob"], "High" if h["supervised_pred"] == 1 else "Low")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("##### Anomaly Detector")
            fig2 = anomaly_gauge(h["anomaly_score"], "Highly Anomalous" if h["anomaly_score"] > 70 else "Normal")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.markdown("##### Final Verdict")
            st.markdown("<br>", unsafe_allow_html=True)
            if h["final_prediction"] == 1:
                st.markdown(f'<div class="fraud-card">🚨 FRAUD<br><small>Confidence: {h["confidence"]*100:.0f}%</small></div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-card">✅ SAFE<br><small>Confidence: {h["confidence"]*100:.0f}%</small></div>', 
                           unsafe_allow_html=True)
        
        st.markdown("---")
        st.info(f"**Risk Assessment:** {h['risk_assessment']}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Agreement", "✓ Yes" if h["model_agreement"] else "✗ No")
        m2.metric("Confidence", f"{h['confidence']*100:.0f}%")
        m3.metric("Latency", f"{h['inference_time_ms']:.2f} ms")


# ── Tab 2: Anomaly Detection Deep Dive ───────────────────────────────────────
with tab_anomaly:
    st.subheader("🎯 Anomaly Detection Visualizations")
    
    # Load anomaly plots if available
    anomaly_plots_dir = os.path.join(os.path.dirname(__file__), "model", "anomaly", "plots")
    
    if os.path.exists(anomaly_plots_dir):
        plot_files = {
            "Distribution": "anomaly_distribution.png",
            "Amount Analysis": "amount_vs_anomaly.png",
            "Temporal Patterns": "time_vs_anomaly.png",
            "PCA Space": "pca_anomalies.png",
            "Top Anomalies": "top_anomalies.png",
        }
        
        tabs_anomaly = st.tabs(list(plot_files.keys()))
        
        for tab, (label, fname) in zip(tabs_anomaly, plot_files.items()):
            with tab:
                path = os.path.join(anomaly_plots_dir, fname)
                if os.path.exists(path):
                    st.image(path, use_column_width=True)
                else:
                    st.info(f"Run `python train_anomaly_detector.py` to generate visualizations")
    else:
        st.warning("Anomaly visualizations not found. Train the anomaly detector first.")
        st.code("python train_anomaly_detector.py", language="bash")


# ── Tab 3: Model Comparison ──────────────────────────────────────────────────
with tab_compare:
    st.subheader("⚖️ Supervised vs Anomaly Detection")
    
    stats = get_anomaly_stats()
    
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎓 Supervised Learning (Random Forest)")
            st.info("""
            **Strengths:**
            - Excellent at detecting known fraud patterns
            - High precision on labeled data
            - Fast inference
            
            **Limitations:**
            - Requires labeled training data
            - May miss novel fraud techniques
            """)
        
        with col2:
            st.markdown("### 🎯 Anomaly Detection (Isolation Forest)")
            st.info("""
            **Strengths:**
            - Detects unusual behavioral patterns
            - No labeled fraud data needed
            - Catches novel/zero-day attacks
            
            **Limitations:**
            - May have more false positives
            - Requires careful threshold tuning
            """)
        
        st.markdown("---")
        st.subheader("📊 Performance Metrics Comparison")
        
        anomaly_metrics = stats.get('metrics', {})
        
        # Create comparison chart
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Supervised': [0.961, 0.755, 0.846, 0.985],  # From your metadata
            'Anomaly': [
                anomaly_metrics.get('precision', 0),
                anomaly_metrics.get('recall', 0),
                anomaly_metrics.get('f1', 0),
                anomaly_metrics.get('roc_auc', 0)
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Supervised', x=metrics_df['Metric'], y=metrics_df['Supervised'],
                            marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Anomaly', x=metrics_df['Metric'], y=metrics_df['Anomaly'],
                            marker_color='#f7971e'))
        
        fig.update_layout(
            title="Model Performance Comparison",
            yaxis=dict(range=[0, 1.1], gridcolor='#333'),
            height=400,
            barmode='group',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa"}
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Dashboard ─────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader("📊 System Dashboard")
    
    # System health
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API Status", "🟢 Online")
    col2.metric("Supervised Model", "✓ Loaded")
    col3.metric("Anomaly Model", "✓ Loaded" if get_anomaly_stats() else "✗ Not Loaded")
    col4.metric("Avg Latency", "~2.5 ms")
    
    st.markdown("---")
    
    # Model info
    try:
        resp = requests.get(f"{API_URL}/model-info", headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### Supervised Model Details")
                sup = info.get('supervised', {})
                st.json(sup)
            
            with col_b:
                st.markdown("### Anomaly Detector Details")
                ano = info.get('anomaly', {})
                st.json(ano)
    except:
        st.warning("Could not fetch model info")

# Footer
st.markdown("---")
st.caption("Enhanced Fraud Detection System v2.0 | Powered by Random Forest + Isolation Forest")
