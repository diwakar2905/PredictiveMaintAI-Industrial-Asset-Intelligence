"""
app.py  —  Streamlit Dashboard
-------------------------------
Interactive UI for the Predictive Maintenance Platform.
Run: streamlit run dashboard/app.py
"""

try:
    import streamlit as st  # type: ignore[reportMissingImports]
except ImportError as e:
    raise ImportError(
        "streamlit is not installed. Please install it with `pip install streamlit` "
        "and re-run the dashboard (streamlit run dashboard/app.py)."
    ) from e
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import sys
import os
import io
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="PredictiveMaintAI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary: #00d4ff;
        --warning: #ff9800;
        --danger: #ff3b3b;
        --success: #00e676;
        --bg-card: rgba(255,255,255,0.04);
    }

    .main { background: #0a0e1a; }
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,212,255,0.02));
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #8892a4; letter-spacing: 0.05em; text-transform: uppercase; }

    /* Risk badges */
    .risk-low      { color: #00e676; background: rgba(0,230,118,0.1); border: 1px solid rgba(0,230,118,0.3); }
    .risk-medium   { color: #ff9800; background: rgba(255,152,0,0.1); border: 1px solid rgba(255,152,0,0.3); }
    .risk-high     { color: #ff5722; background: rgba(255,87,34,0.1); border: 1px solid rgba(255,87,34,0.3); }
    .risk-critical { color: #ff3b3b; background: rgba(255,59,59,0.1); border: 1px solid rgba(255,59,59,0.5);
                     animation: pulse 1.5s infinite; }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255,59,59,0.4); }
        50%       { box-shadow: 0 0 0 8px rgba(255,59,59,0); }
    }

    .risk-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* Alert boxes */
    .alert-critical { background: rgba(255,59,59,0.1); border-left: 4px solid #ff3b3b; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0; }
    .alert-warning  { background: rgba(255,152,0,0.1); border-left: 4px solid #ff9800; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0; }
    .alert-ok       { background: rgba(0,230,118,0.1); border-left: 4px solid #00e676; padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 6px 0; }

    /* Section headers */
    .section-title {
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8892a4;
        margin: 1.5rem 0 0.75rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        padding-bottom: 6px;
    }

    /* Sidebar */
    .css-1d391kg { background: #090d18; }

    /* Plotly chart backgrounds */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }

    h1 { color: #e8eaf6 !important; }
    p, li { color: #b0bec5; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}. Make sure the backend is running.")
        return None
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None


def api_post(endpoint: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_URL}.")
        return None
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return None


def risk_color(category: str) -> str:
    return {"Low": "#00e676", "Medium": "#ff9800", "High": "#ff5722", "Critical": "#ff3b3b"}.get(category, "#ccc")


def gauge_chart(value: float, title: str, max_val: float = 100,
                color: str = "#00d4ff", unit: str = "") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#8892a4"}},
        number={"suffix": unit, "font": {"size": 24, "color": "#e8eaf6"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#8892a4"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.04)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, max_val * 0.33], "color": "rgba(0,230,118,0.08)"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "rgba(255,152,0,0.08)"},
                {"range": [max_val * 0.66, max_val], "color": "rgba(255,59,59,0.08)"},
            ],
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def line_chart(x, y_dict: dict, title: str) -> go.Figure:
    fig = go.Figure()
    colors = ["#00d4ff", "#ff9800", "#00e676", "#ff3b3b", "#a78bfa", "#34d399"]
    for i, (name, vals) in enumerate(y_dict.items()):
        fig.add_trace(go.Scatter(
            x=x, y=vals, name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            mode="lines"
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8eaf6", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0bec5")),
        height=280, margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig


def bar_chart(names: list, values: list, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker=dict(
            color=values[::-1],
            colorscale=[[0, "#1a2035"], [0.5, "#0066cc"], [1.0, "#00d4ff"]],
            showscale=False
        )
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8eaf6", size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,14,26,0.8)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
        height=320, margin=dict(l=160, r=20, t=40, b=20)
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ PredictiveMaintAI")
    st.markdown("*Industrial Asset Intelligence*")
    st.divider()

    # Health check
    health = api_get("/health")
    if health:
        status_color = "#00e676" if health.get("status") == "ok" else "#ff3b3b"
        st.markdown(f"**API Status:** <span style='color:{status_color}'>● Online</span>", unsafe_allow_html=True)
        trained_color = "#00e676" if health.get("trained") else "#ff9800"
        trained_text = "Trained ✓" if health.get("trained") else "Not trained"
        st.markdown(f"**Models:** <span style='color:{trained_color}'>{trained_text}</span>", unsafe_allow_html=True)
    else:
        st.markdown("**API Status:** <span style='color:#ff3b3b'>● Offline</span>", unsafe_allow_html=True)

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📂 Upload & Train", "🔍 Analysis", "📋 Report"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(
        "<small style='color:#4a5568'>Built with Flask + PyTorch + XGBoost + Streamlit</small>",
        unsafe_allow_html=True
    )


# ─── Page: Upload & Train ────────────────────────────────────────────────────

if page == "📂 Upload & Train":
    st.markdown("# 📂 Data Upload & Model Training")

    st.info("""
    **Required CSV format:**
    - `timestamp` column (any datetime format)
    - `machine_id` column (optional)
    - One or more numeric sensor columns (e.g., temperature, vibration, pressure)
    """)

    # Download sample data button
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from utils.generate_sample_data import generate_sensor_data
        sample_df = generate_sensor_data()
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_sensor_data.csv",
            mime="text/csv"
        )
    except Exception:
        pass

    st.divider()

    uploaded_file = st.file_uploader("Upload your sensor CSV file", type=["csv"])

    if uploaded_file:
        with st.spinner("Uploading and processing data..."):
            result = api_post("/upload_data", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")})

        if result:
            st.success(f"✅ {result['message']}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", result["rows"])
            col2.metric("Sensor Columns", len(result["sensor_columns"]))
            col3.metric("Total Features", result["total_features"])

            st.markdown("**Sensor columns detected:**", )
            st.code(", ".join(result["sensor_columns"]))

            st.markdown("**Data Preview (first 5 rows):**")
            st.dataframe(pd.DataFrame(result["preview"]), use_container_width=True)

            st.divider()
            st.markdown("### 🚀 Train Models")
            st.markdown("Click below to train LSTM, XGBoost, Isolation Forest, and RUL models.")

            if st.button("▶ Train All Models", use_container_width=True, type="primary"):
                with st.spinner("Training models... (this may take 30–60 seconds)"):
                    train_result = api_post("/train_model")

                if train_result:
                    st.success(f"✅ {train_result['message']}")
                    st.markdown("**Models trained:**")
                    for m in train_result["models_trained"]:
                        st.markdown(f"  - ✓ {m}")
                    st.markdown("**Top predictive features:**")
                    for f in train_result["top_features"]:
                        st.markdown(f"  - `{f}`")
                    st.balloons()
                    st.info("Navigate to **🏠 Dashboard** to see predictions!")


# ─── Page: Dashboard ─────────────────────────────────────────────────────────

elif page == "🏠 Dashboard":
    st.markdown("# ⚙️ Asset Risk Intelligence Dashboard")

    health = api_get("/health")
    if not health or not health.get("trained"):
        st.warning("⚠️ Models not trained yet. Go to **📂 Upload & Train** first.")
        st.stop()

    # Fetch all data
    with st.spinner("Loading dashboard data..."):
        report = api_get("/full_report")
        pred = api_get("/predict")
        rul = api_get("/get_rul")
        anm = api_get("/anomaly_score")

    if not all([report, pred, rul, anm]):
        st.error("Failed to load dashboard data.")
        st.stop()

    # ── Top KPI row ──
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    risk_score = report["risk_score"]
    risk_cat = report["risk_category"]
    risk_col = risk_color(risk_cat)

    with col1:
        st.plotly_chart(gauge_chart(risk_score, "Risk Score", 100, risk_col, ""), use_container_width=True)

    with col2:
        fp = pred["ensemble_failure_prob"] * 100
        st.plotly_chart(gauge_chart(fp, "Failure Probability", 100, "#ff9800", "%"), use_container_width=True)

    with col3:
        rul_h = rul["rul"]["latest_rul_hours"]
        st.plotly_chart(gauge_chart(rul_h, "Remaining Useful Life", 125, "#00d4ff", "h"), use_container_width=True)

    with col4:
        anm_s = anm["summary"]["latest_score"] * 100
        st.plotly_chart(gauge_chart(anm_s, "Anomaly Score", 100, "#a78bfa", ""), use_container_width=True)

    # ── Risk Category Banner ──
    st.markdown(
        f"<div style='text-align:center; margin: 10px 0 20px;'>"
        f"<span class='risk-badge risk-{risk_cat.lower()}'>● {risk_cat.upper()} RISK</span>"
        f"&nbsp;&nbsp;<span style='color:#8892a4; font-size:0.9rem'>"
        f"Machine: {report['machine_id']}</span></div>",
        unsafe_allow_html=True
    )

    # ── Model Comparison ──
    st.markdown('<div class="section-title">Model Predictions</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("XGBoost Failure Prob", f"{pred['xgb_failure_prob']:.1%}")
    with col2:
        lstm_val = pred.get("lstm_failure_prob")
        st.metric("LSTM Failure Prob", f"{lstm_val:.1%}" if lstm_val is not None else "N/A (need more data)")
    with col3:
        st.metric("Ensemble (Final)", f"{pred['ensemble_failure_prob']:.1%}")

    st.markdown(f"**Failure Reason:** *{pred['failure_reason']}*")

    # ── Alerts ──
    st.markdown('<div class="section-title">Active Alerts</div>', unsafe_allow_html=True)
    for alert in report["alerts"]:
        level = alert["level"].lower()
        css_class = f"alert-{level}" if level in ["critical", "warning", "ok"] else "alert-ok"
        st.markdown(
            f"<div class='{css_class}'><strong>{alert['message']}</strong><br>"
            f"<small style='color:#8892a4'>→ {alert['action']}</small></div>",
            unsafe_allow_html=True
        )

    # ── Cost Estimate ──
    st.markdown('<div class="section-title">Financial Impact Estimate</div>', unsafe_allow_html=True)
    cost = report["cost_estimate"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Failure Cost", f"${cost['expected_failure_cost_usd']:,.0f}")
    c2.metric("Planned Maintenance Cost", f"${cost['planned_maintenance_cost_usd']:,.0f}")
    c3.metric("Potential Savings", f"${cost['potential_savings_usd']:,.0f}",
              delta="by acting now" if cost['potential_savings_usd'] > 0 else None)

    st.info(f"💡 **Recommendation:** {cost['recommendation']}")


# ─── Page: Analysis ───────────────────────────────────────────────────────────

elif page == "🔍 Analysis":
    st.markdown("# 🔍 Deep Analysis")

    health = api_get("/health")
    if not health or not health.get("trained"):
        st.warning("⚠️ Models not trained yet.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📈 Sensor Trends", "🎯 Feature Importance", "⚠️ Anomaly Timeline"])

    with tab1:
        sensor_data = api_get("/sensor_series")
        if sensor_data:
            ts = sensor_data.pop("timestamp")
            if sensor_data:
                fig = line_chart(ts, sensor_data, "Sensor Readings Over Time")
                st.plotly_chart(fig, use_container_width=True)

                # RUL trend
                rul = api_get("/get_rul")
                if rul:
                    rul_series = rul["rul_series"]
                    x_range = list(range(len(rul_series)))
                    fig2 = line_chart(x_range, {"RUL (hours)": rul_series},
                                      "Remaining Useful Life Trend")
                    st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fi = api_get("/feature_importance")
        if fi:
            fig = bar_chart(fi["features"], fi["importance"], "Top 10 SHAP Feature Importances")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **How to read this:** Features with higher importance scores had a
            greater influence on the model's failure predictions.
            Lag and rolling features capture temporal degradation patterns.
            """)

    with tab3:
        anm = api_get("/anomaly_score")
        if anm:
            summary = anm["summary"]
            scores = anm["score_series"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Points", f"{summary['anomaly_count']} / {summary['total_points']}")
            c2.metric("Anomaly %", f"{summary['anomaly_pct']}%")
            c3.metric("Latest Score", f"{summary['latest_score']:.3f}",
                      delta="⚠️ ACTIVE" if summary["latest_is_anomaly"] else "✓ Normal",
                      delta_color="inverse" if summary["latest_is_anomaly"] else "normal")

            x_range = list(range(len(scores)))
            threshold = 0.5

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_range, y=scores, name="Anomaly Score",
                fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
                line=dict(color="#a78bfa", width=2)
            ))
            fig.add_hline(y=threshold, line_dash="dash",
                          line_color="#ff3b3b", annotation_text="Threshold")
            fig.update_layout(
                title="Anomaly Score Timeline",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,14,26,0.8)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892a4"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#8892a4", range=[0, 1]),
                height=300, margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Page: Report ─────────────────────────────────────────────────────────────

elif page == "📋 Report":
    st.markdown("# 📋 Full Business Report")

    health = api_get("/health")
    if not health or not health.get("trained"):
        st.warning("⚠️ Models not trained yet.")
        st.stop()

    report = api_get("/full_report")
    if not report:
        st.stop()

    st.json(report)

    # Export as JSON
    report_json = json.dumps(report, indent=2)
    st.download_button(
        label="⬇️ Download Report (JSON)",
        data=report_json,
        file_name=f"maintenance_report_{report['machine_id']}.json",
        mime="application/json"
    )
