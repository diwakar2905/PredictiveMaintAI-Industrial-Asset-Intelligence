"""
main.py  —  FastAPI Backend
---------------------------
REST API for the Predictive Maintenance Platform.

Endpoints:
  POST /upload_data     - Upload and validate CSV
  POST /train_model     - Train all ML models
  POST /predict         - Get failure prediction
  GET  /get_rul         - Get RUL estimate
  GET  /anomaly_score   - Get anomaly scores
  GET  /health          - Health check
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Ensure project root is on sys.path so `from src...` imports work
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features, create_sequences
# model_lstm depends on PyTorch which may not be installed in lightweight setups.
# Import it lazily and fall back to stubs so the API can still start for non-LSTM flows.
try:
    from src.model_lstm import train_lstm, predict_lstm, create_synthetic_labels, LSTMClassifier
    LSTM_AVAILABLE = True
except Exception as _err:
    LSTM_AVAILABLE = False
    def train_lstm(*args, **kwargs):
        raise RuntimeError("LSTM training unavailable because PyTorch failed to import: %s" % _err)
    def predict_lstm(*args, **kwargs):
        raise RuntimeError("LSTM prediction unavailable because PyTorch failed to import: %s" % _err)
    def create_synthetic_labels(n):
        # Fallback: return zeros (no failures)
        return [0] * n
    class LSTMClassifier:  # placeholder
        pass
from src.model_xgb import train_xgb, predict_xgb, get_shap_values, get_failure_reason
from src.anomaly import train_isolation_forest, get_anomaly_scores, get_anomaly_labels, anomaly_summary
from src.rul import train_rul_model, predict_rul, rul_summary, create_rul_labels
from src.business_logic import generate_business_report

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Predictive Maintenance API",
    description="AI-powered industrial asset risk intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory state (use Redis/DB in production) ─────────────────────────────

STATE = {
    "df": None,
    "feature_df": None,
    "feature_cols": None,
    "sensor_cols": None,
    "scaler": None,
    "xgb_model": None,
    "lstm_model": None,
    "iso_forest": None,
    "rul_model": None,
    "shap_info": None,
    "is_trained": False,
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "trained": STATE["is_trained"]}


@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Accepts a CSV file, validates schema, preprocesses, engineers features.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    # Preprocess
    processed_df, scaler, sensor_cols, validation = preprocess_pipeline(df)

    if not validation["valid"]:
        raise HTTPException(400, validation["message"])

    # Feature engineering
    feature_df, feature_cols = engineer_features(processed_df, sensor_cols)

    # Save state
    STATE["df"] = processed_df
    STATE["feature_df"] = feature_df
    STATE["feature_cols"] = feature_cols
    STATE["sensor_cols"] = sensor_cols
    STATE["scaler"] = scaler

    return {
        "message": "Data uploaded and processed successfully.",
        "rows": len(feature_df),
        "sensor_columns": sensor_cols,
        "total_features": len(feature_cols),
        "preview": feature_df.head(5).to_dict(orient="records"),
    }


@app.post("/train_model")
def train_model():
    """
    Trains all ML models: LSTM, XGBoost, Isolation Forest, RUL.
    """
    if STATE["feature_df"] is None:
        raise HTTPException(400, "No data loaded. Call /upload_data first.")

    feature_df = STATE["feature_df"]
    feature_cols = STATE["feature_cols"]
    n = len(feature_df)

    if n < 20:
        raise HTTPException(400, f"Need at least 20 rows for training. Got {n}.")

    X = feature_df[feature_cols].values.astype(np.float32)
    y_labels = create_synthetic_labels(n)
    y_rul = create_rul_labels(n)

    # ── XGBoost ──────────────────────────────────────────────
    xgb_model = train_xgb(X, y_labels)
    STATE["xgb_model"] = xgb_model

    # SHAP
    shap_info = get_shap_values(xgb_model, X, feature_cols)
    STATE["shap_info"] = shap_info

    # ── LSTM ──────────────────────────────────────────────────
    SEQ_LEN = min(10, n // 3)
    X_seq, y_seq = create_sequences(X, y_labels, seq_len=SEQ_LEN)

    if len(X_seq) >= 10:
        lstm_model, losses = train_lstm(X_seq, y_seq, epochs=15)
        STATE["lstm_model"] = lstm_model
        STATE["seq_len"] = SEQ_LEN
    else:
        STATE["lstm_model"] = None  # Not enough data for LSTM

    # ── Isolation Forest ──────────────────────────────────────
    iso_forest = train_isolation_forest(X)
    STATE["iso_forest"] = iso_forest

    # ── RUL Model ─────────────────────────────────────────────
    rul_model = train_rul_model(X, y_rul)
    STATE["rul_model"] = rul_model

    STATE["is_trained"] = True

    # Save models to disk
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "iso_forest.pkl"))
    joblib.dump(rul_model, os.path.join(MODELS_DIR, "rul_model.pkl"))
    if STATE["lstm_model"]:
        torch.save(STATE["lstm_model"].state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))

    return {
        "message": "All models trained successfully!",
        "models_trained": ["XGBoost", "Isolation Forest", "RUL (Ridge)"] + (
            ["LSTM"] if STATE["lstm_model"] else []
        ),
        "top_features": list(shap_info["top_features"].keys())[:5],
    }


@app.get("/predict")
def predict():
    """
    Returns failure prediction from LSTM and XGBoost for the latest data window.
    """
    _check_trained()
    feature_df = STATE["feature_df"]
    feature_cols = STATE["feature_cols"]
    X = feature_df[feature_cols].values.astype(np.float32)

    # XGBoost prediction (last sample)
    xgb_prob = float(predict_xgb(STATE["xgb_model"], X)[-1])

    # LSTM prediction (last sequence)
    lstm_prob = None
    if STATE["lstm_model"] is not None:
        seq_len = STATE.get("seq_len", 10)
        if len(X) >= seq_len:
            X_last = X[-seq_len:].reshape(1, seq_len, -1)
            lstm_prob = float(predict_lstm(STATE["lstm_model"], X_last)[0])

    # Ensemble: average if both available
    if lstm_prob is not None:
        ensemble_prob = 0.6 * lstm_prob + 0.4 * xgb_prob
    else:
        ensemble_prob = xgb_prob

    # Failure reason
    shap_info = STATE["shap_info"]
    failure_reason = get_failure_reason(shap_info["top_features"]) if shap_info else "N/A"

    return {
        "xgb_failure_prob": round(xgb_prob, 4),
        "lstm_failure_prob": round(lstm_prob, 4) if lstm_prob is not None else None,
        "ensemble_failure_prob": round(ensemble_prob, 4),
        "failure_reason": failure_reason,
    }


@app.get("/get_rul")
def get_rul():
    """Returns Remaining Useful Life estimate."""
    _check_trained()
    X = STATE["feature_df"][STATE["feature_cols"]].values.astype(np.float32)
    rul_values = predict_rul(STATE["rul_model"], X)
    summary = rul_summary(rul_values)
    return {"rul": summary, "rul_series": rul_values.tolist()[-50:]}  # last 50 for chart


@app.get("/anomaly_score")
def anomaly_score():
    """Returns anomaly scores and summary."""
    _check_trained()
    X = STATE["feature_df"][STATE["feature_cols"]].values.astype(np.float32)
    scores = get_anomaly_scores(STATE["iso_forest"], X)
    labels = get_anomaly_labels(STATE["iso_forest"], X)
    summary = anomaly_summary(scores, labels)
    return {"summary": summary, "score_series": scores.tolist()[-50:]}


@app.get("/full_report")
def full_report():
    """Generates a complete business intelligence report."""
    _check_trained()

    pred = predict()
    rul = get_rul()
    anm = anomaly_score()

    machine_id = "MACHINE-001"
    if STATE["df"] is not None and "machine_id" in STATE["df"].columns:
        machine_id = str(STATE["df"]["machine_id"].iloc[-1])

    report = generate_business_report(
        failure_prob=pred["ensemble_failure_prob"],
        anomaly_score=anm["summary"]["latest_score"],
        rul_hours=rul["rul"]["latest_rul_hours"],
        anomaly_is_active=anm["summary"]["latest_is_anomaly"],
        failure_reason=pred["failure_reason"],
        machine_id=machine_id
    )
    return report


@app.get("/feature_importance")
def feature_importance():
    """Returns top SHAP feature importances."""
    _check_trained()
    shap_info = STATE["shap_info"]
    if shap_info is None:
        raise HTTPException(400, "SHAP not computed.")
    top = shap_info["mean_abs_shap"].head(10)
    return {
        "features": top.index.tolist(),
        "importance": top.values.tolist()
    }


@app.get("/sensor_series")
def sensor_series():
    """Returns raw sensor time series for plotting."""
    if STATE["df"] is None:
        raise HTTPException(400, "No data loaded.")
    df = STATE["df"]
    sensor_cols = STATE["sensor_cols"]
    result = {"timestamp": df["timestamp"].astype(str).tolist()[-100:]}
    for col in sensor_cols[:6]:  # Max 6 sensors
        result[col] = df[col].tolist()[-100:]
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_trained():
    if not STATE["is_trained"]:
        raise HTTPException(400, "Models not trained yet. Call /train_model first.")
