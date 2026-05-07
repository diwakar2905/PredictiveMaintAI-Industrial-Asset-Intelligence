"""
Flask WSGI app wrapping the predictive maintenance handlers.

This file exposes a Flask-compatible WSGI entrypoint so you can deploy with
`gunicorn src.api.app:app` on Render.
"""
import os
import io
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_pipeline
from src.feature_engineering import engineer_features, create_sequences
from src.model_xgb import train_xgb, predict_xgb, get_shap_values, get_failure_reason
from src.anomaly import train_isolation_forest, get_anomaly_scores, get_anomaly_labels, anomaly_summary
from src.rul import train_rul_model, predict_rul, rul_summary, create_rul_labels
from src.business_logic import generate_business_report

# Try importing optional torch-based LSTM helpers; provide graceful fallbacks
try:
    from src.model_lstm import train_lstm, predict_lstm, create_synthetic_labels, LSTMClassifier
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False
    def train_lstm(*args, **kwargs):
        raise RuntimeError("LSTM training unavailable")
    def predict_lstm(*args, **kwargs):
        raise RuntimeError("LSTM prediction unavailable")
    def create_synthetic_labels(n):
        return [0] * n
    class LSTMClassifier:
        pass

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__)

# Simple in-memory state (replace with Redis/DB in production)
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


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "trained": STATE["is_trained"]})


@app.route("/upload_data", methods=["POST"])
def upload_data():
    if "file" not in request.files:
        return jsonify({"error": "file field missing"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    try:
        df = pd.read_csv(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {e}"}), 400

    processed_df, scaler, sensor_cols, validation = preprocess_pipeline(df)
    if not validation.get("valid", False):
        return jsonify({"error": validation.get("message", "validation failed")}), 400

    feature_df, feature_cols = engineer_features(processed_df, sensor_cols)

    STATE["df"] = processed_df
    STATE["feature_df"] = feature_df
    STATE["feature_cols"] = feature_cols
    STATE["sensor_cols"] = sensor_cols
    STATE["scaler"] = scaler

    return jsonify({
        "message": "Data uploaded and processed successfully.",
        "rows": len(feature_df),
        "sensor_columns": sensor_cols,
        "total_features": len(feature_cols),
        "preview": feature_df.head(5).to_dict(orient="records"),
    })


@app.route("/train_model", methods=["POST"])
def train_model():
    if STATE["feature_df"] is None:
        return jsonify({"error": "No data loaded. Call /upload_data first."}), 400

    feature_df = STATE["feature_df"]
    feature_cols = STATE["feature_cols"]
    n = len(feature_df)
    if n < 20:
        return jsonify({"error": f"Need at least 20 rows for training. Got {n}."}), 400

    X = feature_df[feature_cols].values.astype(np.float32)
    y_labels = create_synthetic_labels(n)
    y_rul = create_rul_labels(n)

    xgb_model = train_xgb(X, y_labels)
    STATE["xgb_model"] = xgb_model

    shap_info = get_shap_values(xgb_model, X, feature_cols)
    STATE["shap_info"] = shap_info

    SEQ_LEN = min(10, n // 3)
    X_seq, y_seq = create_sequences(X, y_labels, seq_len=SEQ_LEN)

    if len(X_seq) >= 10 and LSTM_AVAILABLE and TORCH_AVAILABLE:
        lstm_model, losses = train_lstm(X_seq, y_seq, epochs=15)
        STATE["lstm_model"] = lstm_model
        STATE["seq_len"] = SEQ_LEN
    else:
        STATE["lstm_model"] = None

    iso_forest = train_isolation_forest(X)
    STATE["iso_forest"] = iso_forest

    rul_model = train_rul_model(X, y_rul)
    STATE["rul_model"] = rul_model

    STATE["is_trained"] = True

    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "iso_forest.pkl"))
    joblib.dump(rul_model, os.path.join(MODELS_DIR, "rul_model.pkl"))
    if STATE["lstm_model"] and TORCH_AVAILABLE:
        torch.save(STATE["lstm_model"].state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))

    return jsonify({
        "message": "All models trained successfully!",
        "models_trained": ["XGBoost", "Isolation Forest", "RUL (Ridge)"] + (["LSTM"] if STATE["lstm_model"] else []),
        "top_features": list(shap_info["top_features"].keys())[:5],
    })


def _check_trained():
    if not STATE["is_trained"]:
        from flask import abort
        abort(400, "Models not trained yet. Call /train_model first.")


@app.route("/predict", methods=["GET"])
def predict():
    _check_trained()
    feature_df = STATE["feature_df"]
    feature_cols = STATE["feature_cols"]
    X = feature_df[feature_cols].values.astype(np.float32)

    xgb_prob = float(predict_xgb(STATE["xgb_model"], X)[-1])

    lstm_prob = None
    if STATE["lstm_model"] is not None:
        seq_len = STATE.get("seq_len", 10)
        if len(X) >= seq_len:
            X_last = X[-seq_len:].reshape(1, seq_len, -1)
            lstm_prob = float(predict_lstm(STATE["lstm_model"], X_last)[0])

    if lstm_prob is not None:
        ensemble_prob = 0.6 * lstm_prob + 0.4 * xgb_prob
    else:
        ensemble_prob = xgb_prob

    shap_info = STATE["shap_info"]
    failure_reason = get_failure_reason(shap_info["top_features"]) if shap_info else "N/A"

    return jsonify({
        "xgb_failure_prob": round(xgb_prob, 4),
        "lstm_failure_prob": round(lstm_prob, 4) if lstm_prob is not None else None,
        "ensemble_failure_prob": round(ensemble_prob, 4),
        "failure_reason": failure_reason,
    })


@app.route("/get_rul", methods=["GET"])
def get_rul():
    _check_trained()
    X = STATE["feature_df"][STATE["feature_cols"]].values.astype(np.float32)
    rul_values = predict_rul(STATE["rul_model"], X)
    summary = rul_summary(rul_values)
    return jsonify({"rul": summary, "rul_series": rul_values.tolist()[-50:]})


@app.route("/anomaly_score", methods=["GET"])
def anomaly_score():
    _check_trained()
    X = STATE["feature_df"][STATE["feature_cols"]].values.astype(np.float32)
    scores = get_anomaly_scores(STATE["iso_forest"], X)
    labels = get_anomaly_labels(STATE["iso_forest"], X)
    summary = anomaly_summary(scores, labels)
    return jsonify({"summary": summary, "score_series": scores.tolist()[-50:]})


@app.route("/full_report", methods=["GET"])
def full_report():
    _check_trained()
    pred = predict().get_json() if hasattr(predict(), 'get_json') else predict()
    rul = get_rul().get_json() if hasattr(get_rul(), 'get_json') else get_rul()
    anm = anomaly_score().get_json() if hasattr(anomaly_score(), 'get_json') else anomaly_score()

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
    return jsonify(report)


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    _check_trained()
    shap_info = STATE["shap_info"]
    if shap_info is None:
        return jsonify({"error": "SHAP not computed."}), 400
    top = shap_info["mean_abs_shap"].head(10)
    return jsonify({"features": top.index.tolist(), "importance": top.values.tolist()})


@app.route("/sensor_series", methods=["GET"])
def sensor_series():
    if STATE["df"] is None:
        return jsonify({"error": "No data loaded."}), 400
    df = STATE["df"]
    sensor_cols = STATE["sensor_cols"]
    result = {"timestamp": df["timestamp"].astype(str).tolist()[-100:]}
    for col in sensor_cols[:6]:
        result[col] = df[col].tolist()[-100:]
    return jsonify(result)


if __name__ == "__main__":
    # Useful for local debugging: run with Flask's built-in server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
