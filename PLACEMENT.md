# Placement Guide — Predictive Maintenance Project

This document helps you explain the project clearly in interviews and placement settings. It contains: a short elevator pitch, architecture overview, model explanations, deployment notes, how to explain metrics/impact numbers, common interview questions with sample answers, and demo/talking-point scripts.

---

**Elevator pitch (1 line)**

Built an end-to-end AI predictive maintenance platform that ingests time-series sensor CSVs, trains LSTM + XGBoost models with SHAP explainability, estimates Remaining Useful Life (RUL), and exposes a hardened Flask API with a Streamlit/Plotly dashboard for operational risk scoring and cost impact reporting.

**30‑second pitch**

I built a production-ready predictive maintenance system that turns raw sensor streams into failure probability, anomaly detection and RUL estimates. The platform includes model explainability (SHAP), automated feature engineering, an API for training/inference, and a Streamlit dashboard for operations teams to get risk scores, alerts and financial impact estimates.

---

## Where the code lives

- Dashboard UI: [dashboard/app.py](dashboard/app.py)
- Backend API: [src/api/app.py](src/api/app.py)
- LSTM model: [src/model_lstm.py](src/model_lstm.py)
- XGBoost + SHAP: [src/model_xgb.py](src/model_xgb.py)
- RUL logic: [src/rul.py](src/rul.py)
- Business logic (risk, alerts, cost): [src/business_logic.py](src/business_logic.py)

Link to the repo README for setup and run steps: [README.md](README.md)

---

## Architecture (simple explanation)

- Data ingestion: CSV upload endpoint validates schema and converts timestamps into time series.
- Preprocessing & feature engineering: lag, rolling stats, trend features and sequence generation for LSTM.
- Models:
  - XGBoost: fast tree-based classifier for per-sample failure probability and SHAP explainability.
  - LSTM: sequence model for temporal pattern detection and short-term failure forecasting.
  - Isolation Forest: unsupervised anomaly scoring.
  - Ridge regression: simple RUL estimator.
- Business logic: combines probabilities, anomaly score, and RUL into a 0–100 risk score, categories and human alerts.
- Presentation: Streamlit dashboard with Plotly charts and a Flask API for all operations.

---

## How to explain the main technical pieces (easy language)

- LSTM (why): "I used an LSTM to capture temporal patterns — it learns how sensor readings evolve over time and flags sequences that look like prior failures."
- XGBoost (why): "XGBoost is a strong baseline for tabular data; it trains fast and integrates with SHAP to explain which sensors drove a particular prediction."
- SHAP (what it does): "SHAP assigns each feature a contribution score for a prediction so users can see *why* the model thinks failure is likely." 
- RUL (what it is): "Remaining Useful Life estimates how many hours the machine can run before likely failure — helpful for scheduling maintenance rather than reacting to breakdowns."

---

## How to explain metrics / numbers (short scripts you can say)

- R² 0.89: "R² of 0.89 means the regression explains 89% of variance in RUL labels — a strong fit for our simulated labels."
- RMSE 0.08 (or RMSE in original units): "RMSE measures typical prediction error; lower is better — ours is small relative to the label range, indicating tight estimates." 
- AUC / Accuracy: "AUC (area under ROC) tells how well the model ranks failure vs. non-failure; values closer to 1.0 are better. Accuracy is intuitive but can be misleading with imbalanced failures — we report probability-based metrics and precision/recall as well." 
- Business impact (example): "Simulated impact shows maintenance cost reduction of ~25–30% and up to 40% lower downtime — these come from comparing expected unplanned failure costs vs. planned maintenance costs using the platform's cost model in `src/business_logic.py`."

Tips: always state the metric, what direction is better, and why it matters for the business (cost, safety, uptime).

---

## Common interview questions & sample answers

Q: What problem does this solve?

A: "It predicts equipment failures before they happen so teams can schedule maintenance, reducing unplanned downtime and repair costs." 

Q: What data does the model use?

A: "Time-series sensor data: timestamps, optional machine ID, and numeric sensor columns like temperature, vibration and pressure. The system builds lag and rolling features and sequence windows for the LSTM." 

Q: How did you validate the models?

A: "I used holdout / cross-validation where possible, monitored classification metrics (AUC, precision/recall) and regression metrics (MAE/RMSE). For production we emphasize stable models, feature drift checks, and have fallbacks so the dashboard never crashes if a prediction fails." 

Q: Why an ensemble of LSTM and XGBoost?

A: "They capture complementary signals — LSTM captures temporal trends, XGBoost excels on engineered tabular features. Ensemble improves robustness and provides multiple lenses for explainability." 

Q: How is explainability surfaced to users?

A: "We compute SHAP values for XGBoost and display top contributing features; for LSTM we show sequence trends and high-impact features separately. We also translate top features to human-friendly failure reasons (see `src/model_xgb.py`)." 

Q: How would you deploy this in production?

A: "Serve the API via Gunicorn behind a reverse proxy, persist models to disk and S3 for versioning, add health checks and monitoring, and run the Streamlit dashboard as a separate web service or embed UI into an internal portal." 

Q: What would you do next to improve the project?

A: "Add model monitoring and drift detection, implement continuous retraining pipelines, integrate live streaming (Kafka/MQTT), and add auth + RBAC for multi-tenant deployments." 

---

## How to demo the project (step-by-step script)

1. Start the backend: `python -u src/api/app.py` (or `gunicorn src.api.app:app --bind 0.0.0.0:8000`).
2. Start the dashboard: `streamlit run dashboard/app.py`.
3. Show the **📂 Upload & Train** flow: download sample CSV, upload, and click **Train All Models** → explain the training logs and models saved.
4. Go to **🏠 Dashboard**: point out KPIs (risk score, failure probability, RUL), click feature importance and read SHAP top features.
5. Explain a single example: show sensor series, explain why SHAP highlights temperature/vibration and how that matches the failure reason text.

Demo tip: keep the demo short (3–5 minutes) and have one pre-trained model saved in `models/` to avoid long training times during interviews.

---

## How to explain your personal contribution (1–2 lines)

"I designed and implemented the end-to-end pipeline: data validation and feature engineering, LSTM and XGBoost models, SHAP explainability, and the production UI/API to make model outputs actionable for operations teams." 

---

## Quick cheat-sheet: one-liners to memorize

- "Predictive maintenance platform converting sensors → risk scores → cost-savings." 
- "Hybrid LSTM + XGBoost ensemble with SHAP for explainability." 
- "Risk scoring + RUL + alerts translate ML into maintenance actions and dollar-savings." 

---

If you want, I can:

- Tailor this `PLACEMENT.md` into a printable one‑page cheat sheet.
- Insert your actual measured metrics (R², RMSE, AUC, user/time savings) into the text for stronger impact.
- Generate a short slide deck or speak/training notes for a 5‑minute demo.

---

File: PLACEMENT.md — added to repo root for your placement prep.
