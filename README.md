# ⚙️ AI-Powered Predictive Maintenance & Asset Risk Intelligence Platform

> **Predict machine failures before they happen. Reduce downtime. Save costs.**

A production-ready, end-to-end industrial AI platform that uses time-series sensor data to predict equipment failures, estimate Remaining Useful Life (RUL), detect anomalies, and generate actionable business intelligence.

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [API Reference](#api-reference)
- [Sample Data Format](#sample-data-format)
- [Future Improvements](#future-improvements)

---

## 🏭 Problem Statement

Industrial equipment failures cause **billions in unplanned downtime** annually. Traditional maintenance is either:
- **Reactive** (fix after breakdown — expensive, dangerous)
- **Time-based** (schedule too early or too late)

**Predictive Maintenance** uses sensor data and ML to predict failures *before* they happen, enabling:
- 25–30% reduction in maintenance costs
- 70–75% decrease in breakdowns
- Up to 40% reduction in downtime

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                 │
│  (Upload → Train → Predict → Visualize → Report)    │
└─────────────────────┬───────────────────────────────┘
                       │ HTTP REST
┌─────────────────────▼───────────────────────────────┐
│                   Flask Backend                      │
│  /upload_data  /train_model  /predict  /get_rul      │
│  /anomaly_score  /full_report  /feature_importance   │
└─────────────────────┬───────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌───────────┐  ┌──────────────┐  ┌──────────┐
│   Data    │  │  ML Models   │  │ Business │
│  Pipeline │  │  LSTM / XGB  │  │  Logic   │
│  Preproc  │  │  IsoForest   │  │  Alerts  │
│  FeatEng  │  │  RUL Ridge   │  │  Costs   │
└───────────┘  └──────────────┘  └──────────┘
## 🚀 Quick Start (Local)

---

| Feature | Description |
|---------|-------------|
| 📤 CSV Upload | Upload any sensor CSV with automatic schema validation |
| 🧹 Preprocessing | Missing value handling, normalization, timestamp parsing |
| 🔧 Feature Engineering | Lag, rolling mean/std, trend features auto-generated |
| 🧠 LSTM Model | Deep learning failure prediction (PyTorch) |
| 🌲 XGBoost Model | Gradient boosting baseline with SHAP explainability |
| 🔍 Anomaly Detection | Isolation Forest with normalized anomaly scores |
| ⏱️ RUL Prediction | Remaining Useful Life regression model |
| 📊 Risk Scoring | Composite 0–100 risk score with Low/Medium/High/Critical categories |
| 💰 Cost Estimation | Financial impact of predicted failures vs. planned maintenance |
| 🚨 Smart Alerts | Context-aware alerts with recommended actions |
| 📈 Dashboard | Interactive Plotly charts with gauge indicators |
| 🐳 Docker Ready | One-command deployment |

---

## 📁 Project Structure

```
predictive_maintenance/
│
├── data/                          # Data storage
│   └── sample_sensor_data.csv     # Generated sample data
│
├── models/                        # Saved trained models
│   ├── xgb_model.pkl
│   ├── iso_forest.pkl
│   ├── rul_model.pkl
│   └── lstm_model.pt
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Data validation, cleaning, scaling
│   ├── feature_engineering.py     # Lag/rolling/trend features + sequences
│   ├── model_lstm.py              # PyTorch LSTM classifier
│   ├── model_xgb.py               # XGBoost + SHAP explainability
│   ├── anomaly.py                 # Isolation Forest anomaly detection
│   ├── rul.py                     # Remaining Useful Life regression
│   ├── business_logic.py          # Risk scoring, alerts, cost estimation
│   └── api/
│       └── app.py                 # Flask WSGI backend
├── dashboard/
## 🐳 Docker
│
Docker support has been removed from this repository. To run the project locally, follow the "Quick Start (Local)" instructions above using Python and Streamlit, or use the `start.sh` script.
├── utils/
#
├── requirements.txt
├── start.sh
└── README.md
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9+
- pip

### Step 1: Clone and Install

```bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive_maintenance

pip install -r requirements.txt
```

### Step 2: Generate Sample Data (Optional)

```bash
python utils/generate_sample_data.py
```

### Step 3: Start the Backend

```bash
# From project root (development)
python -u src/api/app.py

# For production (use gunicorn)
gunicorn src.api.app:app --bind 0.0.0.0:8000
```

Backend available at: http://localhost:8000
Run the Flask WSGI app locally; interactive Swagger docs are not provided for the Flask wrapper.

### Step 4: Start the Dashboard (new terminal)

```bash
streamlit run dashboard/app.py
```
Dashboard at: http://localhost:8501

### Step 5: Use the Platform

1. Open http://localhost:8501
2. Go to **📂 Upload & Train**
3. Download the sample CSV or upload your own
4. Click **▶ Train All Models**
5. Navigate to **🏠 Dashboard** to see predictions!

---




---

## ☁️ Cloud Deployment

### Deploy on Render

1. Push project to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `./start.sh`
5. Add environment variable: `API_URL=https://your-app.onrender.com`
### Deploy on AWS EC2

```bash
# 1. SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip
# 2. Install Docker
sudo yum install docker -y
sudo service docker start

# 3. Clone and run
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive_maintenance
docker-compose up --build -d

# Access at: http://your-ec2-ip:8501
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check & training status |
| POST | `/upload_data` | Upload CSV, returns schema & preview |
| POST | `/train_model` | Train all ML models |
| GET | `/predict` | Get failure predictions (XGB + LSTM) |
| GET | `/get_rul` | Get RUL estimates |
| GET | `/anomaly_score` | Get anomaly scores |
| GET | `/full_report` | Complete business intelligence report |
| GET | `/feature_importance` | Top SHAP feature importances |
| GET | `/sensor_series` | Raw sensor time series for plotting |

Interactive Swagger docs: http://localhost:8000/docs

---

## 📄 Sample Data Format

```csv
timestamp,machine_id,temperature,vibration,pressure,rpm
2024-01-01 00:00:00,MACHINE-001,74.5,0.048,101.2,1498
2024-01-01 01:00:00,MACHINE-001,75.1,0.051,100.8,1502
2024-01-01 02:00:00,MACHINE-001,74.8,0.049,101.5,1497
...
```

**Minimum requirements:**
- `timestamp` column (any parseable datetime)
- At least 1 numeric sensor column
- At least 20 rows recommended (500+ for best results)

---

## 🔮 Future Improvements

- [ ] **Real-time streaming** — Kafka/MQTT integration for live sensor data
- [ ] **Multi-machine dashboard** — Monitor fleet of machines simultaneously
- [ ] **Alert notifications** — Email/Slack alerts when risk exceeds threshold
- [ ] **Model retraining** — Auto-retrain when new labeled data arrives
- [ ] **Time-series Transformer** — Replace LSTM with Temporal Fusion Transformer
- [ ] **Database persistence** — PostgreSQL/InfluxDB for historical data
- [ ] **Authentication** — JWT-based user management
- [ ] **Export reports** — PDF report generation
- [ ] **Confidence intervals** — Uncertainty quantification on predictions
- [ ] **Digital twin** — 3D visualization of machine state

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit + Plotly |
| Backend | Flask + Gunicorn |
| Deep Learning | PyTorch (LSTM) |
| ML | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Containerization | Docker, Docker Compose |

---

## 📝 License

MIT License — free to use for personal and commercial projects.

---

*Built as a portfolio project demonstrating end-to-end MLOps for industrial AI applications.*
