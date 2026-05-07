#!/usr/bin/env bash
set -euo pipefail
# start.sh — Starts both FastAPI backend and Streamlit frontend

echo "🚀 Starting Predictive Maintenance Platform..."

# Start FastAPI backend in background
echo "⚙️  Starting Flask backend (gunicorn) on port 8000..."
gunicorn src.api.app:app --bind 0.0.0.0:8000 &
BACKEND_PID=$!

# Give backend a moment to initialize
sleep 2

echo "🖥️  Starting Streamlit dashboard on port 8501..."
streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --theme.base dark \
    --theme.primaryColor "#00d4ff" \
    --theme.backgroundColor "#0a0e1a" \
    --theme.secondaryBackgroundColor "#111827" \
    --theme.textColor "#e8eaf6"

# When Streamlit exits, kill backend (if still running)
if ps -p $BACKEND_PID > /dev/null 2>&1; then
    kill $BACKEND_PID || true
fi
