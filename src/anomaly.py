"""
anomaly.py
----------
Anomaly detection using Isolation Forest.
Outputs anomaly scores for each data point.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


def train_isolation_forest(X: np.ndarray, contamination: float = 0.1) -> IsolationForest:
    """
    Trains Isolation Forest for anomaly detection.

    Args:
        X: feature array
        contamination: expected fraction of anomalies (0.1 = 10%)

    Returns:
        fitted IsolationForest model
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    logger.info("Isolation Forest trained.")
    return model


def get_anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Computes anomaly scores for each sample.

    Isolation Forest returns negative scores:
    - More negative = more anomalous
    We normalize to [0, 1] where 1 = most anomalous.

    Returns:
        np.ndarray of shape (n,) with scores in [0, 1]
    """
    raw_scores = model.decision_function(X)  # Negative = anomalous

    # Convert: flip sign so higher = more anomalous, then normalize
    flipped = -raw_scores
    min_s, max_s = flipped.min(), flipped.max()
    if max_s == min_s:
        normalized = np.zeros_like(flipped)
    else:
        normalized = (flipped - min_s) / (max_s - min_s)

    return normalized


def get_anomaly_labels(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Returns binary labels: 1 = anomaly, 0 = normal.
    """
    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    return (preds == -1).astype(int)


def anomaly_summary(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Returns summary statistics for anomaly detection results.
    """
    return {
        "total_points": len(scores),
        "anomaly_count": int(labels.sum()),
        "anomaly_pct": round(float(labels.mean()) * 100, 2),
        "mean_score": round(float(scores.mean()), 4),
        "max_score": round(float(scores.max()), 4),
        "latest_score": round(float(scores[-1]), 4),
        "latest_is_anomaly": bool(labels[-1] == 1)
    }
