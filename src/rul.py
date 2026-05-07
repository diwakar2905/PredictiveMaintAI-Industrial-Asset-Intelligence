"""
rul.py
------
Remaining Useful Life (RUL) estimation using regression.
Predicts how many timesteps remain before machine failure.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def create_rul_labels(n_samples: int) -> np.ndarray:
    """
    Creates synthetic RUL labels (countdown from max to 0).
    In real scenarios these come from domain knowledge or historical failure data.

    Example: If machine runs for 200 timesteps before failure,
    RUL at step 0 = 200, at step 100 = 100, at step 200 = 0

    We apply an exponential decay pattern to simulate real-world degradation.
    """
    # Linear countdown (simple and interpretable)
    rul = np.linspace(n_samples, 1, n_samples)

    # Clip at max_rul (e.g., 125 hours as ceiling — common in literature)
    max_rul = min(125, n_samples)
    rul = np.clip(rul, 0, max_rul)

    return rul.astype(float)


def train_rul_model(X: np.ndarray, y: np.ndarray) -> Ridge:
    """
    Trains a Ridge regression model to predict RUL.

    Args:
        X: feature array
        y: RUL labels

    Returns:
        fitted Ridge model
    """
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    train_preds = model.predict(X)
    mae = mean_absolute_error(y, train_preds)
    logger.info(f"RUL model trained. Train MAE: {mae:.2f} timesteps")

    return model


def predict_rul(model: Ridge, X: np.ndarray) -> np.ndarray:
    """
    Predicts RUL (clipped to non-negative values).

    Returns:
        np.ndarray of RUL values (timesteps remaining)
    """
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def rul_to_hours(rul_value: float, sampling_interval_minutes: int = 60) -> float:
    """
    Converts RUL timesteps to hours.

    Args:
        rul_value: predicted RUL in timesteps
        sampling_interval_minutes: how often data is sampled

    Returns:
        RUL in hours
    """
    return round(rul_value * sampling_interval_minutes / 60, 1)


def rul_summary(rul_values: np.ndarray) -> dict:
    """Returns summary statistics for RUL predictions."""
    latest_rul = float(rul_values[-1])
    return {
        "latest_rul_timesteps": round(latest_rul, 1),
        "latest_rul_hours": rul_to_hours(latest_rul),
        "mean_rul": round(float(rul_values.mean()), 1),
        "min_rul": round(float(rul_values.min()), 1),
    }
