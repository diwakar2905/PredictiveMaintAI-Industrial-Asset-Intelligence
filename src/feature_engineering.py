"""
feature_engineering.py
-----------------------
Generates lag features, rolling statistics, and trend features
from normalized time-series sensor data.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_lag_features(df: pd.DataFrame, sensor_cols: list, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Creates lag features for each sensor column.
    Lag features capture recent historical values.

    Example: temperature_lag_1 = temperature shifted by 1 timestep
    """
    df = df.copy()
    for col in sensor_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, sensor_cols: list, windows: list = [3, 5]) -> pd.DataFrame:
    """
    Creates rolling mean and standard deviation features.
    These capture short-term trends and volatility.

    Example: temperature_roll_mean_5 = average over last 5 timesteps
    """
    df = df.copy()
    for col in sensor_cols:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0)
    return df


def create_trend_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Creates trend features: difference between current and previous value.
    Positive = rising, Negative = falling.

    Example: temperature_trend = temperature - temperature_lag_1
    """
    df = df.copy()
    for col in sensor_cols:
        df[f"{col}_trend"] = df[col].diff().fillna(0)
    return df


def engineer_features(df: pd.DataFrame, sensor_cols: list) -> tuple:
    """
    Full feature engineering pipeline.
    Applies: lag features, rolling stats, trend features.

    Returns:
        (feature_df, feature_columns_list)
    """
    df = create_lag_features(df, sensor_cols)
    df = create_rolling_features(df, sensor_cols)
    df = create_trend_features(df, sensor_cols)

    # Drop rows with NaN from lag features
    df = df.dropna().reset_index(drop=True)

    # All engineered feature columns (exclude timestamp, machine_id)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "machine_id"]]

    logger.info(f"Feature engineering complete. Total features: {len(feature_cols)}")
    return df, feature_cols


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10):
    """
    Converts flat feature arrays into sequences for LSTM input.

    Args:
        X: feature array (n_samples, n_features)
        y: label array (n_samples,)
        seq_len: number of timesteps per sequence

    Returns:
        (X_seq, y_seq) with shapes (n_seq, seq_len, n_features) and (n_seq,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)
