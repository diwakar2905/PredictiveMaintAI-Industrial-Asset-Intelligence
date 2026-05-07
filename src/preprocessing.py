"""
preprocessing.py
----------------
Handles data ingestion, validation, cleaning, and normalization
for time-series sensor data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ["timestamp"]
OPTIONAL_COLUMNS = ["machine_id"]


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validates that the uploaded CSV has the minimum required columns.

    Returns:
        dict with 'valid' (bool), 'message' (str), 'sensor_cols' (list)
    """
    result = {"valid": True, "message": "", "sensor_cols": []}

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        result["valid"] = False
        result["message"] = f"Missing required columns: {missing}"
        return result

    # Identify sensor columns (everything except timestamp and machine_id)
    exclude = set(["timestamp", "machine_id"])
    sensor_cols = [c for c in df.columns if c not in exclude]

    if len(sensor_cols) == 0:
        result["valid"] = False
        result["message"] = "No sensor columns found. Please include numeric sensor data."
        return result

    result["sensor_cols"] = sensor_cols
    result["message"] = f"Schema valid. Found {len(sensor_cols)} sensor columns: {sensor_cols}"
    logger.info(result["message"])
    return result


def handle_missing_values(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    Handles missing values using forward fill, then backward fill,
    then column median as fallback.
    """
    df = df.copy()

    for col in sensor_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            # Final fallback: column median
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"Missing values handled for {len(sensor_cols)} sensor columns.")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parses and sorts by timestamp column."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_sensors(df: pd.DataFrame, sensor_cols: list, scaler=None):
    """
    Applies MinMax normalization to sensor columns.

    Args:
        df: DataFrame
        sensor_cols: columns to normalize
        scaler: optional pre-fitted scaler (for inference)

    Returns:
        (normalized_df, fitted_scaler)
    """
    df = df.copy()
    if scaler is None:
        scaler = MinMaxScaler()
        df[sensor_cols] = scaler.fit_transform(df[sensor_cols].values)
    else:
        df[sensor_cols] = scaler.transform(df[sensor_cols].values)
    return df, scaler


def preprocess_pipeline(df: pd.DataFrame, scaler=None):
    """
    Full preprocessing pipeline:
    1. Validate schema
    2. Parse timestamps
    3. Handle missing values
    4. Normalize sensors

    Returns:
        (processed_df, scaler, sensor_cols, validation_result)
    """
    validation = validate_schema(df)
    if not validation["valid"]:
        return None, None, None, validation

    sensor_cols = validation["sensor_cols"]
    df = parse_timestamps(df)
    df = handle_missing_values(df, sensor_cols)
    df, scaler = normalize_sensors(df, sensor_cols, scaler)

    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df, scaler, sensor_cols, validation
