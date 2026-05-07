"""
generate_sample_data.py
-----------------------
Generates realistic synthetic sensor data for testing the platform.
Run: python utils/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_sensor_data(
    n_samples: int = 500,
    n_sensors: int = 4,
    machine_id: str = "MACHINE-001",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates synthetic time-series sensor data with a realistic
    degradation pattern: normal → slight drift → failure onset.
    """
    np.random.seed(seed)

    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]

    # Phase boundaries
    normal_end = int(n_samples * 0.65)
    drift_end = int(n_samples * 0.85)
    # failure_end = n_samples (last 15%)

    data = {"timestamp": timestamps, "machine_id": machine_id}

    sensor_configs = [
        {"name": "temperature",  "base": 75.0,  "noise": 2.0,  "drift_rate": 0.08, "fail_rate": 0.25},
        {"name": "vibration",    "base": 0.05,  "noise": 0.01, "drift_rate": 0.003,"fail_rate": 0.015},
        {"name": "pressure",     "base": 101.3, "noise": 1.5,  "drift_rate": 0.05, "fail_rate": 0.20},
        {"name": "rpm",          "base": 1500,  "noise": 20.0, "drift_rate": -2.0, "fail_rate": -8.0},
    ]

    for cfg in sensor_configs[:n_sensors]:
        values = []
        for i in range(n_samples):
            base = cfg["base"]
            noise = np.random.normal(0, cfg["noise"])

            if i < normal_end:
                # Normal operation with small random noise
                val = base + noise

            elif i < drift_end:
                # Gradual drift/degradation
                drift_steps = i - normal_end
                val = base + noise + cfg["drift_rate"] * drift_steps

            else:
                # Failure zone: accelerating degradation + spikes
                drift_steps = drift_end - normal_end
                fail_steps = i - drift_end
                spike = np.random.normal(0, cfg["noise"] * 3)  # More volatile
                val = (base +
                       cfg["drift_rate"] * drift_steps +
                       cfg["fail_rate"] * fail_steps +
                       spike)

            values.append(round(val, 4))

        data[cfg["name"]] = values

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_sensor_data(n_samples=500)
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sensor_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Sample data saved to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(df.head())
