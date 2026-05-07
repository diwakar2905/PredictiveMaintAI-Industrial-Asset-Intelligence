"""
model_xgb.py
------------
XGBoost baseline model for failure prediction and SHAP explainability.
"""

import numpy as np
import pandas as pd
try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False
import logging
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Trains an XGBoost classifier for failure prediction.

    Args:
        X_train: feature array
        y_train: binary labels

    Returns:
        trained XGBClassifier
    """
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


def predict_xgb(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Returns failure probabilities from XGBoost."""
    return model.predict_proba(X)[:, 1]


def get_shap_values(model: XGBClassifier, X: np.ndarray, feature_names: list) -> dict:
    """
    Computes SHAP feature importance values.

    Returns:
        dict with 'mean_abs_shap' (Series), 'top_features' (list of top 10)
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Mean absolute SHAP across all samples
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_series = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

        top_features = importance_series.head(10).to_dict()
        logger.info(f"SHAP computed. Top feature: {importance_series.index[0]}")

        return {
            "mean_abs_shap": importance_series,
            "top_features": top_features
        }
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        # Fallback to built-in feature importance
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return {
            "mean_abs_shap": fi,
            "top_features": fi.head(10).to_dict()
        }


def get_failure_reason(top_features: dict) -> str:
    """
    Generates a human-readable explanation of why failure is predicted.

    Args:
        top_features: dict of {feature_name: importance_score}

    Returns:
        Explanation string
    """
    if not top_features:
        return "Insufficient data to determine failure reason."

    top = list(top_features.keys())[:3]
    reasons = []

    for feat in top:
        name_lower = feat.lower()
        if "temp" in name_lower:
            reasons.append("elevated temperature readings")
        elif "vib" in name_lower:
            reasons.append("abnormal vibration levels")
        elif "press" in name_lower:
            reasons.append("irregular pressure fluctuations")
        elif "trend" in name_lower:
            reasons.append(f"degrading trend in {feat.replace('_trend','')}")
        elif "roll" in name_lower:
            reasons.append(f"volatile rolling pattern in {feat.split('_roll')[0]}")
        else:
            reasons.append(f"anomalous behavior in {feat}")

    if reasons:
        return "Predicted failure due to: " + ", ".join(reasons[:2]) + "."
    return f"Predicted failure driven by {top[0]}."
