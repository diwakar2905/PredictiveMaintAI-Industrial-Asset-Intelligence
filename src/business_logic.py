"""
business_logic.py
-----------------
Converts model outputs into actionable business intelligence:
- Risk score (0-100)
- Risk category (Low / Medium / High / Critical)
- Alerts and recommendations
- Estimated downtime cost
"""

import numpy as np
from typing import Dict, Any


# Cost configuration (easily adjustable per industry)
COST_CONFIG = {
    "hourly_downtime_cost": 5000,     # USD per hour of unplanned downtime
    "planned_maintenance_cost": 800,   # USD for planned maintenance
    "emergency_repair_cost": 15000,    # USD for emergency repair
    "production_loss_per_hour": 2000,  # USD
}


def compute_risk_score(
    failure_prob: float,
    anomaly_score: float,
    rul_hours: float,
    max_rul_hours: float = 125.0
) -> int:
    """
    Computes a composite risk score from 0 to 100.

    Formula:
        risk = 40% failure probability
             + 30% anomaly score
             + 30% (1 - normalized RUL)

    Returns:
        Integer risk score in [0, 100]
    """
    # Normalize RUL: low RUL = high risk
    rul_norm = min(rul_hours / max_rul_hours, 1.0)
    rul_risk = 1.0 - rul_norm

    risk = (
        0.40 * float(failure_prob) +
        0.30 * float(anomaly_score) +
        0.30 * rul_risk
    )

    return int(np.clip(risk * 100, 0, 100))


def get_risk_category(risk_score: int) -> str:
    """
    Maps risk score to a category label.

    0-25   → Low
    26-50  → Medium
    51-75  → High
    76-100 → Critical
    """
    if risk_score <= 25:
        return "Low"
    elif risk_score <= 50:
        return "Medium"
    elif risk_score <= 75:
        return "High"
    else:
        return "Critical"


def generate_alerts(
    risk_score: int,
    failure_prob: float,
    rul_hours: float,
    anomaly_score: float,
    anomaly_is_active: bool
) -> list:
    """
    Generates a list of alert messages based on current machine state.
    """
    alerts = []

    if failure_prob >= 0.8:
        alerts.append({
            "level": "CRITICAL",
            "message": f"⚠️ Failure likely within next 48 hours! (prob: {failure_prob:.1%})",
            "action": "Schedule emergency maintenance immediately."
        })
    elif failure_prob >= 0.5:
        alerts.append({
            "level": "WARNING",
            "message": f"⚡ Elevated failure probability detected ({failure_prob:.1%})",
            "action": "Schedule maintenance within 72 hours."
        })

    if rul_hours <= 24:
        alerts.append({
            "level": "CRITICAL",
            "message": f"🕐 Machine has only {rul_hours:.0f} hours of useful life remaining.",
            "action": "Immediate inspection required."
        })
    elif rul_hours <= 72:
        alerts.append({
            "level": "WARNING",
            "message": f"🕐 Remaining useful life: {rul_hours:.0f} hours.",
            "action": "Plan maintenance within 3 days."
        })

    if anomaly_is_active and anomaly_score >= 0.7:
        alerts.append({
            "level": "WARNING",
            "message": f"🔍 Active anomaly detected (score: {anomaly_score:.2f}).",
            "action": "Inspect sensor readings and mechanical components."
        })

    if not alerts:
        alerts.append({
            "level": "OK",
            "message": "✅ Machine operating within normal parameters.",
            "action": "Continue scheduled monitoring."
        })

    return alerts


def estimate_downtime_cost(rul_hours: float, failure_prob: float) -> Dict[str, Any]:
    """
    Estimates financial impact using a simple cost model.

    Expected cost = P(failure) * (downtime_hours * hourly_cost + emergency_cost)
                  vs.
    Planned maintenance cost (much lower)

    Returns:
        dict with cost estimates and savings
    """
    cfg = COST_CONFIG

    # Expected downtime hours: roughly 2x RUL depletion pace
    expected_downtime_hours = max(2, min(rul_hours * 0.1, 48))

    # Unplanned failure cost
    failure_cost = (
        expected_downtime_hours * cfg["hourly_downtime_cost"] +
        expected_downtime_hours * cfg["production_loss_per_hour"] +
        cfg["emergency_repair_cost"]
    )

    # Expected cost considering probability
    expected_failure_cost = failure_prob * failure_cost

    # Planned maintenance is always cheaper
    planned_cost = cfg["planned_maintenance_cost"]

    savings = expected_failure_cost - planned_cost

    return {
        "expected_failure_cost_usd": round(expected_failure_cost, 0),
        "planned_maintenance_cost_usd": planned_cost,
        "potential_savings_usd": round(max(0, savings), 0),
        "recommendation": (
            "Schedule planned maintenance NOW to save costs."
            if savings > 0 else
            "Machine appears healthy. Continue monitoring."
        )
    }


def generate_business_report(
    failure_prob: float,
    anomaly_score: float,
    rul_hours: float,
    anomaly_is_active: bool,
    failure_reason: str,
    machine_id: str = "MACHINE-001"
) -> Dict[str, Any]:
    """
    Generates a complete business intelligence report.

    Returns:
        Complete dict with all KPIs, alerts, and recommendations.
    """
    risk_score = compute_risk_score(failure_prob, anomaly_score, rul_hours)
    risk_category = get_risk_category(risk_score)
    alerts = generate_alerts(risk_score, failure_prob, rul_hours, anomaly_score, anomaly_is_active)
    cost_estimate = estimate_downtime_cost(rul_hours, failure_prob)

    return {
        "machine_id": machine_id,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "failure_probability": round(float(failure_prob), 4),
        "rul_hours": round(float(rul_hours), 1),
        "anomaly_score": round(float(anomaly_score), 4),
        "anomaly_active": anomaly_is_active,
        "failure_reason": failure_reason,
        "alerts": alerts,
        "cost_estimate": cost_estimate,
    }
