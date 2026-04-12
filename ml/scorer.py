"""
ml/scorer.py

Runtime anomaly scoring layer — the only ML module imported by the agent.

Loads the trained Isolation Forest and StandardScaler on first call and
caches them for the process lifetime. Scores each Order object by
transforming its features into the same space the model was trained in,
computing the Isolation Forest decision score, and converting it to a
0-1 anomaly score.

Anomaly scoring is best-effort: if the model file is missing or scoring
fails for any reason, orders are returned without anomaly data rather
than failing the entire response.

Functions:
    score_order(order)   -- score a single Order, returns enrichment dict
    score_orders(orders) -- score a list of Orders, returns list of dicts
"""

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from models.schemas import Order
from ml.features import categorize_items, extract_features, ANOMALY_THRESHOLD
from ml.model_store import load_model

logger = logging.getLogger(__name__)

# ── Module-level model cache ───────────────────────────────────────────────────

_iso_forest: Optional[IsolationForest] = None
_scaler:     Optional[StandardScaler]  = None


def _ensure_loaded() -> None:
    """Load models into module-level cache if not already loaded."""
    global _iso_forest, _scaler
    if _iso_forest is None:
        _iso_forest, _scaler = load_model()


# ── Runtime scoring ────────────────────────────────────────────────────────────

def score_order(order: Order) -> dict:
    """
    Score a single Order object for anomalies.

    Loads models on first call and caches for the process lifetime.
    Returns a structured dict with the anomaly score, flag, category,
    actual total, and a human-readable explanation grounded in the
    Isolation Forest score and threshold.

    Args:
        order -- validated Order object from the agent pipeline

    Returns:
        dict with keys:
            anomaly_score -- float 0-1, higher = more suspicious
            is_flagged    -- bool, True if score exceeds threshold
            category      -- detected item category string
            actual_total  -- order.total
            reason        -- human-readable explanation
    """
    try:
        _ensure_loaded()

        category = categorize_items(order.items or [])

        X, _     = extract_features([order])
        X_scaled = _scaler.transform(X)

        raw_score     = _iso_forest.decision_function(X_scaled)[0]
        # decision_function returns ~[-0.5, +0.5]: positive=normal, negative=anomalous.
        # Invert and shift to [0,1] so 1.0 = most anomalous. Clip handles rare out-of-range values.
        anomaly_score = float(np.clip(1 - (raw_score + 0.5), 0, 1))
        is_flagged    = anomaly_score > ANOMALY_THRESHOLD

        if is_flagged:
            reason = (
                f"Anomaly score {anomaly_score:.3f} exceeds threshold "
                f"{ANOMALY_THRESHOLD} for {category.replace('_', ' ')} — "
                f"order total of ${order.total:,.2f} deviates significantly "
                f"from the normal distribution learned during training."
            )
        else:
            reason = (
                f"Anomaly score {anomaly_score:.3f} is within normal range "
                f"(threshold {ANOMALY_THRESHOLD}) for "
                f"{category.replace('_', ' ')} — "
                f"order total of ${order.total:,.2f} is consistent with "
                f"training data."
            )

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_flagged":    is_flagged,
            "category":      category,
            "actual_total":  order.total,
            "reason":        reason,
        }

    except FileNotFoundError:
        return {
            "anomaly_score": 0.0,
            "is_flagged":    False,
            "category":      categorize_items(order.items or []),
            "actual_total":  order.total,
            "reason":        "Anomaly model not trained. Run python -m ml.trainer.",
        }
    except Exception as e:
        logger.error(f"Scoring failed for order {order.orderId}: {e}")
        return {
            "anomaly_score": 0.0,
            "is_flagged":    False,
            "category":      "unknown",
            "actual_total":  order.total,
            "reason":        f"Scoring error: {e}",
        }


def score_orders(orders: list[Order]) -> list[dict]:
    """
    Score a list of Order objects for anomalies.

    Called by the agent output node to enrich every query result
    with fraud intelligence before returning to the user.

    Args:
        orders -- list of validated Order objects

    Returns:
        List of anomaly score dicts in same order as input
    """
    return [score_order(order) for order in orders]
