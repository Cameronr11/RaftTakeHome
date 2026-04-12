"""
ml/features.py

Feature engineering layer for the anomaly detector.

Defines the item category taxonomy, the ANOMALY_THRESHOLD, and the
two functions that convert Order objects into numeric feature vectors
for the Isolation Forest model.

This module has no dependency on training, persistence, or scoring —
it is pure data transformation and is safe to import anywhere.

Constants:
    CATEGORY_KEYWORDS -- mapping of category name to keyword list
    CATEGORIES        -- sorted list of all categories including 'other'
    ANOMALY_THRESHOLD -- score above which an order is flagged

Functions:
    categorize_items(items)    -- map item list to a category string
    extract_features(orders)   -- build feature matrix from Order objects
"""

import numpy as np
import pandas as pd

from models.schemas import Order

# ── Item category definitions ──────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "tech_equipment":  ["laptop", "desktop", "monitor", "server", "computer", "pc", "gaming"],
    "peripherals":     ["mouse", "keyboard", "webcam", "headphones", "hdmi", "cable", "usb", "hub", "microphone"],
    "networking":      ["router", "switch", "ethernet", "firewall", "modem"],
    "audio_visual":    ["projector", "speaker", "display", "screen", "camera"],
    "office_supplies": ["desk", "chair", "lamp", "whiteboard", "furniture", "stand"],
    "mobile_devices":  ["tablet", "smartphone", "phone", "smartwatch"],
}

# Sorted list of all categories including "other" — used for one-hot encoding
# in extract_features(). Derived from CATEGORY_KEYWORDS so adding a new
# keyword group automatically extends the feature space.
CATEGORIES = sorted([*CATEGORY_KEYWORDS.keys(), "other"])

# Precision-over-recall threshold: only flag orders the model is confident
# are anomalous, reducing false positives in analyst review queues.
ANOMALY_THRESHOLD = 0.60


# ── Feature engineering ────────────────────────────────────────────────────────

def categorize_items(items: list[str]) -> str:
    """
    Map a list of item strings to one of our procurement categories.

    Scans each item against keyword lists and returns the first
    matching category. Returns 'other' if no keywords match.

    Args:
        items -- list of item name strings from an Order object

    Returns:
        Category string e.g. 'peripherals', 'tech_equipment', 'other'
    """
    if not items:
        return "other"

    items_lower = " ".join(items).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in items_lower for kw in keywords):
            return category

    return "other"


def extract_features(orders: list[Order]) -> tuple[np.ndarray, list[str]]:
    """
    Engineer features from a list of Order objects for model input.

    Features:
        total     -- raw order total in dollars
        num_items -- number of items in the order
        cat_*     -- one-hot encoded category (7 binary columns)

    price_ratio is deliberately excluded. The model must learn price
    distributions from raw totals and category context alone — not
    from a ratio derived from the same thresholds used to generate
    the synthetic anomalies.

    Args:
        orders -- list of validated Order objects

    Returns:
        Tuple of (feature_matrix as np.ndarray, feature_names as list)
    """
    rows = []

    for order in orders:
        category  = categorize_items(order.items or [])
        num_items = len(order.items) if order.items else 0

        cat_encoding = {f"cat_{c}": int(c == category) for c in CATEGORIES}

        row = {
            "total":     order.total,
            "num_items": num_items,
            **cat_encoding,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    feature_names = list(df.columns)
    return df.values.astype(float), feature_names
