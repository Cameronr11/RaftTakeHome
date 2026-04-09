"""
ml/anomaly_detector.py

Isolation Forest-based anomaly detector for procurement order fraud detection.

Uses a semi-supervised approach — the model trains exclusively on normal
orders so it learns a clean baseline of what legitimate procurement looks
like. Anomalies are detected as deviations from that baseline at inference
time. This avoids the common mistake of training on contaminated data and
prevents the model from learning to replicate anomalous patterns.

Data handling:
    - Loads pre-parsed orders from ml/parsed_orders.json — no LLM calls
      required at training time. Run ml/save_parsed_orders.py first.
    - Stratified 80/20 train/test split preserving the 90/10 class ratio
    - Model trained on normal orders only (semi-supervised)
    - Evaluation performed on held-out test set never seen during training
    - StandardScaler fitted on training data only, applied to test data
      to prevent distribution leakage across the split

Feature engineering:
    - total:      raw order total in dollars
    - num_items:  number of items in the order
    - cat_*:      one-hot encoded item category (7 binary columns)

    price_ratio is deliberately excluded as a model feature to prevent
    circular logic between how anomalies were generated and how they
    are detected. It appears only in human-readable score explanations.

Statistical model:
    A linear regression is also trained on the training split to predict
    expected order total from item category and item count. This satisfies
    the Raft spec requirement for a traditional statistical model and
    provides the expected_total baseline shown in anomaly explanations.

Usage:
    # Train (run once after save_parsed_orders.py)
    python -m ml.anomaly_detector

    # Score at runtime (called by agent output node)
    from ml.anomaly_detector import score_orders
    results = score_orders(filtered_orders)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.schemas import Order

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

ML_DIR       = Path(__file__).parent
PARSED_PATH  = ML_DIR / "parsed_orders.json"
LABELS_PATH  = ML_DIR / "anomaly_labels.json"
MODEL_PATH   = ML_DIR / "anomaly_model.pkl"

# ── Item category definitions ──────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "tech_equipment":  ["laptop", "desktop", "monitor", "server", "computer", "pc", "gaming"],
    "peripherals":     ["mouse", "keyboard", "webcam", "headphones", "hdmi", "cable", "usb", "hub", "microphone"],
    "networking":      ["router", "switch", "ethernet", "firewall", "modem"],
    "audio_visual":    ["projector", "speaker", "display", "screen", "camera"],
    "office_supplies": ["desk", "chair", "lamp", "whiteboard", "furniture", "stand"],
    "mobile_devices":  ["tablet", "smartphone", "phone", "smartwatch"],
}

# Expected price ceiling per category — used ONLY for human readable
# explanations in score_order(). Never used as a model feature.
CATEGORY_NORMAL_MAX: dict[str, float] = {
    "tech_equipment":  3000.0,
    "peripherals":     500.0,
    "networking":      2000.0,
    "audio_visual":    1500.0,
    "office_supplies": 800.0,
    "mobile_devices":  1500.0,
    "other":           2000.0,
}

CATEGORIES = sorted(CATEGORY_NORMAL_MAX.keys())

# Score threshold above which an order is flagged for review
ANOMALY_THRESHOLD = 0.6


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
        num_items = len(order.items) if order.items else 1

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


# ── Linear regression ──────────────────────────────────────────────────────────

def train_regression(orders_train: list[Order]) -> LinearRegression:
    """
    Train a linear regression on the training split predicting order
    total from item category and number of items.

    Trained on the same training split as the Isolation Forest to
    maintain the same data boundaries and prevent leakage.

    Satisfies the Raft spec requirement for a traditional statistical
    model. At inference time the regression provides an expected_total
    baseline shown in anomaly explanations.

    Args:
        orders_train -- training split Order objects

    Returns:
        Trained LinearRegression model
    """
    rows = []
    for order in orders_train:
        category  = categorize_items(order.items or [])
        num_items = len(order.items) if order.items else 1
        cat_encoding = {f"cat_{c}": int(c == category) for c in CATEGORIES}
        rows.append({
            "num_items": num_items,
            **cat_encoding,
            "total": order.total,
        })

    df = pd.DataFrame(rows)
    X  = df.drop("total", axis=1).values.astype(float)
    y  = df["total"].values

    reg = LinearRegression()
    reg.fit(X, y)

    logger.info(f"Linear regression R² on training split: {reg.score(X, y):.3f}")
    return reg


def predict_expected_total(
    order: Order,
    regression: LinearRegression,
) -> float:
    """
    Predict the expected total for an order using the trained regression.

    Used in score_order() to populate the expected_total explanation
    shown to investigators.

    Args:
        order      -- Order object to predict for
        regression -- trained LinearRegression model

    Returns:
        Predicted total as float, floored at 0
    """
    category  = categorize_items(order.items or [])
    num_items = len(order.items) if order.items else 1
    cat_encoding = [int(c == category) for c in CATEGORIES]
    X = np.array([[num_items, *cat_encoding]])
    return max(float(regression.predict(X)[0]), 0.0)


# ── Isolation Forest training ──────────────────────────────────────────────────

def train_isolation_forest(
    X_train_clean: np.ndarray,
) -> tuple[IsolationForest, StandardScaler]:
    """
    Train an Isolation Forest on NORMAL orders only.

    Semi-supervised approach: the model learns exclusively what normal
    procurement looks like. It never sees anomalous orders during
    training. At inference time any order that deviates significantly
    from the learned normal distribution receives a high anomaly score.

    The scaler is fitted on clean training data only. When applied
    to test data or runtime orders it transforms them into the same
    space the model was trained in without leaking test distribution
    information into the fit.

    contamination='auto' because we are training on normal orders
    only — there is no expected contamination in the training set.
    The decision threshold is set by ANOMALY_THRESHOLD at inference.

    Args:
        X_train_clean -- feature matrix of normal training orders only

    Returns:
        Tuple of (trained IsolationForest, StandardScaler fitted on
        clean training data only)
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_clean)

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        max_samples="auto",
    )
    model.fit(X_scaled)

    logger.info(
        f"Isolation Forest trained on {X_train_clean.shape[0]} "
        f"normal orders"
    )
    return model, scaler


# ── Model persistence ──────────────────────────────────────────────────────────

def save_model(
    iso_forest: IsolationForest,
    scaler: StandardScaler,
    regression: LinearRegression,
) -> None:
    """Save trained models to ml/anomaly_model.pkl."""
    payload = {
        "iso_forest": iso_forest,
        "scaler":     scaler,
        "regression": regression,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Models saved to {MODEL_PATH}")


def load_model() -> tuple[IsolationForest, StandardScaler, LinearRegression]:
    """
    Load trained models from ml/anomaly_model.pkl.

    Raises:
        FileNotFoundError if model has not been trained yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run python -m ml.anomaly_detector to train first."
        )
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    return (
        payload["iso_forest"],
        payload["scaler"],
        payload["regression"],
    )


# ── Runtime scoring ────────────────────────────────────────────────────────────

_iso_forest: Optional[IsolationForest] = None
_scaler:     Optional[StandardScaler]  = None
_regression: Optional[LinearRegression] = None


def _ensure_loaded() -> None:
    """Load models into module-level cache if not already loaded."""
    global _iso_forest, _scaler, _regression
    if _iso_forest is None:
        _iso_forest, _scaler, _regression = load_model()


def score_order(order: Order) -> dict:
    """
    Score a single Order object for anomalies.

    Loads models on first call and caches for the process lifetime.
    Returns a structured dict with the anomaly score, flag, category,
    regression baseline, and a human-readable explanation.

    price_ratio and CATEGORY_NORMAL_MAX appear here only to generate
    the reason string — they played no role in model training.

    Args:
        order -- validated Order object from the agent pipeline

    Returns:
        dict with keys:
            anomaly_score  -- float 0-1, higher = more suspicious
            is_flagged     -- bool, True if score exceeds threshold
            category       -- detected item category string
            expected_total -- regression-predicted baseline total
            actual_total   -- order.total
            reason         -- human-readable explanation
    """
    try:
        _ensure_loaded()

        category       = categorize_items(order.items or [])
        expected_total = predict_expected_total(order, _regression)
        expected_max   = CATEGORY_NORMAL_MAX.get(category, 2000.0)

        X, _     = extract_features([order])
        X_scaled = _scaler.transform(X)

        raw_score     = _iso_forest.decision_function(X_scaled)[0]
        anomaly_score = float(np.clip(1 - (raw_score + 0.5), 0, 1))
        is_flagged    = anomaly_score > ANOMALY_THRESHOLD

        if is_flagged:
            ratio  = order.total / expected_max
            reason = (
                f"Order total of ${order.total:,.2f} is {ratio:.1f}x "
                f"the expected maximum of ${expected_max:,.0f} for "
                f"{category.replace('_', ' ')}. "
                f"Regression baseline: ${expected_total:,.2f}."
            )
        else:
            reason = (
                f"Order total of ${order.total:,.2f} is within normal "
                f"range for {category.replace('_', ' ')}. "
                f"Regression baseline: ${expected_total:,.2f}."
            )

        return {
            "anomaly_score":  round(anomaly_score, 3),
            "is_flagged":     is_flagged,
            "category":       category,
            "expected_total": round(expected_total, 2),
            "actual_total":   order.total,
            "reason":         reason,
        }

    except FileNotFoundError:
        return {
            "anomaly_score":  0.0,
            "is_flagged":     False,
            "category":       categorize_items(order.items or []),
            "expected_total": 0.0,
            "actual_total":   order.total,
            "reason":         "Anomaly model not trained. Run python -m ml.anomaly_detector.",
        }
    except Exception as e:
        logger.error(f"Scoring failed for order {order.orderId}: {e}")
        return {
            "anomaly_score":  0.0,
            "is_flagged":     False,
            "category":       "unknown",
            "expected_total": 0.0,
            "actual_total":   order.total,
            "reason":         f"Scoring error: {e}",
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


# ── Training pipeline ──────────────────────────────────────────────────────────

def train_and_evaluate() -> None:
    """
    Full training pipeline — no LLM calls required.

    Loads pre-parsed Order objects from parsed_orders.json which was
    produced by ml/save_parsed_orders.py. Training is entirely offline.

    Split strategy:
        - 80% training, 20% test
        - Stratified by anomaly label to preserve 90/10 class ratio
        - Isolation Forest trained on normal training orders only
        - Scaler fitted on normal training data only
        - Linear regression trained on full training split
        - All evaluation on held-out test set only — no data leakage
    """
    logger.info("Loading pre-parsed orders from parsed_orders.json...")

    if not PARSED_PATH.exists():
        raise FileNotFoundError(
            f"{PARSED_PATH} not found. "
            f"Run python -m ml.save_parsed_orders first."
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH} not found. "
            f"Run python ml/data_generator.py first."
        )

    with open(PARSED_PATH) as f:
        parsed_dicts = json.load(f)

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    # reconstruct Order objects from saved dicts
    orders = [Order(**d) for d in parsed_dicts]
    logger.info(f"Loaded {len(orders)} parsed orders")

    # build ground truth labels aligned to parsed orders
    y_true = np.array([
        1 if labels.get(o.orderId, False) else 0
        for o in orders
    ])

    normal_total  = int((y_true == 0).sum())
    anomaly_total = int((y_true == 1).sum())
    logger.info(
        f"Dataset: {normal_total} normal, {anomaly_total} anomalous "
        f"({anomaly_total/len(orders)*100:.1f}%)"
    )

    # extract features
    logger.info("Engineering features...")
    X, feature_names = extract_features(orders)
    logger.info(f"Feature matrix: {X.shape[0]} x {X.shape[1]}")
    logger.info(f"Features: {feature_names}")

    # stratified train/test split
    (
        X_train, X_test,
        orders_train, orders_test,
        y_train, y_test,
    ) = train_test_split(
        X, orders, y_true,
        test_size=0.20,
        random_state=42,
        stratify=y_true,
    )

    logger.info(
        f"Train: {len(orders_train)} orders "
        f"({int((y_train==0).sum())} normal, "
        f"{int((y_train==1).sum())} anomalous)"
    )
    logger.info(
        f"Test:  {len(orders_test)} orders "
        f"({int((y_test==0).sum())} normal, "
        f"{int((y_test==1).sum())} anomalous)"
    )

    # train linear regression on full training split
    logger.info("Training linear regression...")
    regression = train_regression(orders_train)

    # train Isolation Forest on normal training orders only
    normal_mask   = y_train == 0
    X_train_clean = X_train[normal_mask]
    logger.info(
        f"Training Isolation Forest on {X_train_clean.shape[0]} "
        f"normal orders only..."
    )
    iso_forest, scaler = train_isolation_forest(X_train_clean)

    # evaluate on held-out test set
    X_test_scaled  = scaler.transform(X_test)
    raw_scores     = iso_forest.decision_function(X_test_scaled)
    anomaly_scores = np.clip(1 - (raw_scores + 0.5), 0, 1)
    predictions    = (anomaly_scores > ANOMALY_THRESHOLD).astype(int)

    tp = int(((predictions == 1) & (y_test == 1)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())
    tn = int(((predictions == 0) & (y_test == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    # save models
    save_model(iso_forest, scaler, regression)

    # print report
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 60)
    print(f"Dataset:            {len(orders)} total orders")
    print(f"  Normal:           {normal_total}")
    print(f"  Anomalous:        {anomaly_total} ({anomaly_total/len(orders)*100:.1f}%)")
    print(f"\nSplit:              80/20 stratified train/test")
    print(f"  Train:            {len(orders_train)} orders")
    print(f"  Test:             {len(orders_test)} orders (held out)")
    print(f"\nTraining approach:  Semi-supervised")
    print(f"  IF trained on:    {X_train_clean.shape[0]} normal orders only")
    print(f"  Regression on:    {len(orders_train)} full training split")
    print(f"\nFeatures:           {feature_names}")
    print(f"  Note: price_ratio excluded — prevents circular logic")
    print(f"        between data generation and anomaly detection")
    print(f"\nTest set results (no data leakage):")
    print(f"  True positives:   {tp}  (anomalies correctly flagged)")
    print(f"  False positives:  {fp}  (normal orders wrongly flagged)")
    print(f"  True negatives:   {tn}  (normal orders correctly cleared)")
    print(f"  False negatives:  {fn}  (anomalies missed)")
    print(f"\nMetrics:")
    print(f"  Precision:        {precision:.3f}")
    print(f"  Recall:           {recall:.3f}")
    print(f"  F1 Score:         {f1:.3f}")
    print(f"\nCaveat: metrics reflect synthetic data performance.")
    print(f"Real-world deployment requires production data and")
    print(f"threshold tuning with domain experts.")
    print(f"\nTop flagged orders in test set:")

    flagged_idx = np.where(predictions == 1)[0]
    if len(flagged_idx) > 0:
        top_idx     = flagged_idx[
            np.argsort(anomaly_scores[flagged_idx])[::-1]
        ][:5]
        test_orders = list(orders_test)

        for idx in top_idx:
            o       = test_orders[idx]
            cat     = categorize_items(o.items or [])
            exp_max = CATEGORY_NORMAL_MAX.get(cat, 2000.0)
            ratio   = o.total / exp_max
            score   = anomaly_scores[idx]
            known   = labels.get(o.orderId, False)
            tag     = "KNOWN ANOMALY" if known else "flagged"
            print(
                f"  Order {o.orderId}: "
                f"{', '.join(o.items or [])} — "
                f"${o.total:,.2f} — "
                f"score: {score:.3f} — "
                f"{ratio:.1f}x expected — [{tag}]"
            )
    else:
        print("  No orders flagged in test set.")

    print("=" * 60)
    print(f"Model saved to {MODEL_PATH}")
    print("Next: integrate score_orders() into agent output node.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_and_evaluate()