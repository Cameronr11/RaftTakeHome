"""
ml/trainer.py

Offline training pipeline for the Isolation Forest anomaly detector.

This module is never imported at runtime by the agent. It exists solely
to train the model from pre-parsed order data and evaluate its performance
on a held-out test set. Run it once after ml/save_parsed_orders.py has
produced ml/data/parsed_orders.json.

Uses a semi-supervised approach — the Isolation Forest trains exclusively
on normal orders so it learns a clean baseline of what legitimate
procurement looks like. Anomalies are detected as deviations from that
baseline at inference time.

Data handling:
    - Loads pre-parsed orders from ml/data/parsed_orders.json
    - Stratified 80/20 train/test split preserving the 90/10 class ratio
    - Model trained on normal orders only
    - Evaluation on held-out test set — no data leakage
    - StandardScaler fitted on training data only

Usage:
    python -m ml.trainer

Functions:
    train_isolation_forest(X_train_clean) -- fit IF + scaler on normal orders
    train_and_evaluate()                  -- full pipeline: load, split, train, eval, save
"""

import json
import logging

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.schemas import Order
from ml.features import extract_features, ANOMALY_THRESHOLD
from ml.model_store import save_model, MODEL_PATH, PARSED_PATH, LABELS_PATH

logger = logging.getLogger(__name__)


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
    save_model(iso_forest, scaler)

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
            o     = test_orders[idx]
            score = anomaly_scores[idx]
            known = labels.get(o.orderId, False)
            tag   = "KNOWN ANOMALY" if known else "flagged"
            print(
                f"  Order {o.orderId}: "
                f"{', '.join(o.items or [])} — "
                f"${o.total:,.2f} — "
                f"score: {score:.3f} — [{tag}]"
            )
    else:
        print("  No orders flagged in test set.")

    print("=" * 60)
    print(f"Model saved to {MODEL_PATH}")
    print("Next: run python main.py to test anomaly scoring.")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_and_evaluate()
