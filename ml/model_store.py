"""
ml/model_store.py

Model persistence layer for the anomaly detector.

Handles saving and loading the trained Isolation Forest and StandardScaler
to/from disk as a single pickle payload. All file I/O for the ML model
is isolated here so nothing else in the system needs to know the storage
format or file path.

Constants:
    MODEL_PATH -- path to the trained model pickle file

Functions:
    save_model(iso_forest, scaler) -- serialize models to disk
    load_model()                   -- deserialize models from disk
"""

import logging
import pickle
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

ML_DIR      = Path(__file__).parent
DATA_DIR    = ML_DIR / "data"
PARSED_PATH = DATA_DIR / "parsed_orders.json"
LABELS_PATH = DATA_DIR / "anomaly_labels.json"
MODEL_PATH  = DATA_DIR / "anomaly_model.pkl"


# ── Model persistence ──────────────────────────────────────────────────────────

def save_model(
    iso_forest: IsolationForest,
    scaler: StandardScaler,
) -> None:
    """Save trained models to ml/data/anomaly_model.pkl."""
    payload = {
        "iso_forest": iso_forest,
        "scaler":     scaler,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Models saved to {MODEL_PATH}")


def load_model() -> tuple[IsolationForest, StandardScaler]:
    """
    Load trained models from ml/data/anomaly_model.pkl.

    Raises:
        FileNotFoundError if model has not been trained yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run python -m ml.trainer to train first."
        )
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    return (
        payload["iso_forest"],
        payload["scaler"],
    )
