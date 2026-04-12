"""
ml/generate_demo.py

Generates a small, separate demo dataset of ~200 orders that is served
by dummy_customer_extended_api.py when the "Extended dataset" toggle is on.

This dataset is COMPLETELY SEPARATE from the training data in
extended_orders.json. The anomaly detection model was never trained on
these orders, so the model's scoring behaviour on them is a genuine
prediction — not a memory of training data.

Order IDs start at 3000 to avoid any collision with the training dataset
(IDs 1001-1005 real, 2000-2999 synthetic training orders).

Target: ~200 orders, ~10% out-of-policy (same ratio as training set).

Run once after training the model:
    python -m ml.generate_demo

Output:
    ml/data/demo_orders.json  -- raw order strings served by the extended API
"""

import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv

from ml.data_generator import (
    _build_llm,
    _generate_batch,
    _validate_raw_string,
    _extract_order_id,
    NORMAL_PROMPT,
    ANOMALY_PROMPT,
    REAL_ORDER_STRINGS,
    DATA_DIR,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEMO_PATH = DATA_DIR / "demo_orders.json"

# IDs start at 3000 — clear of training data range (2000-2999)
DEMO_START_ID = 3000


def generate_demo(
    normal_batches: int = 3,
    anomaly_batches: int = 1,
    batch_size: int = 50,
) -> list[str]:
    """
    Generate a small demo dataset of raw order strings.

    Uses the same prompts and LLM as the main generator but saves
    to demo_orders.json and uses IDs starting at 3000 to avoid
    any overlap with the training dataset.

    Args:
        normal_batches  -- number of normal order batches
        anomaly_batches -- number of anomaly batches
        batch_size      -- orders per LLM call

    Returns:
        List of raw order strings (mix of normal and out-of-policy)
    """
    llm = _build_llm()
    all_strings: list[str] = []
    seen_ids: set[str] = set()
    total_discarded = 0

    # Seed with real orders so format reference is consistent
    for raw in REAL_ORDER_STRINGS:
        order_id = _extract_order_id(raw)
        if order_id:
            seen_ids.add(order_id)
        all_strings.append(raw)

    next_id = DEMO_START_ID

    # Normal batches
    for i in range(normal_batches):
        prompt = NORMAL_PROMPT.format(count=batch_size, start_id=next_id)
        raw_lines = _generate_batch(llm, prompt, i + 1, is_anomaly=False)

        for raw in raw_lines:
            if _validate_raw_string(raw, seen_ids):
                order_id = _extract_order_id(raw)
                seen_ids.add(order_id)
                all_strings.append(raw)
            else:
                total_discarded += 1

        next_id += batch_size
        if i < normal_batches - 1:
            time.sleep(1)

    # Anomaly batches
    for i in range(anomaly_batches):
        prompt = ANOMALY_PROMPT.format(count=batch_size, start_id=next_id)
        raw_lines = _generate_batch(llm, prompt, i + 1, is_anomaly=True)

        for raw in raw_lines:
            if _validate_raw_string(raw, seen_ids):
                order_id = _extract_order_id(raw)
                seen_ids.add(order_id)
                all_strings.append(raw)
            else:
                total_discarded += 1

        next_id += batch_size
        if i < anomaly_batches - 1:
            time.sleep(1)

    logger.info("Demo generation complete:")
    logger.info(f"  Total orders: {len(all_strings)}")
    logger.info(f"  Discarded:    {total_discarded}")
    logger.info(f"  ID range:     {DEMO_START_ID} – {next_id - 1}")

    return all_strings


if __name__ == "__main__":
    logger.info("Generating demo dataset (~200 orders, separate from training data)...")
    logger.info("~4 LLM calls. Estimated time: 1-2 minutes.")

    strings = generate_demo(
        normal_batches=3,
        anomaly_batches=1,
        batch_size=50,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_PATH, "w") as f:
        json.dump(strings, f, indent=2)

    logger.info(f"Saved {len(strings)} demo orders to {DEMO_PATH}")
    logger.info("Done. Run python main.py --extended to use the demo dataset.")
