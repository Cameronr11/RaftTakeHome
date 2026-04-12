"""
ml/save_parsed_orders.py

One-time script that parses all orders from extended_orders.json
through the LLM parser and saves the results to parsed_orders.json.

Run once after data_generator.py:
    python -m ml.save_parsed_orders

The saved file is committed to the repo so anomaly_detector.py
can train without making any LLM calls.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ML_DIR         = Path(__file__).parent
DATA_DIR       = ML_DIR / "data"
ORDERS_PATH    = DATA_DIR / "extended_orders.json"
PARSED_PATH    = DATA_DIR / "parsed_orders.json"


def main():
    with open(ORDERS_PATH) as f:
        raw_strings = json.load(f)

    logger.info(f"Loaded {len(raw_strings)} raw order strings")
    logger.info("Parsing through LLM — this takes several minutes...")

    from services.parser import parse_orders
    orders, errors = parse_orders(raw_strings)

    logger.info(f"Parsed {len(orders)} orders, {len(errors)} failed")

    # serialize Order objects to dicts
    parsed_dicts = [o.model_dump(exclude={"raw"}) for o in orders]

    with open(PARSED_PATH, "w") as f:
        json.dump(parsed_dicts, f, indent=2)

    logger.info(f"Saved {len(parsed_dicts)} parsed orders to {PARSED_PATH}")
    logger.info("Now run: python -m ml.anomaly_detector")


if __name__ == "__main__":
    main()