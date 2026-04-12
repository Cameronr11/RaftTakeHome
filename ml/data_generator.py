"""
ml/data_generator.py

Generates synthetic order data in the same unstructured text format as
dummy_customer_api.py and saves it to ml/extended_orders.json.

The synthetic strings are intentionally formatted with slight variations
to simulate real-world unpredictable API schema changes — directly
addressing that edge case from the Raft spec.

Approximately 10% of generated orders are anomalous — normal items
with inflated prices that simulate procurement fraud patterns.

Run once to generate the dataset:
    python ml/data_generator.py

The output file is committed to the repo so the anomaly detector
trains on a stable consistent dataset. dummy_customer_api.py is
never modified — the extended dataset is served by
dummy_customer_api_extended.py.

Output:
    ml/extended_orders.json  -- list of raw order strings
    ml/anomaly_labels.json   -- maps orderId to is_anomaly boolean
                                (for model evaluation only, never
                                exposed through the API)
"""

import os
import re
import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Output paths ───────────────────────────────────────────────────────────────

ML_DIR            = Path(__file__).parent
DATA_DIR          = ML_DIR / "data"
ORDERS_PATH       = DATA_DIR / "extended_orders.json"
LABELS_PATH       = DATA_DIR / "anomaly_labels.json"

# ── LLM client ─────────────────────────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0.9,
    )

# ── Real orders from Raft (used for format reference and included in output) ───

REAL_ORDER_STRINGS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
    "Order 1004: Buyer=Rachel Kim, Location=Seattle, WA, Total=$89.50, Items: coffee maker",
    "Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp",
]

# ── Generation prompts ─────────────────────────────────────────────────────────

NORMAL_PROMPT = """Generate exactly {count} synthetic procurement order strings.

These strings simulate a real-world customer API that returns messy,
unstructured text. The format should be SIMILAR to these real examples
but with SLIGHT VARIATIONS to simulate unpredictable schema changes:

Real examples:
"Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable"
"Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones"
"Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse"

Use order IDs starting from {start_id}.

FORMAT VARIATIONS to use across orders (mix them up):
- Standard: "Order XXXX: Buyer=Name, Location=City, ST, Total=$X.XX, Items: item1, item2"
- Variation A: "Order XXXX | Buyer: Name | Location: City, ST | Total: $X.XX | Items: item1, item2"
- Variation B: "Order XXXX: Buyer=Name, Location=City, ST, Amount=$X.XX, Products: item1; item2"
- Variation C: "Order XXXX: Customer=Name, Location=City, ST, Total=$X.XX, Items: item1, item2"

ITEM CATEGORIES and NORMAL price ranges:
- tech equipment (laptop, desktop, monitor, server): $400 - $3000
- peripherals (mouse, keyboard, hdmi cable, webcam, headphones): $20 - $500
- networking (router, switch, ethernet cable, firewall): $100 - $2000
- audio visual (projector, speaker, display, microphone): $100 - $1500
- office supplies (desk lamp, chair, whiteboard, desk): $50 - $800
- mobile devices (tablet, smartphone, smartwatch): $200 - $1500

STATES and realistic cities:
OH: Columbus, Cleveland, Cincinnati, Dayton, Toledo
TX: Austin, Houston, Dallas, San Antonio, Fort Worth
CA: Los Angeles, San Diego, San Francisco, Sacramento, San Jose
VA: Arlington, Richmond, Norfolk, Roanoke, Alexandria
NC: Charlotte, Raleigh, Durham, Fayetteville, Greensboro
GA: Atlanta, Savannah, Augusta, Macon, Columbus
FL: Miami, Tampa, Orlando, Jacksonville, Pensacola
WA: Seattle, Tacoma, Spokane, Bellevue, Olympia
CO: Denver, Colorado Springs, Boulder, Aurora, Fort Collins
AZ: Phoenix, Tucson, Scottsdale, Tempe, Mesa
KY: Louisville, Lexington, Fort Campbell, Bowling Green
OK: Oklahoma City, Tulsa, Lawton, Norman

RULES:
- Each string on its own line
- Order IDs must be unique integers as strings
- Total must match item category price range
- Use realistic full names for buyers
- Vary states, cities, items, and formats across all orders
- Items list can have 1-3 items

Return ONLY the raw strings, one per line. No JSON, no numbering,
no explanation, no markdown."""

ANOMALY_PROMPT = """Generate exactly {count} ANOMALOUS procurement order strings.

These simulate procurement fraud — ordinary items with massively
inflated prices. The format should match normal orders exactly so
they are indistinguishable except for the suspicious total.

Use order IDs starting from {start_id}.

Use the STANDARD format only:
"Order XXXX: Buyer=Name, Location=City, ST, Total=$X.XX, Items: item1, item2"

FRAUD PATTERN — normal mundane items with prices 5-15x higher than normal:

Normal peripheral ($20-$500) → Anomalous ($2,000-$7,500)
Examples:
  "Order XXXX: Buyer=Name, Location=City, ST, Total=$4,250.00, Items: mouse"
  "Order XXXX: Buyer=Name, Location=City, ST, Total=$3,800.00, Items: keyboard, hdmi cable"

Normal office supply ($50-$800) → Anomalous ($3,000-$10,000)
Examples:
  "Order XXXX: Buyer=Name, Location=City, ST, Total=$6,500.00, Items: desk lamp"
  "Order XXXX: Buyer=Name, Location=City, ST, Total=$8,200.00, Items: whiteboard"

Normal audio visual ($100-$1,500) → Anomalous ($8,000-$20,000)
Examples:
  "Order XXXX: Buyer=Name, Location=City, ST, Total=$15,000.00, Items: microphone"
  "Order XXXX: Buyer=Name, Location=City, ST, Total=$12,500.00, Items: speaker"

RULES:
- Items must look completely ordinary — the fraud is in the price only
- Use realistic buyer names and real US cities
- Vary states across orders
- Each string on its own line

Return ONLY the raw strings, one per line. No JSON, no numbering,
no explanation, no markdown."""

# ── Validation ─────────────────────────────────────────────────────────────────

def _extract_order_id(raw: str) -> str | None:
    """
    Extract orderId from a raw order string using regex.
    Handles the format variations we generate.
    """
    match = re.search(r'Order\s+(\d+)', raw, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _validate_raw_string(raw: str, seen_ids: set[str]) -> bool:
    """
    Basic validation that a raw string looks like a valid order.
    Does not require strict format — we want format variation.
    """
    raw = raw.strip()

    if not raw:
        return False

    # must contain an order ID
    order_id = _extract_order_id(raw)
    if not order_id:
        return False

    # must be unique
    if order_id in seen_ids:
        return False

    # must contain a dollar amount
    if not re.search(r'\$[\d,]+\.?\d*', raw):
        return False

    # must be reasonable length
    if len(raw) < 30 or len(raw) > 500:
        return False

    return True


# ── LLM batch generation ───────────────────────────────────────────────────────

def _generate_batch(
    llm: ChatOpenAI,
    prompt: str,
    batch_num: int,
    is_anomaly: bool,
) -> list[str]:
    """
    Call the LLM to generate one batch of raw order strings.
    Returns list of raw strings or empty list on failure.
    """
    try:
        label = "anomaly" if is_anomaly else "normal"
        logger.info(f"Generating {label} batch {batch_num}...")

        response = llm.invoke(prompt)
        content = response.content.strip()

        # split on newlines and clean each line
        lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip() and not line.strip().startswith("```")
        ]

        logger.info(f"Batch {batch_num}: received {len(lines)} strings")
        return lines

    except Exception as e:
        logger.error(f"Batch {batch_num}: error — {e}")
        return []


# ── Main generation pipeline ───────────────────────────────────────────────────

def generate_dataset(
    normal_batches: int = 9,
    anomaly_batches: int = 1,
    batch_size: int = 50,
) -> tuple[list[str], dict[str, bool]]:
    """
    Generate the full synthetic dataset as raw order strings.

    Args:
        normal_batches  -- number of normal order batches
        anomaly_batches -- number of anomaly batches
        batch_size      -- orders per LLM call

    Returns:
        Tuple of:
            all_strings  -- list of raw order strings (real + synthetic)
            labels       -- dict mapping orderId to is_anomaly boolean
    """
    llm = _build_llm()
    all_strings: list[str] = []
    labels: dict[str, bool] = {}
    seen_ids: set[str] = set()
    total_discarded = 0

    # include real orders first
    logger.info("Including 5 real orders from Raft challenge...")
    for raw in REAL_ORDER_STRINGS:
        order_id = _extract_order_id(raw)
        if order_id:
            seen_ids.add(order_id)
            labels[order_id] = False
        all_strings.append(raw)

    # synthetic IDs start at 2000 to avoid collision with real 1001-1005
    next_id = 2000

    # generate normal batches
    for i in range(normal_batches):
        prompt = NORMAL_PROMPT.format(
            count=batch_size,
            start_id=next_id,
        )
        raw_lines = _generate_batch(llm, prompt, i + 1, is_anomaly=False)

        for raw in raw_lines:
            if _validate_raw_string(raw, seen_ids):
                order_id = _extract_order_id(raw)
                seen_ids.add(order_id)
                labels[order_id] = False
                all_strings.append(raw)
            else:
                total_discarded += 1

        next_id += batch_size

        if i < normal_batches - 1:
            time.sleep(1)

    # generate anomaly batches
    for i in range(anomaly_batches):
        prompt = ANOMALY_PROMPT.format(
            count=batch_size,
            start_id=next_id,
        )
        raw_lines = _generate_batch(llm, prompt, i + 1, is_anomaly=True)

        for raw in raw_lines:
            if _validate_raw_string(raw, seen_ids):
                order_id = _extract_order_id(raw)
                seen_ids.add(order_id)
                labels[order_id] = True
                all_strings.append(raw)
            else:
                total_discarded += 1

        next_id += batch_size

        if i < anomaly_batches - 1:
            time.sleep(1)

    normal_count  = sum(1 for v in labels.values() if not v)
    anomaly_count = sum(1 for v in labels.values() if v)

    logger.info("Generation complete:")
    logger.info(f"  Total strings:  {len(all_strings)}")
    logger.info(f"  Normal orders:  {normal_count}")
    logger.info(f"  Anomalous:      {anomaly_count}")
    logger.info(f"  Anomaly rate:   {anomaly_count/len(all_strings)*100:.1f}%")
    logger.info(f"  Discarded:      {total_discarded}")

    return all_strings, labels


def save_dataset(
    strings: list[str],
    labels: dict[str, bool],
) -> None:
    """Save generated strings and labels to ml/data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(ORDERS_PATH, "w") as f:
        json.dump(strings, f, indent=2)
    logger.info(f"Order strings saved to {ORDERS_PATH}")
    logger.info(f"File size: {ORDERS_PATH.stat().st_size / 1024:.1f} KB")

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Anomaly labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    logger.info("Starting synthetic data generation...")
    logger.info("~10 LLM calls. Estimated time: 2-3 minutes.")

    strings, labels = generate_dataset(
        normal_batches=9,
        anomaly_batches=1,
        batch_size=50,
    )

    save_dataset(strings, labels)

    logger.info("Done.")
    logger.info("Next steps:")
    logger.info("  1. python dummy_customer_api_extended.py  (start extended API)")
    logger.info("  2. python ml/anomaly_detector.py          (train model)")