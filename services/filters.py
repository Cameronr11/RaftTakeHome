"""
services/filters.py

Deterministic Python filter layer that applies a FilterSpec against a
list of validated Order objects. No LLM calls are made here — this is
pure typed Python logic operating on Pydantic objects.

This is intentional by design. Filtering is the step that must be
correct and repeatable every time regardless of LLM behavior. By keeping
it in Python we guarantee that the same query always produces the same
result from the same dataset.

Functions:
    apply_filters(orders, spec) -- Filter, sort, and limit a list of
                                   Order objects according to a FilterSpec.
"""

import logging
from models.schemas import FilterSpec, Order

logger = logging.getLogger(__name__)


def apply_filters(orders: list[Order], spec: FilterSpec) -> list[Order]:
    """
    Apply a FilterSpec to a list of Order objects and return matching orders.

    Filters are applied in sequence — each active field on the FilterSpec
    narrows the result set. Fields that are None are skipped entirely so
    an empty FilterSpec returns all orders unchanged.

    After filtering, results are optionally sorted and limited based on
    the sort_by, sort_order, and limit fields of the FilterSpec.

    Matching behavior:
        state       -- exact match, case-insensitive (both normalized to uppercase)
        min_total   -- order total must be >= this value
        max_total   -- order total must be <= this value
        item_keyword-- case-insensitive partial match against any item in the order
        buyer_name  -- case-insensitive partial match against the buyer name

    Args:
        orders -- list of validated Order objects from the parser
        spec   -- FilterSpec produced by the query planner node

    Returns:
        Filtered, sorted, and limited list of Order objects.
    """
    logger.info(
        f"Applying filters to {len(orders)} orders — "
        f"state={spec.state}, min_total={spec.min_total}, "
        f"max_total={spec.max_total}, item_keyword={spec.item_keyword}, "
        f"buyer_name={spec.buyer_name}, sort_by={spec.sort_by}, "
        f"sort_order={spec.sort_order}, limit={spec.limit}"
    )

    results = orders

    # ── State filter ──────────────────────────────────────────────────────────
    # Exact match — both sides already uppercase due to schema normalization
    if spec.state is not None:
        before = len(results)
        results = [o for o in results if o.state == spec.state]
        logger.debug(f"State filter '{spec.state}': {before} → {len(results)} orders")

    # ── Total range filters ───────────────────────────────────────────────────
    if spec.min_total is not None:
        before = len(results)
        results = [o for o in results if o.total >= spec.min_total]
        logger.debug(f"Min total filter {spec.min_total}: {before} → {len(results)} orders")

    if spec.max_total is not None:
        before = len(results)
        results = [o for o in results if o.total <= spec.max_total]
        logger.debug(f"Max total filter {spec.max_total}: {before} → {len(results)} orders")

    # ── Item keyword filter ───────────────────────────────────────────────────
    # Case-insensitive partial match against any item in the order's item list
    if spec.item_keyword is not None:
        keyword = spec.item_keyword.lower()
        before = len(results)
        results = [
            o for o in results
            if o.items and any(keyword in item.lower() for item in o.items)
        ]
        logger.debug(
            f"Item keyword filter '{spec.item_keyword}': "
            f"{before} → {len(results)} orders"
        )

    # ── Buyer name filter ─────────────────────────────────────────────────────
    # Case-insensitive partial match against buyer name
    if spec.buyer_name is not None:
        name = spec.buyer_name.lower()
        before = len(results)
        results = [o for o in results if name in o.buyer.lower()]
        logger.debug(
            f"Buyer name filter '{spec.buyer_name}': "
            f"{before} → {len(results)} orders"
        )

    # ── Sorting ───────────────────────────────────────────────────────────────
    if spec.sort_by is not None:
        reverse = spec.sort_order == "desc"

        sort_keys = {
            "total":   lambda o: o.total,
            "buyer":   lambda o: o.buyer.lower(),
            "orderId": lambda o: o.orderId,
        }

        if spec.sort_by in sort_keys:
            results = sorted(results, key=sort_keys[spec.sort_by], reverse=reverse)
            logger.debug(
                f"Sorted by '{spec.sort_by}' "
                f"({'desc' if reverse else 'asc'})"
            )
        else:
            logger.warning(
                f"Unknown sort_by field '{spec.sort_by}' — skipping sort"
            )

    # ── Limit ─────────────────────────────────────────────────────────────────
    if spec.limit is not None:
        before = len(results)
        results = results[:spec.limit]
        logger.debug(f"Limit {spec.limit}: {before} → {len(results)} orders")

    logger.info(f"Filter complete — {len(results)} orders match")
    return results