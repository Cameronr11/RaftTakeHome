"""
tests/test_context_guard.py

Tests for the context_guard_node in agent/nodes.py.

The context guard is pure Python with no LLM calls, which makes it
fully unit-testable without mocking. The Raft spec explicitly calls
out 'context window overflow' as a required edge case. These tests
verify the budget math and truncation behavior, including the non-obvious
case where a single order exceeds the entire budget.

We call the function directly rather than running the full LangGraph graph.
"""

import pytest
from agent.nodes import context_guard_node, MAX_TOKENS_PER_CHUNK, CHARS_PER_TOKEN


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_state(raw_orders: list[str]) -> dict:
    """Minimal AgentState dict for context_guard_node input."""
    return {"raw_orders": raw_orders}


def make_order_string(length_chars: int) -> str:
    """Return a plausible-looking order string of approximately the given length."""
    base = "Order 1001: Buyer=Test User, Location=Columbus, OH, Total=$100.00, Items: item"
    if length_chars <= len(base):
        return base[:length_chars]
    return base + ("x" * (length_chars - len(base)))


BUDGET_CHARS = MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN


# ── Basic passthrough ──────────────────────────────────────────────────────────

class TestPassthrough:

    def test_empty_list_returns_empty(self):
        result = context_guard_node(make_state([]))
        assert result["raw_orders"] == []

    def test_small_dataset_passes_through_unchanged(self):
        """The 6-order default dataset must never trigger truncation."""
        orders = [
            "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop",
            "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
            "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc",
        ]
        result = context_guard_node(make_state(orders))
        assert result["raw_orders"] == orders

    def test_orders_exactly_at_budget_pass_through(self):
        """Orders whose total size equals the budget exactly must not be truncated."""
        single = make_order_string(BUDGET_CHARS)
        result = context_guard_node(make_state([single]))
        assert result["raw_orders"] == [single]


# ── Truncation behavior ────────────────────────────────────────────────────────

class TestTruncation:

    def test_oversized_list_is_truncated(self):
        """More orders than the budget allows must result in a shorter list."""
        order = make_order_string(1000)
        num_orders = (BUDGET_CHARS // 1000) + 10  # guaranteed to overflow
        orders = [order] * num_orders
        result = context_guard_node(make_state(orders))
        assert len(result["raw_orders"]) < len(orders)

    def test_truncation_preserves_order_from_front(self):
        """
        When truncating, the FIRST orders in the list must be kept.
        This ensures the earliest-fetched orders are prioritized, not
        arbitrary ones selected by truncation.
        """
        small  = make_order_string(100)
        large  = make_order_string(BUDGET_CHARS + 1)
        orders = [small, large]
        result = context_guard_node(make_state(orders))
        assert result["raw_orders"] == [small]

    def test_truncated_orders_all_fit_within_budget(self):
        """After truncation, total chars of kept orders must fit the budget."""
        order = make_order_string(1000)
        num_orders = (BUDGET_CHARS // 1000) + 10
        orders = [order] * num_orders
        result = context_guard_node(make_state(orders))
        kept = result["raw_orders"]
        total_chars = sum(len(o) for o in kept)
        assert total_chars <= BUDGET_CHARS

    def test_single_oversized_order_returns_empty(self):
        """
        EDGE CASE: if the very first order alone exceeds the token budget,
        the guard returns an empty list. The downstream llm_parser_node will
        then produce an error response via the 'no raw orders' path.

        FLOOR DIVISION ROUNDING: estimated_tokens = total_chars // CHARS_PER_TOKEN.
        An order of exactly BUDGET_CHARS + 1 chars produces estimated_tokens =
        (BUDGET_CHARS + 1) // 4 = BUDGET_CHARS // 4 = MAX_TOKENS_PER_CHUNK,
        which does NOT satisfy the strict > condition. You need at least
        BUDGET_CHARS + CHARS_PER_TOKEN bytes to trigger truncation. The
        effective budget is up to CHARS_PER_TOKEN - 1 bytes above the stated limit.
        """
        # Must add CHARS_PER_TOKEN, not just 1, to breach the floor-division threshold
        giant_order = make_order_string(BUDGET_CHARS + CHARS_PER_TOKEN)
        result = context_guard_node(make_state([giant_order]))
        assert result["raw_orders"] == []
