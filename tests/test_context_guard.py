"""
tests/test_context_guard.py

Tests for the context_guard_node in agent/nodes.py.

The context guard is pure Python with no LLM calls, which makes it
fully unit-testable without mocking. The Raft spec explicitly calls
out 'context window overflow' as a required edge case. These tests
verify the budget math and truncation behavior, including the non-obvious
case where a single order exceeds the entire budget.

We call the function directly rather than running the full LangGraph graph.
Token counts are computed with tiktoken (cl100k_base) to match the node's
own counting logic exactly.
"""

import pytest
import tiktoken
from agent.nodes import context_guard_node, MAX_TOKENS_PER_CHUNK


# ── Helpers ────────────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(s: str) -> int:
    """Return the exact tiktoken token count for a string."""
    return len(_enc.encode(s))


def make_state(raw_orders: list[str]) -> dict:
    """Minimal AgentState dict for context_guard_node input."""
    return {"raw_orders": raw_orders}


def make_order_string(target_tokens: int) -> str:
    """
    Return an order string whose tiktoken token count is at least target_tokens.

    Starts from a realistic order string and pads with ' x' (each instance
    is a single token in cl100k_base) until the target is reached.
    """
    base = "Order 1001: Buyer=Test User, Location=Columbus, OH, Total=$100.00, Items: item"
    while count_tokens(base) < target_tokens:
        base += " x"
    return base


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
        """Orders whose total token count equals the budget must not be truncated."""
        single = make_order_string(MAX_TOKENS_PER_CHUNK)
        assert count_tokens(single) >= MAX_TOKENS_PER_CHUNK
        # Trim back to exactly the budget using the tokenizer
        tokens = _enc.encode(single)[:MAX_TOKENS_PER_CHUNK]
        single_exact = _enc.decode(tokens)
        result = context_guard_node(make_state([single_exact]))
        assert result["raw_orders"] == [single_exact]


# ── Truncation behavior ────────────────────────────────────────────────────────

class TestTruncation:

    def test_oversized_list_is_truncated(self):
        """More orders than the budget allows must result in a shorter list."""
        order = make_order_string(1000)
        num_orders = (MAX_TOKENS_PER_CHUNK // count_tokens(order)) + 10
        orders = [order] * num_orders
        result = context_guard_node(make_state(orders))
        assert len(result["raw_orders"]) < len(orders)

    def test_truncation_preserves_order_from_front(self):
        """
        When truncating, the FIRST orders in the list must be kept.
        This ensures the earliest-fetched orders are prioritized, not
        arbitrary ones selected by truncation.
        """
        small = make_order_string(10)
        large = make_order_string(MAX_TOKENS_PER_CHUNK + 1)
        orders = [small, large]
        result = context_guard_node(make_state(orders))
        assert result["raw_orders"] == [small]

    def test_truncated_orders_all_fit_within_budget(self):
        """After truncation, total tokens of kept orders must fit the budget."""
        order = make_order_string(1000)
        num_orders = (MAX_TOKENS_PER_CHUNK // count_tokens(order)) + 10
        orders = [order] * num_orders
        result = context_guard_node(make_state(orders))
        kept = result["raw_orders"]
        total_tokens = sum(count_tokens(o) for o in kept)
        assert total_tokens <= MAX_TOKENS_PER_CHUNK

    def test_single_oversized_order_returns_empty(self):
        """
        EDGE CASE: if the very first order alone exceeds the token budget,
        the guard returns an empty list. The downstream llm_parser_node will
        then produce an error response via the 'no raw orders' path.
        """
        giant_order = make_order_string(MAX_TOKENS_PER_CHUNK + 1)
        assert count_tokens(giant_order) > MAX_TOKENS_PER_CHUNK
        result = context_guard_node(make_state([giant_order]))
        assert result["raw_orders"] == []
