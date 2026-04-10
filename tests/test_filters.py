"""
tests/test_filters.py

Tests for services/filters.py — the deterministic Python filter layer.

This is the highest-ROI test file in the project because apply_filters()
is the system's primary correctness and determinism guarantee. The same
FilterSpec + same Orders must always produce the same result, every time,
regardless of LLM behavior. If this layer has a bug, no amount of LLM
prompt engineering will compensate.

Focus areas:
  - Boundary conditions on numeric ranges (inclusive vs exclusive)
  - String matching semantics (case, partial vs exact)
  - Sort stability and type coercion surprises
  - Interactions between multiple active filters
  - Empty inputs and null specs
"""

import pytest
from models.schemas import Order, FilterSpec
from services.filters import apply_filters


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_order(order_id, buyer, state, total, items=None) -> Order:
    """Minimal Order factory. No raw= so hallucination check is skipped."""
    return Order(
        orderId=order_id,
        buyer=buyer,
        city="TestCity",
        state=state,
        total=total,
        items=items or [],
        raw=None,
    )


@pytest.fixture
def sample_orders() -> list[Order]:
    """
    A fixed dataset that mirrors the Raft-provided API orders closely
    enough to be representative, but is defined here rather than fetched
    from the API so tests are never network-dependent.
    """
    return [
        make_order("1001", "John Davis",   "OH", 742.10,  ["laptop", "hdmi cable"]),
        make_order("1002", "Sarah Liu",    "TX", 156.55,  ["headphones"]),
        make_order("1003", "Mike Turner",  "OH", 1299.99, ["gaming pc", "mouse"]),
        make_order("1004", "Rachel Kim",   "WA",  89.50,  ["coffee maker"]),
        make_order("1005", "Chris Myers",  "OH", 512.00,  ["monitor", "desk lamp"]),
    ]


# ── Empty spec — passthrough ───────────────────────────────────────────────────

class TestEmptySpec:

    def test_empty_spec_returns_all_orders(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec())
        assert len(result) == len(sample_orders)

    def test_empty_input_returns_empty(self):
        result = apply_filters([], FilterSpec(state="OH"))
        assert result == []


# ── State filter ───────────────────────────────────────────────────────────────

class TestStateFilter:

    def test_returns_only_matching_state(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(state="OH"))
        assert all(o.state == "OH" for o in result)
        assert len(result) == 3

    def test_nonexistent_state_returns_empty(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(state="NY"))
        assert result == []

    def test_state_match_is_exact_not_partial(self, sample_orders):
        """'O' must not match 'OH' — state codes are exact."""
        result = apply_filters(sample_orders, FilterSpec(state="O"))
        assert result == []


# ── Total range filters ────────────────────────────────────────────────────────

class TestTotalFilters:

    def test_min_total_is_inclusive(self, sample_orders):
        """
        BOUNDARY CONDITION: an order with total == min_total must be INCLUDED.
        The spec says 'over $X' but the implementation uses >=.
        This test pins the inclusive contract so a refactor to > is caught.
        """
        result = apply_filters(sample_orders, FilterSpec(min_total=512.00))
        order_ids = {o.orderId for o in result}
        assert "1005" in order_ids  # total is exactly 512.00

    def test_min_total_excludes_below(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(min_total=512.01))
        order_ids = {o.orderId for o in result}
        assert "1005" not in order_ids

    def test_max_total_is_inclusive(self, sample_orders):
        """BOUNDARY CONDITION: an order with total == max_total must be INCLUDED."""
        result = apply_filters(sample_orders, FilterSpec(max_total=156.55))
        order_ids = {o.orderId for o in result}
        assert "1002" in order_ids  # total is exactly 156.55

    def test_max_total_excludes_above(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(max_total=156.54))
        order_ids = {o.orderId for o in result}
        assert "1002" not in order_ids

    def test_range_keeps_only_orders_within_bounds(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(min_total=500.00, max_total=800.00))
        for order in result:
            assert 500.00 <= order.total <= 800.00

    def test_impossible_range_returns_empty(self, sample_orders):
        """min > max is a user error; the filter should simply return nothing."""
        result = apply_filters(sample_orders, FilterSpec(min_total=1000.00, max_total=100.00))
        assert result == []


# ── Item keyword filter ────────────────────────────────────────────────────────

class TestItemKeywordFilter:

    def test_exact_item_match(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(item_keyword="laptop"))
        assert len(result) == 1
        assert result[0].orderId == "1001"

    def test_partial_item_match(self, sample_orders):
        """'lap' must match 'laptop' — the filter is a substring match."""
        result = apply_filters(sample_orders, FilterSpec(item_keyword="lap"))
        assert any(o.orderId == "1001" for o in result)

    def test_keyword_is_case_insensitive(self, sample_orders):
        result_lower = apply_filters(sample_orders, FilterSpec(item_keyword="laptop"))
        result_upper = apply_filters(sample_orders, FilterSpec(item_keyword="LAPTOP"))
        assert {o.orderId for o in result_lower} == {o.orderId for o in result_upper}

    def test_keyword_does_not_match_buyer_name(self, sample_orders):
        """
        Item keyword must only match against order.items, not buyer names.
        'Kim' is part of buyer 'Rachel Kim' but should not appear in results
        when filtering by item_keyword='kim'.
        """
        result = apply_filters(sample_orders, FilterSpec(item_keyword="kim"))
        assert result == []

    def test_order_with_empty_items_is_excluded(self):
        orders = [make_order("1001", "John Davis", "OH", 100.00, items=[])]
        result = apply_filters(orders, FilterSpec(item_keyword="laptop"))
        assert result == []


# ── Buyer name filter ──────────────────────────────────────────────────────────

class TestBuyerNameFilter:

    def test_exact_name_match(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(buyer_name="Sarah Liu"))
        assert len(result) == 1
        assert result[0].orderId == "1002"

    def test_partial_first_name_match(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(buyer_name="Sarah"))
        assert len(result) == 1

    def test_partial_last_name_match(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(buyer_name="Liu"))
        assert len(result) == 1

    def test_buyer_match_is_case_insensitive(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(buyer_name="sarah liu"))
        assert len(result) == 1

    def test_no_buyer_match_returns_empty(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(buyer_name="Nonexistent Person"))
        assert result == []


# ── Sorting ────────────────────────────────────────────────────────────────────

class TestSorting:

    def test_sort_by_total_descending(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(sort_by="total", sort_order="desc"))
        totals = [o.total for o in result]
        assert totals == sorted(totals, reverse=True)

    def test_sort_by_total_ascending(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(sort_by="total", sort_order="asc"))
        totals = [o.total for o in result]
        assert totals == sorted(totals)

    def test_sort_by_buyer_alphabetical(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(sort_by="buyer", sort_order="asc"))
        names = [o.buyer.lower() for o in result]
        assert names == sorted(names)

    def test_sort_by_order_id_is_lexicographic(self):
        """
        DOCUMENTED BEHAVIOR: orderId is sorted as a string, not a number.
        '1001' < '999' in numeric order, but '1001' < '999' lexicographically
        because '1' < '9'. For 4-digit IDs of equal length this matches
        numeric order, but the behavior diverges with mixed-length IDs.
        This test documents the actual behavior so a future switch to
        numeric sort is a conscious decision.
        """
        orders = [
            make_order("999",  "A", "OH", 100.0),
            make_order("1001", "B", "OH", 200.0),
            make_order("2000", "C", "OH", 300.0),
        ]
        result = apply_filters(orders, FilterSpec(sort_by="orderId", sort_order="asc"))
        ids = [o.orderId for o in result]
        # lexicographic: "1001" < "2000" < "999"
        assert ids == ["1001", "2000", "999"]

    def test_unknown_sort_field_returns_unsorted_not_error(self, sample_orders):
        """An unrecognized sort_by field must be silently skipped, not raise."""
        result = apply_filters(sample_orders, FilterSpec(sort_by="nonexistent_field"))
        assert len(result) == len(sample_orders)


# ── Limit ─────────────────────────────────────────────────────────────────────

class TestLimit:

    def test_limit_truncates_results(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(limit=2))
        assert len(result) == 2

    def test_limit_larger_than_results_returns_all(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(limit=100))
        assert len(result) == len(sample_orders)

    def test_limit_applied_after_sort(self, sample_orders):
        """Top 2 most expensive — limit must come after sort."""
        result = apply_filters(
            sample_orders,
            FilterSpec(sort_by="total", sort_order="desc", limit=2)
        )
        assert len(result) == 2
        assert result[0].total >= result[1].total
        assert result[0].orderId == "1003"  # 1299.99 is highest


# ── Combined filters ───────────────────────────────────────────────────────────

class TestCombinedFilters:

    def test_state_and_min_total(self, sample_orders):
        """The Raft spec's example query: Ohio AND total >= 500.
        Three OH orders qualify: 1001 (742.10), 1003 (1299.99), 1005 (512.00).
        min_total is inclusive so 512.00 is included."""
        result = apply_filters(sample_orders, FilterSpec(state="OH", min_total=500.00))
        assert all(o.state == "OH" for o in result)
        assert all(o.total >= 500.00 for o in result)
        assert len(result) == 3  # 1001 (742.10), 1003 (1299.99), 1005 (512.00)

    def test_state_and_item_keyword(self, sample_orders):
        result = apply_filters(sample_orders, FilterSpec(state="OH", item_keyword="laptop"))
        assert len(result) == 1
        assert result[0].orderId == "1001"

    def test_all_filters_active_no_match(self, sample_orders):
        """Applying contradictory filters together must return empty, not error."""
        result = apply_filters(
            sample_orders,
            FilterSpec(state="TX", min_total=1000.00, item_keyword="laptop")
        )
        assert result == []
