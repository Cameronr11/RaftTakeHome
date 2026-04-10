"""
tests/test_schemas.py

Tests for the Pydantic models in models/schemas.py.

These tests do NOT test LLM behavior. They test the validation and
normalization logic that runs AFTER the LLM responds — the code-level
contracts that protect the rest of the pipeline regardless of what
any model returns.

Every test here is derived from a stated contract in architecture.md
or a known failure mode (hallucination, bad input format), not from
reading the implementation and encoding its assumptions.
"""

import pytest
from pydantic import ValidationError

from models.schemas import Order, FilterSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

RAW_1001 = "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable"
RAW_1003 = "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse"


def make_order(**overrides) -> Order:
    """Return a valid Order, optionally with fields overridden."""
    defaults = dict(
        orderId="1001",
        buyer="John Davis",
        city="Columbus",
        state="OH",
        total=742.10,
        items=["laptop", "hdmi cable"],
        raw=RAW_1001,
    )
    defaults.update(overrides)
    return Order(**defaults)


# ── Hallucination defense ──────────────────────────────────────────────────────
#
# These tests protect the model_validator in Order.verify_order_id_in_raw.
# This validator is the single code-level guarantee that hallucinated order
# IDs are rejected before they reach the filter or output layers.
# If it is removed or weakened, phantom orders silently flow through.

class TestHallucinationDefense:

    def test_valid_order_id_in_raw_passes(self):
        """Happy path: orderId appears as a discrete token in raw."""
        order = make_order(orderId="1001", raw=RAW_1001)
        assert order.orderId == "1001"

    def test_hallucinated_order_id_raises(self):
        """orderId that is not present in raw must be rejected."""
        with pytest.raises(ValidationError, match="hallucination"):
            make_order(orderId="9999", raw=RAW_1001)

    def test_word_boundary_prevents_substring_match(self):
        """
        orderId='1001' must NOT match inside the token '10011'.

        The validator uses a word-boundary regex (\\b). Without it,
        '1001' would spuriously match inside '10011' and a hallucinated
        short ID could slip through if the raw string contained a longer
        ID that happened to start with the same digits.
        """
        raw_with_longer_id = "Order 10011: Buyer=Jane, Location=Austin, TX, Total=$200.00"
        with pytest.raises(ValidationError, match="hallucination"):
            make_order(orderId="1001", raw=raw_with_longer_id)

    def test_correct_id_in_longer_raw_passes(self):
        """Sanity check: the right id for the longer raw string passes."""
        raw_with_longer_id = "Order 10011: Buyer=Jane, Location=Austin, TX, Total=$200.00"
        order = make_order(orderId="10011", raw=raw_with_longer_id)
        assert order.orderId == "10011"

    def test_no_raw_skips_hallucination_check(self):
        """
        When raw is None the validator is skipped entirely.

        This is documented behavior: offline or pre-parsed orders that
        do not carry their source string should not be rejected solely
        because the source is absent. This test pins that contract so
        a future refactor does not accidentally start requiring raw.
        """
        order = make_order(raw=None)
        assert order.orderId == "1001"


# ── orderId field validator ────────────────────────────────────────────────────

class TestOrderIdValidator:

    def test_non_numeric_string_raises(self):
        """'Order 1001' contains non-digit characters and must be rejected.
        This is the realistic LLM failure mode — model outputs the label, not just digits."""
        with pytest.raises(ValidationError, match="orderId"):
            make_order(orderId="Order 1001", raw=RAW_1001)

    def test_leading_whitespace_is_stripped(self):
        """The validator strips whitespace before checking, so '  1001  ' is valid."""
        order = make_order(orderId="  1001  ", raw=RAW_1001)
        assert order.orderId == "1001"


# ── State field validator ──────────────────────────────────────────────────────

class TestStateValidator:

    def test_lowercase_is_uppercased(self):
        """'oh' must be normalized to 'OH', not rejected."""
        assert make_order(state="oh").state == "OH"

    def test_full_state_name_raises(self):
        """Full names like 'Ohio' are 4+ chars and must be rejected.
        The LLM is supposed to produce 2-letter codes; the validator
        is the backstop if it doesn't."""
        with pytest.raises(ValidationError, match="two-letter"):
            make_order(state="Ohio")

    def test_numeric_state_raises(self):
        """'12' passes the length check but fails isalpha() — tests that
        both conditions in the validator are active."""
        with pytest.raises(ValidationError):
            make_order(state="12")


# ── Total field validator ──────────────────────────────────────────────────────

class TestTotalValidator:

    def test_currency_string_is_stripped(self):
        """'$742.10' from a raw order string must be cleaned to 742.10."""
        assert make_order(total="$742.10").total == 742.10

    def test_zero_total_raises(self):
        """A total of zero is not a valid order amount."""
        with pytest.raises(ValidationError):
            make_order(total=0)

    def test_negative_float_raises(self):
        """Negative totals passed as floats must be rejected."""
        with pytest.raises(ValidationError):
            make_order(total=-50.00)

    def test_negative_string_strips_to_positive(self):
        """
        DOCUMENTED BEHAVIOR: '-100.00' as a string becomes 100.00 because
        re.sub strips the minus sign. This is a known edge case — LLMs should
        never produce negative totals, but if one does, the value is silently
        made positive rather than rejected. This test documents that behavior
        so a future change to reject negatives from strings is a conscious decision.
        """
        order = make_order(total="-100.00")
        assert order.total == 100.00

    def test_empty_string_total_raises(self):
        with pytest.raises(ValidationError):
            make_order(total="")


# ── FilterSpec validators ──────────────────────────────────────────────────────

class TestFilterSpecValidators:

    def test_state_is_uppercased(self):
        spec = FilterSpec(state="oh")
        assert spec.state == "OH"

    def test_order_id_strips_non_digits(self):
        """'Order 1003' must be normalized to '1003'."""
        spec = FilterSpec(order_id="Order 1003")
        assert spec.order_id == "1003"

    def test_order_id_all_alpha_raises(self):
        """An order_id with no digits at all must be rejected."""
        with pytest.raises(ValidationError):
            FilterSpec(order_id="ABC")

    def test_empty_spec_all_none(self):
        """An empty FilterSpec must have all fields as None — return-everything semantics."""
        spec = FilterSpec()
        assert all(v is None for v in spec.model_dump().values())
