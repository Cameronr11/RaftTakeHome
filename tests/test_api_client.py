"""
tests/test_api_client.py

Tests for services/api_client.py — specifically the schema drift
detection logic.

The Raft spec explicitly states: 'Must handle edge cases: unpredictable
API schema changes.' The api_client probes a prioritized list of known
envelope keys and logs a warning when a non-primary key is used. These
tests verify that contract without making any real network calls.

We use unittest.mock.patch to replace requests.get so these tests run
instantly and offline, and to avoid any dependency on a running API.
"""

import pytest
from unittest.mock import patch, MagicMock

from services.api_client import fetch_orders, fetch_order_by_id, APIClientError


# ── Helpers ────────────────────────────────────────────────────────────────────

def mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Return a mock requests.Response with the given JSON body."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
    return mock


SAMPLE_ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55",
]


# ── fetch_orders — primary key ─────────────────────────────────────────────────

class TestFetchOrdersPrimaryKey:

    def test_primary_key_raw_orders_returns_list(self):
        """Happy path: API returns 'raw_orders' key as specified."""
        resp = mock_response({"status": "ok", "raw_orders": SAMPLE_ORDERS})
        with patch("services.api_client.requests.get", return_value=resp):
            result = fetch_orders()
        assert result == SAMPLE_ORDERS

    def test_no_warning_on_primary_key(self, caplog):
        """Primary key must not emit a schema-drift warning."""
        import logging
        resp = mock_response({"status": "ok", "raw_orders": SAMPLE_ORDERS})
        with patch("services.api_client.requests.get", return_value=resp):
            with caplog.at_level(logging.WARNING, logger="services.api_client"):
                fetch_orders()
        assert "schema change" not in caplog.text


# ── fetch_orders — schema drift detection ─────────────────────────────────────

class TestFetchOrdersSchemaDrift:

    def test_fallback_key_orders_still_returns_data(self):
        """
        If the API renames 'raw_orders' to 'orders', the client must still
        succeed rather than raising — resilience to format drift is a stated
        requirement.
        """
        resp = mock_response({"status": "ok", "orders": SAMPLE_ORDERS})
        with patch("services.api_client.requests.get", return_value=resp):
            result = fetch_orders()
        assert result == SAMPLE_ORDERS

    def test_fallback_key_emits_warning(self, caplog):
        """
        When a non-primary key is used, a WARNING must be logged so schema
        drift is visible in the logs without crashing the pipeline.
        """
        import logging
        resp = mock_response({"status": "ok", "orders": SAMPLE_ORDERS})
        with patch("services.api_client.requests.get", return_value=resp):
            with caplog.at_level(logging.WARNING, logger="services.api_client"):
                fetch_orders()
        assert "schema change" in caplog.text

    def test_unknown_key_raises_api_client_error(self):
        """
        If the API response contains none of the known envelope keys,
        the client must raise APIClientError rather than returning silently
        empty data that would appear as 'no orders found' to the user.
        """
        resp = mock_response({"status": "ok", "completely_unknown_key": SAMPLE_ORDERS})
        with patch("services.api_client.requests.get", return_value=resp):
            with pytest.raises(APIClientError, match="No usable order list"):
                fetch_orders()

    def test_empty_orders_list_raises_api_client_error(self):
        """An empty list under a valid key provides no usable data."""
        resp = mock_response({"status": "ok", "raw_orders": []})
        with patch("services.api_client.requests.get", return_value=resp):
            with pytest.raises(APIClientError):
                fetch_orders()


# ── fetch_orders — network errors ─────────────────────────────────────────────

class TestFetchOrdersNetworkErrors:

    def test_connection_error_raises_api_client_error(self):
        from requests.exceptions import ConnectionError as ReqConnErr
        with patch("services.api_client.requests.get", side_effect=ReqConnErr):
            with pytest.raises(APIClientError, match="Could not connect"):
                fetch_orders()

    def test_timeout_raises_api_client_error(self):
        from requests.exceptions import Timeout
        with patch("services.api_client.requests.get", side_effect=Timeout):
            with pytest.raises(APIClientError, match="timed out"):
                fetch_orders()


# ── fetch_order_by_id ──────────────────────────────────────────────────────────

class TestFetchOrderById:

    def test_found_order_returns_string(self):
        raw = "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99"
        resp = mock_response({"status": "ok", "raw_order": raw})
        with patch("services.api_client.requests.get", return_value=resp):
            result = fetch_order_by_id("1003")
        assert result == raw

    def test_not_found_returns_none(self):
        """A 404 response must return None, not raise — it is a user-facing
        'order not found' condition, not an infrastructure error."""
        resp = mock_response({"status": "not_found"}, status_code=404)
        with patch("services.api_client.requests.get", return_value=resp):
            result = fetch_order_by_id("9999")
        assert result is None

    def test_fallback_key_still_returns_data(self):
        """Same schema drift tolerance applies to the single-order endpoint."""
        raw = "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99"
        resp = mock_response({"status": "ok", "order": raw})
        with patch("services.api_client.requests.get", return_value=resp):
            result = fetch_order_by_id("1003")
        assert result == raw

    def test_too_short_value_raises_api_client_error(self):
        """
        A value shorter than _MIN_ORDER_STR_LEN (20 chars) is not a real
        order string — it is likely a status value like 'ok'. The client
        must reject it.
        """
        resp = mock_response({"status": "ok", "raw_order": "short"})
        with patch("services.api_client.requests.get", return_value=resp):
            with pytest.raises(APIClientError):
                fetch_order_by_id("1003")
