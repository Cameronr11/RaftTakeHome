"""
services/api_client.py

Thin HTTP wrapper around the dummy customer orders API.
Provides two functions that fetch raw unstructured order strings
from the running Flask server. All HTTP concerns — base URL,
timeouts, error handling — are isolated here so nothing else
in the system needs to think about them.

Functions:
    fetch_orders(limit)      -- Fetch all orders or a limited sample.
    fetch_order_by_id(id)    -- Fetch a single order by its ID.

Raises:
    APIClientError           -- On any network, timeout, or unexpected
                                response shape error.
"""

import os
import json
import logging
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5001")
TIMEOUT = 10  # seconds

# Candidate envelope keys tried in order when extracting order data from
# API responses. The first key whose value passes type validation is used.
# A warning is logged whenever a non-primary key matches so schema drift
# is visible in logs without failing the request.
_ORDER_LIST_KEYS = ["raw_orders", "orders", "data", "results", "records"]
_ORDER_STR_KEYS  = ["raw_order",  "order",  "data", "result",  "record"]

# Minimum length that distinguishes a real order string from a short status
# value like "ok" (2 chars) or "not_found" (9 chars). Assumption: no valid
# order string in this system is shorter than 20 characters.
_MIN_ORDER_STR_LEN = 20


class APIClientError(Exception):
    """Raised when the orders API cannot be reached or returns unexpected data."""
    pass



#Function 1 to gather orders from flask server
def fetch_orders(limit: Optional[int] = None) -> list[str]:
    """
    Fetch raw order strings from GET /api/orders.

    Args:
        limit: optional max number of orders to return.
               If None, the API returns all available orders.

    Returns:
        List of raw unstructured order strings.

    Raises:
        APIClientError: if the request fails or the response shape is unexpected.
    """
    url = f"{API_BASE_URL}/api/orders"
    params = {"limit": limit} if limit is not None else {}

    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        for key in _ORDER_LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list) and value and isinstance(value[0], str):
                if key != _ORDER_LIST_KEYS[0]:
                    logger.warning(
                        f"API schema change detected on /api/orders — "
                        f"expected '{_ORDER_LIST_KEYS[0]}' key, found '{key}'. "
                        f"Response keys: {list(data.keys())}"
                    )
                return value

        raise APIClientError(
            f"No usable order list found in /api/orders response. "
            f"Tried keys: {_ORDER_LIST_KEYS}. Got: {list(data.keys())}"
        )

    except requests.ConnectionError:
        raise APIClientError(
            f"Could not connect to orders API at {API_BASE_URL}. "
            f"Is dummy_customer_api.py running?"
        )
    except requests.Timeout:
        raise APIClientError(
            f"Request to {API_BASE_URL}/api/orders timed out after {TIMEOUT}s."
        )
    except requests.HTTPError as e:
        raise APIClientError(f"HTTP error from /api/orders: {e}")

    except json.JSONDecodeError as e:
        raise APIClientError(f"Invalid JSON response from /api/orders: {e}")


#Function 2 to gather order by id from flask server
def fetch_order_by_id(order_id: str) -> str | None:
    """
    Fetch a single raw order string from GET /api/order/<order_id>.

    Args:
        order_id: the numeric order ID string (e.g. "1003").

    Returns:
        Raw unstructured order string, or None if the order was not found.

    Raises:
        APIClientError: if the request fails or the response shape is unexpected.
    """
    url = f"{API_BASE_URL}/api/order/{order_id}"

    try:
        response = requests.get(url, timeout=TIMEOUT)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        data = response.json()

        for key in _ORDER_STR_KEYS:
            value = data.get(key)
            if isinstance(value, str) and len(value) >= _MIN_ORDER_STR_LEN:
                if key != _ORDER_STR_KEYS[0]:
                    logger.warning(
                        f"API schema change detected on /api/order/{order_id} — "
                        f"expected '{_ORDER_STR_KEYS[0]}' key, found '{key}'. "
                        f"Response keys: {list(data.keys())}"
                    )
                return value

        raise APIClientError(
            f"No usable order string found in /api/order/{order_id} response. "
            f"Tried keys: {_ORDER_STR_KEYS}. Got: {list(data.keys())}"
        )

    except requests.ConnectionError:
        raise APIClientError(
            f"Could not connect to orders API at {API_BASE_URL}. "
            f"Is dummy_customer_api.py running?"
        )
    except requests.Timeout:
        raise APIClientError(
            f"Request to {url} timed out after {TIMEOUT}s."
        )
    except requests.HTTPError as e:
        raise APIClientError(f"HTTP error from /api/order/{order_id}: {e}")

    except json.JSONDecodeError as e:
        raise APIClientError(f"Invalid JSON response from /api/order/{order_id}: {e}")