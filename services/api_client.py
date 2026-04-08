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
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5001")
TIMEOUT = 10  # seconds


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

        if "raw_orders" not in data:
            raise APIClientError(
                f"Unexpected response shape from /api/orders — "
                f"missing 'raw_orders' key. Got: {list(data.keys())}"
            )

        return data["raw_orders"]

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

        if "raw_order" not in data:
            raise APIClientError(
                f"Unexpected response shape from /api/order/{order_id} — "
                f"missing 'raw_order' key. Got: {list(data.keys())}"
            )

        return data["raw_order"]

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