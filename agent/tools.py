"""
agent/tools.py

LangChain tool definitions that wrap the API client functions.
These are the only two ways the agent can retrieve order data —
fetching all orders or fetching a single order by ID.

The @tool decorator exposes each function to the LangGraph agent.
The docstring on each tool is what the LLM reads to decide when
and how to invoke it, so they are written to be unambiguous.

Tools always return strings — either the raw order data or a
structured error message. They never raise exceptions; errors
are surfaced as return values so the graph can handle them
gracefully through conditional edges rather than crashes.
"""

import json
import logging
from langchain_core.tools import tool
from services.api_client import fetch_orders, fetch_order_by_id, APIClientError

logger = logging.getLogger(__name__)


@tool
def get_all_orders(limit: Optional[int] = None) -> str:
    """
    Fetch all customer orders from the orders API.

    Use this tool when the user wants to search, filter, or analyze
    orders without specifying a particular order ID. This is the
    correct tool for queries like:
        - 'Show me all orders in Ohio'
        - 'Find orders over $500'
        - 'Which orders included a laptop?'

    Args:
        limit: optional maximum number of orders to retrieve.
               Pass None to retrieve all available orders.
               Use a limit only if the user explicitly asks for
               a specific number of results.

    Returns:
        A JSON string containing a list of raw order strings,
        or an error message string if the API is unavailable.
    """
    try:
        logger.info(f"Fetching orders from API (limit={limit})")
        raw_orders = fetch_orders(limit=limit)
        logger.info(f"Retrieved {len(raw_orders)} raw orders")
        return json.dumps({
            "status": "ok",
            "count": len(raw_orders),
            "raw_orders": raw_orders
        })
    except APIClientError as e:
        logger.error(f"API error fetching orders: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def get_order_by_id(order_id: str) -> str:
    """
    Fetch a single customer order by its numeric order ID.

    Use this tool only when the user specifies a particular order ID
    directly in their query. This is the correct tool for queries like:
        - 'Show me order 1003'
        - 'What was in order number 1001?'
        - 'Get me the details for order 1005'

    Do NOT use this tool for general filtering or searching —
    use get_all_orders instead.

    Args:
        order_id: the numeric order ID as a string, e.g. '1003'.

    Returns:
        A JSON string containing the raw order string if found,
        a not-found message if the order ID does not exist,
        or an error message if the API is unavailable.
    """
    try:
        logger.info(f"Fetching order {order_id} from API")
        raw_order = fetch_order_by_id(order_id)

        if raw_order is None:
            logger.warning(f"Order {order_id} not found")
            return json.dumps({
                "status": "not_found",
                "message": f"No order found with ID {order_id}"
            })

        logger.info(f"Retrieved order {order_id}")
        return json.dumps({
            "status": "ok",
            "raw_order": raw_order
        })

    except APIClientError as e:
        logger.error(f"API error fetching order {order_id}: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# The list of tools the agent has access to.
# Imported by the graph and bound to the LLM.
TOOLS = [get_all_orders, get_order_by_id]