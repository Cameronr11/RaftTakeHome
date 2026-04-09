"""
agent/nodes.py

LangGraph node functions for the order query agent. Each node is a pure
function that reads from AgentState, performs one unit of work, and returns
a partial state update. Nodes never mutate state directly.

Nodes that have conditional routing return Command objects which combine
state updates and routing decisions in a single return value. Nodes with
fixed routing return plain dicts.

Node execution order:
    query_planner → api_fetcher → context_guard → llm_parser
                                                         ↓
    output ← filter_validate ←──────────────────────────┘

Error short-circuits:
    api_fetcher  → output (on API failure)
    query_planner → output (on out-of-scope query)
    llm_parser   → output (on total parse failure)

Dependencies:
    services/parser.py    -- LLM extraction pipeline
    services/filters.py   -- deterministic filter layer
    services/api_client.py -- HTTP wrapper (via tools)
    models/schemas.py     -- Order, FilterSpec, AgentResponse
"""

import os
import json
import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from agent.state import AgentState
from models.schemas import FilterSpec, AgentResponse
from services.api_client import fetch_orders, fetch_order_by_id, APIClientError
from services.parser import parse_orders
from services.filters import apply_filters

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TOKENS_PER_CHUNK = 12000   # conservative limit per LLM call
CHARS_PER_TOKEN      = 4      # standard approximation

# ── LLM client (shared across nodes that need it) ─────────────────────────────

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b:exacto"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0,
    )

_llm = _build_llm()

# ── Query planner prompt ───────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a query intent extraction engine for an order \
management system.
Your only job is to read a natural language query and extract structured \
filter criteria from it. Nothing more.

─── DATA MODEL ───────────────────────────────────────────────────────────────
The order system contains records with these fields:
  orderId   : numeric order ID e.g. "1001"
  buyer     : customer full name e.g. "John Davis"
  city      : city name e.g. "Columbus"
  state     : two-letter US state code e.g. "OH"
  total     : order value in dollars e.g. 742.10
  items     : list of purchased items e.g. ["laptop", "hdmi cable"]

─── FILTER FIELDS ────────────────────────────────────────────────────────────
Extract values for only these fields:
  state       : two-letter uppercase state code
  min_total   : minimum order total as a float
  max_total   : maximum order total as a float
  item_keyword: single keyword to match against item names
  buyer_name  : partial or full buyer name
  limit       : maximum number of results as an integer
  sort_by     : "total", "buyer", or "orderId"
  sort_order  : "asc" or "desc"

─── RULES ────────────────────────────────────────────────────────────────────
1. Only populate fields explicitly stated in the query
2. NEVER invent or assume values not present in the query
3. Fields not mentioned must be null — no exceptions
4. Convert full state names to codes: "Ohio" → "OH", "Texas" → "TX"
5. "over $X" or "more than $X" → min_total=X (explicit number only)
6. "under $X" or "less than $X" → max_total=X (explicit number only)
7. Vague price words like "expensive" or "cheap" without a number → null
8. "top N" or "most expensive N" → sort_by="total", sort_order="desc", limit=N
9. "cheapest N" or "lowest N"   → sort_by="total", sort_order="asc",  limit=N
10. If the query has nothing to do with orders → all fields null

─── EXAMPLES ─────────────────────────────────────────────────────────────────
Query: "Show me all orders where the buyer was located in Ohio and total \
value was over 500"
Result: state="OH", min_total=500.0

Query: "Who bought a laptop?"
Result: item_keyword="laptop"

Query: "Show me the top 3 most expensive orders"
Result: sort_by="total", sort_order="desc", limit=3

Query: "Show me orders by Sarah Liu"
Result: buyer_name="Sarah Liu"

Query: "Find orders between $100 and $800"
Result: min_total=100.0, max_total=800.0

Query: "Show me expensive orders"
Result: all fields null — no explicit dollar amount

Query: "What is the weather today?"
Result: all fields null — unrelated to orders"""

_PLANNER_HUMAN = "Extract the filter criteria from this query:\n\n{query}"

_planner_chain = (
    ChatPromptTemplate.from_messages([
        ("system", _PLANNER_SYSTEM),
        ("human",  _PLANNER_HUMAN),
    ])
    | _llm.with_structured_output(FilterSpec)
)

# ── ORDER-RELATED KEYWORDS ─────────────────────────────────────────────────────
# Used to detect genuinely out-of-scope queries after the planner returns
# an all-null FilterSpec.

_ORDER_KEYWORDS = {
    "order", "orders", "buyer", "buyers", "total", "item", "items",
    "purchase", "purchased", "bought", "show", "find", "list", "get",
    "fetch", "search", "filter", "ohio", "texas", "washington", "seattle",
    "austin", "columbus", "cleveland", "cincinnati", "monitor", "laptop",
    "gaming", "headphones", "coffee",
}


def _is_order_related(query: str) -> bool:
    """Return True if the query contains at least one order-domain keyword."""
    words = set(query.lower().split())
    return bool(words & _ORDER_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Query Planner
# ══════════════════════════════════════════════════════════════════════════════

def query_planner_node(
    state: AgentState,
) -> Command[Literal["api_fetcher", "output"]]:
    """
    Extract structured filter intent from the natural language query.

    Calls the LLM with with_structured_output(FilterSpec) to convert the
    user's query into a FilterSpec object. If the query is clearly unrelated
    to orders (all fields null AND no order keywords detected), short-circuits
    to the output node with an error message.

    Reads:  state["query"]
    Writes: state["filter_spec"]
            state["error"] (on out-of-scope query only)
    Routes: → api_fetcher (normal)
            → output (out-of-scope query)
    """
    query = state["query"]
    logger.info(f"Query planner processing: '{query}'")

    try:
        filter_spec: FilterSpec = _planner_chain.invoke({"query": query})
        logger.info(f"Extracted FilterSpec: {filter_spec.model_dump()}")

        # check for out-of-scope query — all null AND no order keywords
        all_null = all(v is None for v in filter_spec.model_dump().values())
        if all_null and not _is_order_related(query):
            error_msg = (
                f"Query '{query}' does not appear to be related to orders. "
                f"Please ask about order data."
            )
            logger.warning(error_msg)
            return Command(
                update={"error": error_msg},
                goto="output",
            )

        return Command(
            update={"filter_spec": filter_spec},
            goto="api_fetcher",
        )

    except Exception as e:
        error_msg = f"Query planner failed: {e}"
        logger.error(error_msg)
        return Command(
            update={"error": error_msg},
            goto="output",
        )


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — API Fetcher
# ══════════════════════════════════════════════════════════════════════════════

def api_fetcher_node(
    state: AgentState,
) -> Command[Literal["context_guard", "output"]]:
    """
    Fetch raw order strings from the Flask API.

    Calls the appropriate API endpoint based on the FilterSpec. For most
    queries this means fetching all orders. If the API is unavailable,
    short-circuits to the output node with a descriptive error.

    Reads:  state["filter_spec"]
    Writes: state["raw_orders"]
            state["error"] (on API failure only)
    Routes: → context_guard (normal)
            → output (API failure)
    """
    logger.info("API fetcher: fetching orders from Flask API")

    try:
        raw_orders = fetch_orders()
        logger.info(f"API fetcher: retrieved {len(raw_orders)} raw orders")

        return Command(
            update={"raw_orders": raw_orders},
            goto="context_guard",
        )

    except APIClientError as e:
        error_msg = (
            f"Could not retrieve orders from the API: {e}. "
            f"Ensure dummy_customer_api.py is running on port 5001."
        )
        logger.error(error_msg)
        return Command(
            update={"error": error_msg},
            goto="output",
        )


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Context Guard
# ══════════════════════════════════════════════════════════════════════════════

def context_guard_node(state: AgentState) -> dict:
    """
    Protect against context window overflow before the LLM parser runs.

    Estimates the token count of raw order strings using the standard
    4-chars-per-token heuristic. If the total exceeds MAX_TOKENS_PER_CHUNK,
    truncates the list and logs a warning. For the current dataset of 5
    orders this will never trigger, but the guard must exist to satisfy the
    edge case requirement in the Raft spec.

    Reads:  state["raw_orders"]
    Writes: state["raw_orders"] (potentially truncated)
    Routes: → llm_parser (always, fixed edge)
    """
    raw_orders = state.get("raw_orders", [])

    if not raw_orders:
        logger.warning("Context guard: no raw orders to process")
        return {"raw_orders": []}

    total_chars = sum(len(r) for r in raw_orders)
    estimated_tokens = total_chars // CHARS_PER_TOKEN

    logger.info(
        f"Context guard: {len(raw_orders)} orders, "
        f"~{estimated_tokens} estimated tokens"
    )

    if estimated_tokens > MAX_TOKENS_PER_CHUNK:
        # calculate how many orders fit within the token budget
        budget_chars = MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN
        kept = []
        used = 0
        for raw in raw_orders:
            if used + len(raw) > budget_chars:
                break
            kept.append(raw)
            used += len(raw)

        logger.warning(
            f"Context guard: truncated from {len(raw_orders)} to "
            f"{len(kept)} orders to stay within token limit"
        )
        return {"raw_orders": kept}

    return {"raw_orders": raw_orders}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — LLM Parser
# ══════════════════════════════════════════════════════════════════════════════

def llm_parser_node(
    state: AgentState,
) -> Command[Literal["filter_validate", "output"]]:
    """
    Parse raw unstructured order strings into validated Order objects.

    Calls parse_orders() from services/parser.py which handles individual
    order parsing, hallucination detection, and per-order retries internally.
    If no orders could be parsed at all, short-circuits to the output node.

    Reads:  state["raw_orders"]
    Writes: state["parsed_orders"]
            state["parse_errors"]
            state["retry_count"]
    Routes: → filter_validate (at least one order parsed)
            → output (total parse failure)
    """
    raw_orders = state.get("raw_orders", [])

    if not raw_orders:
        error_msg = "No raw orders available to parse."
        logger.error(error_msg)
        return Command(
            update={
                "parsed_orders": [],
                "parse_errors": [error_msg],
                "error": error_msg,
            },
            goto="output",
        )

    logger.info(f"LLM parser: parsing {len(raw_orders)} raw orders")

    orders, errors = parse_orders(raw_orders)

    logger.info(
        f"LLM parser: {len(orders)} parsed successfully, "
        f"{len(errors)} failed"
    )

    if not orders:
        error_msg = (
            f"Failed to parse any orders. "
            f"Errors: {'; '.join(errors[:3])}"
        )
        logger.error(error_msg)
        return Command(
            update={
                "parsed_orders": [],
                "parse_errors": errors,
                "retry_count": state.get("retry_count", 0) + 1,
                "error": error_msg,
            },
            goto="output",
        )

    return Command(
        update={
            "parsed_orders": orders,
            "parse_errors": errors,
            "retry_count": state.get("retry_count", 0) + 1,
        },
        goto="filter_validate",
    )


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Filter and Validate
# ══════════════════════════════════════════════════════════════════════════════

def filter_validate_node(state: AgentState) -> dict:
    """
    Apply the FilterSpec against parsed orders deterministically.

    Calls apply_filters() from services/filters.py. This node contains
    no LLM calls — filtering is pure Python operating on typed Pydantic
    objects. An empty result is valid and not treated as an error.

    Reads:  state["parsed_orders"], state["filter_spec"]
    Writes: state["filtered_orders"]
    Routes: → output (always, fixed edge)
    """
    parsed_orders = state.get("parsed_orders", [])
    filter_spec   = state.get("filter_spec")

    if not parsed_orders:
        logger.warning("Filter node: no parsed orders to filter")
        return {"filtered_orders": []}

    if filter_spec is None:
        logger.warning(
            "Filter node: no filter spec present — returning all orders"
        )
        return {"filtered_orders": parsed_orders}

    filtered = apply_filters(parsed_orders, filter_spec)

    logger.info(
        f"Filter node: {len(parsed_orders)} orders in, "
        f"{len(filtered)} match filters"
    )

    return {"filtered_orders": filtered}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Output
# ══════════════════════════════════════════════════════════════════════════════

def output_node(state: AgentState) -> dict:
    """
    Build and return the final AgentResponse.

    Always runs regardless of whether earlier nodes succeeded or failed.
    If state["error"] is set, returns an error response with an empty
    orders list. Otherwise serializes filtered_orders, scores each one
    for anomalies, and wraps everything in an AgentResponse.

    Anomaly scoring is best-effort — if the model is not trained or
    scoring fails, orders are returned without anomaly data rather
    than failing the entire response.

    Reads:  state["filtered_orders"], state["query"],
            state["parse_errors"], state["error"]
    Writes: state["result"]
    Routes: → END (always)
    """
    query           = state.get("query", "")
    error           = state.get("error")
    filtered_orders = state.get("filtered_orders", [])
    parse_errors    = state.get("parse_errors", [])

    if error:
        logger.error(f"Output node: returning error response — {error}")
        response = AgentResponse(
            orders=[],
            query=query,
            total_found=0,
            flagged_count=0,
            error=error,
        )
        return {"result": response.model_dump()}

    # score each order for anomalies
    try:
        from ml.anomaly_detector import score_orders
        anomaly_scores = score_orders(filtered_orders)
        logger.info(
            f"Output node: scored {len(filtered_orders)} orders "
            f"for anomalies"
        )
    except Exception as e:
        logger.warning(f"Anomaly scoring unavailable: {e}")
        anomaly_scores = [{}] * len(filtered_orders)

    # merge order data with anomaly scores
    output_orders = []
    flagged_count = 0

    for order, anomaly in zip(filtered_orders, anomaly_scores):
        order_dict = order.to_output()
        if anomaly:
            order_dict.update(anomaly)
            if anomaly.get("is_flagged"):
                flagged_count += 1
        output_orders.append(order_dict)

    if parse_errors:
        logger.warning(
            f"Output node: {len(parse_errors)} parse errors "
            f"occurred during this run"
        )

    if flagged_count > 0:
        logger.warning(
            f"Output node: {flagged_count} orders flagged as "
            f"potentially anomalous"
        )

    response = AgentResponse(
        orders=output_orders,
        query=query,
        total_found=len(output_orders),
        flagged_count=flagged_count,
        error=None,
    )

    logger.info(
        f"Output node: returning {len(output_orders)} orders "
        f"for query '{query}' ({flagged_count} flagged)"
    )

    return {"result": response.model_dump()}