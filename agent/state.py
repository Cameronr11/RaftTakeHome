"""
agent/state.py

This State.py file defines the working memory of the entire LangGraph pipeline. 
The agent's state persists across node boundaries — nodes don't talk to each other directly, 
they communicate by reading from and writing to this shared object. Without it, nodes would have to pass 
data directly to each other and errors would have to be handled in a spiderweb of function arguments. 
Instead, any node can set state['error'] and the graph routes globally to the output node. AgentState is a 
TypedDict rather than a Pydantic model because nodes return partial updates — only the fields they changed,
and Pydantic would require all fields present to validate, which would fail on every partial write

Fields:
    query           -- The original natural language query. Set once at
                       entry, never modified.

    filter_spec     -- Structured intent extracted from the query by the
                       query planner node. Consumed by the filter node.

    raw_orders      -- Raw unstructured strings returned by the Flask API.
                       Written by the API fetcher, read by context guard.

    parsed_orders   -- Validated Order objects produced by the LLM parser.
                       Read by the filter node.

    filtered_orders -- Final filtered and sorted Order objects. Written by
                       the filter node, read by the output node.

    parse_errors    -- Accumulates error messages from failed parse attempts
                       across retries. Uses a reducer so retries append
                       rather than overwrite.

    retry_count     -- Tracks how many parse attempts have been made this run.
                       Stored for observability only — routing is determined
                       by whether any orders were successfully parsed, not
                       by this value. The actual retry logic lives inside
                       parse_orders() in services/parser.py.

    error           -- Fatal error message. If set, the graph short-circuits
                       directly to the output node rather than continuing.
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from models.schemas import FilterSpec, Order


class AgentState(TypedDict):
    # Set once at entry, never changed
    query: str

    # Written by query planner, read by filter node
    filter_spec: Optional[FilterSpec]

    # Written by API fetcher, read by context guard
    raw_orders: list[str]

    # Written by LLM parser, read by filter node
    parsed_orders: list[Order]

    # Written by filter node, read by output node
    filtered_orders: list[Order]

    # Accumulates across retries — reducer appends rather than overwrites
    parse_errors: Annotated[list[str], operator.add]

    # How many times the parser has been retried this run
    retry_count: int

    # Fatal error — if set, graph routes directly to output
    error: Optional[str]

    # Written by output node — the final serialized AgentResponse
    result: Optional[dict]