"""
main.py

Entry point for the Raft order query agent.
Run with: python main.py "your natural language query here"

Examples:
    python main.py "Show me all orders in Ohio over $500"
    python main.py "Who bought a laptop?"
    python main.py "Show me the top 3 most expensive orders"

The agent will:
    1. Extract filter intent from the query
    2. Fetch raw orders from the Flask API
    3. Parse unstructured text into structured Order objects
    4. Filter and sort results deterministically
    5. Return clean JSON output
"""

import sys
import json
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
# Configure before any imports that use logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run(query: str) -> dict:
    """
    Run the order query agent with a natural language query.

    Args:
        query -- natural language query string

    Returns:
        dict containing the AgentResponse output
    """
    # import here so logging is configured first
    from agent.graph import agent_graph

    logger.info(f"Starting agent run for query: '{query}'")

    # build initial state — only query is set, everything else defaults
    initial_state = {
        "query": query,
        "filter_spec": None,
        "raw_orders": [],
        "parsed_orders": [],
        "filtered_orders": [],
        "parse_errors": [],
        "retry_count": 0,
        "error": None,
        "result": None,
    }

    # invoke the graph
    final_state = agent_graph.invoke(initial_state)

    # extract the output written by output_node
    result = final_state.get("result")

    if result is None:
        logger.error("Agent completed but output was not set in state")
        result = {
            "orders": [],
            "query": query,
            "total_found": 0,
            "error": "Agent completed without producing output"
        }

    return result


def main():
    """
    CLI entry point. Reads query from command line arguments.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your query here\"")
        print("Example: python main.py \"Show me all orders in Ohio over $500\"")
        sys.exit(1)

    # join all args after script name to allow queries without quotes
    query = " ".join(sys.argv[1:])

    try:
        result = run(query)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        error_output = {
            "orders": [],
            "query": query,
            "total_found": 0,
            "error": str(e)
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()