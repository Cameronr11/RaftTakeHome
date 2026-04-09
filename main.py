"""
main.py — single-command entry point for the Raft order-query agent.

Usage:
    python main.py "Show me all orders in Ohio over $500"
    python main.py --extended "Who bought a laptop?"

Flags:
    --extended  Use the 503-order synthetic dataset instead of the
                6-order Raft-provided dataset.

The agent will:
    1. Spawn the dummy customer API on port 5001 (if not already running)
    2. Extract filter intent from the natural-language query
    3. Fetch raw orders from the Flask API
    4. Parse unstructured text into structured Order objects via LLM
    5. Filter / sort results deterministically
    6. Return clean JSON output
"""

import sys
import json
import logging
import subprocess
import atexit
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run(query: str) -> dict:
    """Execute the LangGraph agent pipeline and return the result dict."""
    from agent.graph import agent_graph  # deferred so logging is configured first

    logger.info(f"Starting agent run for query: '{query}'")

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

    final_state = agent_graph.invoke(initial_state)
    result = final_state.get("result")

    if result is None:
        logger.error("Agent completed but output was not set in state")
        result = {
            "orders": [],
            "query": query,
            "total_found": 0,
            "error": "Agent completed without producing output",
        }

    return result


def _kill_port(port: int = 5001) -> None:
    """Kill whatever process is listening on the given port (Windows)."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pids.add(line.strip().split()[-1])
        for pid in pids:
            logger.info(f"Killing process {pid} on port {port}")
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", pid],
                capture_output=True, timeout=5,
            )
        if pids:
            time.sleep(1)
    except Exception as e:
        logger.warning(f"Could not kill process on port {port}: {e}")


def _spawn_api(extended: bool = False) -> None:
    """Kill any existing API on port 5001 and spawn the correct one.

    Imports the Flask ``app`` object directly (bypassing ``if __name__``
    blocks) so we run with ``debug=False``.  This avoids Flask's debug
    reloader, which forks an unkillable child process on Windows.
    """
    _kill_port(5001)

    module = "dummy_customer_extended_api" if extended else "dummy_customer_api"
    project_dir = str(Path(__file__).parent)

    logger.info(f"Spawning {module} on port 5001...")
    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, r'{project_dir}'); "
            f"from {module} import app; "
            f"app.run(host='0.0.0.0', port=5001, debug=False)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    time.sleep(2)
    logger.info(f"API subprocess started (pid={proc.pid})")


def main():
    """CLI entry point — parse args, spawn the API, run the agent."""
    extended = "--extended" in sys.argv
    if extended:
        sys.argv.remove("--extended")

    _spawn_api(extended=extended)

    if len(sys.argv) < 2:
        print('Usage: python main.py [--extended] "your query here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    try:
        result = run(query)
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(json.dumps({
            "orders": [],
            "query": query,
            "total_found": 0,
            "error": str(e),
        }, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()