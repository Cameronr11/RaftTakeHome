"""
main.py — single-command entry point for the Raft order-query agent.

Usage:
    python main.py                                   Launch the web UI
    python main.py --extended                        Web UI + extended dataset
    python main.py "Show me all orders over $500"    CLI mode (JSON output)
    python main.py --extended "Who bought a laptop?" CLI mode + extended dataset

Flags:
    --extended  Use the ~200-order demo dataset instead of the
                6-order Raft-provided dataset.
"""

import sys
import json
import logging
import subprocess
import atexit
import time
from pathlib import Path

import psutil

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
    """Kill whatever process is listening on the given port (cross-platform).

    Uses psutil so this works identically on Windows, Linux, and macOS.
    On Linux, psutil.net_connections() may require elevated privileges when
    inspecting connections owned by other users — for self-spawned processes
    (which is all this function is ever called for) standard user permissions
    are sufficient.
    """
    killed = False
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                try:
                    proc = psutil.Process(conn.pid)
                    logger.info(
                        f"Killing process {conn.pid} ({proc.name()}) on port {port}"
                    )
                    proc.kill()
                    killed = True
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.warning(f"Could not kill pid {conn.pid} on port {port}: {e}")
    except Exception as e:
        logger.warning(f"Could not enumerate connections on port {port}: {e}")
    if killed:
        time.sleep(1)


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


def _launch_ui(extended: bool = False) -> None:
    """Start the Raft Order Intelligence web UI on port 5000."""
    import threading
    import webbrowser
    from UI.app import create_app

    _kill_port(5000)

    app = create_app(extended=extended)
    from UI.app import _demo_order_count
    demo_count = _demo_order_count() if extended else 0
    dataset = f"Extended ({demo_count} orders)" if extended else "Default (6 orders)"

    print(f"\n  ( RAFT ) Order Intelligence Agent")
    print(f"  Dataset : {dataset}")
    print(f"  UI      : http://localhost:5000\n")

    threading.Timer(1.5, webbrowser.open, args=["http://localhost:5000"]).start()
    app.run(host="0.0.0.0", port=5000, debug=False)


def main():
    """CLI entry point — parse args, spawn the API, then launch UI or run a query."""
    extended = "--extended" in sys.argv
    if extended:
        sys.argv.remove("--extended")

    _spawn_api(extended=extended)

    if len(sys.argv) < 2:
        _launch_ui(extended=extended)
        return

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