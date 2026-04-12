"""
UI/app.py — Lightweight Flask frontend for the Raft Order Intelligence Agent.

Serves a single-page Raft-themed UI at ``/`` and exposes ``POST /api/query``
for the JavaScript client.  All heavy lifting is delegated to ``main.run()``.
"""

import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify

_DEMO_PATH = Path(__file__).parent.parent / "ml" / "data" / "demo_orders.json"


def _demo_order_count() -> int:
    """Return the number of orders in the demo dataset, or 0 if not generated yet."""
    try:
        return len(json.loads(_DEMO_PATH.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def create_app(extended: bool = False) -> Flask:
    """Factory that returns a configured Flask app.

    Args:
        extended: initial dataset mode (mirrors the --extended CLI flag).
    """
    app = Flask(__name__)
    app.config["EXTENDED"] = extended

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            extended=app.config["EXTENDED"],
            extended_count=_demo_order_count(),
        )

    @app.route("/api/query", methods=["POST"])
    def query_endpoint():
        from main import run, _spawn_api

        data = request.get_json(silent=True) or {}
        query_text = data.get("query", "").strip()
        use_extended = bool(data.get("extended", app.config["EXTENDED"]))

        if not query_text:
            return jsonify({
                "orders": [], "query": "", "total_found": 0,
                "error": "Please enter a query.", "flagged_count": 0,
            }), 400

        if use_extended != app.config["EXTENDED"]:
            _spawn_api(extended=use_extended)
            app.config["EXTENDED"] = use_extended

        try:
            result = run(query_text)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                "orders": [], "query": query_text, "total_found": 0,
                "error": str(e), "flagged_count": 0,
            }), 500

    return app
