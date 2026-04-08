"""
agent/graph.py

Assembles and compiles the LangGraph StateGraph for the order query agent.
This file is purely structural — it imports node functions, registers them,
and defines the edges between them. No business logic lives here.

Graph topology:

    START
      │
      ▼
  query_planner ──(error)──────────────────────────────┐
      │                                                  │
      ▼                                                  │
  api_fetcher ──(error)────────────────────────────────┐│
      │                                                  ││
      ▼                                                  ││
  context_guard                                          ││
      │                                                  ││
      ▼                                                  ││
  llm_parser ──(no orders parsed)──────────────────────┐││
      │                                                  │││
      ▼                                                  │││
  filter_validate                                        │││
      │                                                  │││
      ▼                                                  │││
    output ◄───────────────────────────────────────────┘┘┘
      │
      ▼
     END

Nodes using Command-based routing (combined state update + routing):
    query_planner, api_fetcher, llm_parser

Nodes using fixed edges (always go to same next node):
    context_guard → llm_parser
    filter_validate → output
    output → END
"""

import logging
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    query_planner_node,
    api_fetcher_node,
    context_guard_node,
    llm_parser_node,
    filter_validate_node,
    output_node,
)

logger = logging.getLogger(__name__)


def build_graph():
    """
    Build and compile the order query agent graph.

    Registers all nodes and edges, then compiles into a runnable graph.
    The compiled graph is invoked with an initial AgentState containing
    the user's query and returns the final state including the output field.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("query_planner",   query_planner_node)
    builder.add_node("api_fetcher",     api_fetcher_node)
    builder.add_node("context_guard",   context_guard_node)
    builder.add_node("llm_parser",      llm_parser_node)
    builder.add_node("filter_validate", filter_validate_node)
    builder.add_node("output",          output_node)

    # ── Entry point ────────────────────────────────────────────────────────
    builder.add_edge(START, "query_planner")

    # ── Fixed edges ────────────────────────────────────────────────────────
    # These nodes always route to the same next node.
    # Nodes using Command-based routing (query_planner, api_fetcher,
    # llm_parser) do not need add_edge — their routing is internal.
    builder.add_edge("context_guard",   "llm_parser")
    builder.add_edge("filter_validate", "output")
    builder.add_edge("output",          END)

    # ── Compile ────────────────────────────────────────────────────────────
    graph = builder.compile()

    logger.info("Agent graph compiled successfully")
    return graph


# Build once at module load — imported and invoked by main.py
agent_graph = build_graph()