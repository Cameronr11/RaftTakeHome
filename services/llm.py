"""
services/llm.py

Single source of truth for constructing the ChatOpenAI client used by
the agent pipeline. Both agent/nodes.py and services/parser.py import
build_llm() from here so model name, base URL, and temperature cannot
silently diverge between the planner and the parser.

Note: ml/data_generator.py has its own _build_llm() intentionally —
it runs at temperature=0.9 for creative data generation and has no
default model fallback. It is not affected by this module.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def build_llm() -> ChatOpenAI:
    """
    Construct the ChatOpenAI client pointed at OpenRouter.
    Reads configuration from environment variables loaded from .env.

    Returns:
        Configured ChatOpenAI instance with temperature=0 for
        deterministic extraction.
    """
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b:exacto"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0,
    )
