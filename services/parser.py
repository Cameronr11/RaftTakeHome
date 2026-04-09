"""
services/parser.py

LLM-powered parser that converts raw unstructured order strings into
validated Order objects. This is the only file in the system that makes
LLM calls for data extraction — all other logic is deterministic Python.

The parser uses LangChain's with_structured_output() to bind the LLM
to the OrderExtract schema, forcing structured JSON responses that match
our Pydantic model. This is the primary mechanism for constraining the
LLM — it cannot return free-form text, only schema-conformant data.

After extraction, each OrderExtract is promoted to a full Order object
with the original raw string attached. This triggers the model_validator
in Order which cross-checks the extracted orderId against the source text
as a hallucination defense.

Failure handling:
    - Structural validation failures (wrong types, missing fields) retry
      once with the error appended to the prompt as feedback.
    - Hallucination failures (orderId not in source text) are hard rejected
      and never retried — the data is fundamentally untrustworthy.
    - After one retry, remaining failures are logged and skipped so one
      bad order string never crashes the entire pipeline.

Functions:
    parse_order(raw, retry_error)  -- Parse a single raw string into Order.
    parse_orders(raws)             -- Parse a list of raw strings, returning
                                     validated orders and any error messages.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from models.schemas import Order, OrderExtract

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_RETRIES = 1  # how many times to retry a failed parse before giving up

# ─── Custom Exception ─────────────────────────────────────────────────────────

class ParseError(Exception):
    """
    Raised when a raw order string cannot be parsed into a valid Order.
    Carries enough context for the caller to decide whether to retry
    or log and move on.

    Attributes:
        message   -- human readable description of what went wrong
        is_hallucination -- True if the failure was a hallucination detection,
                            False if it was a structural validation failure.
                            Callers should never retry hallucination failures.
    """
    def __init__(self, message: str, is_hallucination: bool = False):
        super().__init__(message)
        self.is_hallucination = is_hallucination


# ─── LLM and Chain Construction ───────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    """
    Construct the ChatOpenAI client pointed at OpenRouter.
    Reads configuration from environment variables loaded from .env.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b:exacto"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0,  # deterministic extraction — no creativity needed
    )


def _build_chain(llm: ChatOpenAI):
    """
    Bind the LLM to the OrderExtract schema and wrap it in a
    ChatPromptTemplate with system and human messages.

    The system prompt defines extraction rules and provides a concrete
    example. The human message template accepts two variables:
        raw_order  -- the unstructured order string to parse
        error      -- optional error from a previous failed attempt

    Field descriptions on OrderExtract are automatically included in
    the LLM prompt by with_structured_output — they work alongside
    the system prompt to constrain extraction behavior.

    Note: curly braces in the example JSON are escaped as {{ and }}
    to prevent ChatPromptTemplate from treating them as variables.

    Returns:
        A runnable chain: prompt | structured_llm
    """
    structured_llm = llm.with_structured_output(OrderExtract)

    system_prompt = """You are a precise data extraction engine. Your only \
function is to extract structured order information from raw unstructured text.
You do not summarize, explain, or add commentary of any kind.

Rules:
- Extract ONLY what is explicitly present in the text
- NEVER invent, infer, or assume values not in the source text
- state must always be exactly two uppercase letters e.g. "OH" never "Ohio"
- total must always be a plain positive number e.g. 742.10 never "$742.10"
- items must always be a list even if there is only one item
- orderId must be digits only e.g. "1001" never "Order 1001"
- If city cannot be found use an empty string
- If items cannot be found use an empty list

Example:
Input:
"Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, \
Total=$1299.99, Items: gaming pc, mouse"

Output:
{{
  "orderId": "1003",
  "buyer": "Mike Turner",
  "city": "Cleveland",
  "state": "OH",
  "total": 1299.99,
  "items": ["gaming pc", "mouse"]
}}"""

    human_prompt = """Extract the order information from this text:

{raw_order}{error_context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    return prompt | structured_llm


# Build once at module load — reused across all parse calls
_llm = _build_llm()
_chain = _build_chain(_llm)


# ─── Core Parsing Functions ───────────────────────────────────────────────────

def parse_order(raw: str, retry_error: Optional[str] = None) -> Order:
    """
    Parse a single raw unstructured order string into a validated Order.

    Sends the raw string through the LLM extraction chain, then promotes
    the resulting OrderExtract to a full Order with the raw source string
    attached. The Order model_validator fires at construction time and
    cross-checks the extracted orderId against the raw string — if the
    orderId does not appear in the source, the order is rejected as a
    hallucination.

    Args:
        raw         -- raw unstructured order string from the API
        retry_error -- if this is a retry attempt, the error message from
                       the previous failure. Appended to the prompt so the
                       model knows what went wrong and can correct it.

    Returns:
        Validated Order object with raw field populated.

    Raises:
        ParseError -- if extraction fails validation or hallucination check.
                      Check is_hallucination to decide whether to retry.
    """
    # build the error context block if this is a retry
    error_context = ""
    if retry_error:
        error_context = f"""

Previous attempt failed with this error:
{retry_error}

Please try again paying careful attention to the field format rules."""

    logger.info(f"Parsing order (retry={retry_error is not None}): {raw[:60]}...")

    try:
        # invoke the chain — returns an OrderExtract or raises
        extract: OrderExtract = _chain.invoke({
            "raw_order": raw,
            "error_context": error_context,
        })

        logger.debug(f"LLM extracted: orderId={extract.orderId}, "
                     f"state={extract.state}, total={extract.total}")

        # promote OrderExtract to Order with raw attached
        # this triggers model_validator which runs the hallucination check
        try:
            order = Order(
                orderId=extract.orderId,
                buyer=extract.buyer,
                city=extract.city,
                state=extract.state,
                total=extract.total,
                items=extract.items,
                raw=raw,
            )
        except ValidationError as e:
            # check if this is a hallucination failure specifically
            error_str = str(e)
            is_hallucination = "hallucination" in error_str.lower()
            if is_hallucination:
                logger.warning(f"Hallucination detected for order: {raw[:60]}...")
            raise ParseError(error_str, is_hallucination=is_hallucination)

        logger.info(f"Successfully parsed order {order.orderId}")
        return order

    except ValidationError as e:
        # raised by with_structured_output if the LLM response
        # doesn't match the OrderExtract schema at all
        logger.warning(f"Schema validation failed: {e}")
        raise ParseError(str(e), is_hallucination=False)


def parse_orders(raws: list[str]) -> tuple[list[Order], list[str]]:
    """
    Parse a list of raw order strings into validated Order objects.

    Processes each string individually so a single failure does not
    corrupt the rest of the batch. On structural validation failure,
    retries once with the error message appended to the prompt.
    Hallucination failures are never retried.

    Args:
        raws -- list of raw unstructured order strings from the API

    Returns:
        A tuple of (orders, errors) where:
            orders -- list of successfully validated Order objects
            errors -- list of error message strings for failed parses,
                      suitable for writing to AgentState.parse_errors
    """
    orders: list[Order] = []
    errors: list[str] = []

    for i, raw in enumerate(raws):
        logger.info(f"Processing raw order {i + 1}/{len(raws)}")
        try:
            order = parse_order(raw)
            orders.append(order)

        except ParseError as e:
            if e.is_hallucination:
                # hard reject — do not retry
                error_msg = f"Hallucination rejected for order string: '{raw[:60]}...'. Error: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

            else:
                # structural failure — retry once with error context
                logger.info(f"Retrying parse after structural failure: {e}")
                try:
                    order = parse_order(raw, retry_error=str(e))
                    orders.append(order)
                    logger.info("Retry succeeded")

                except ParseError as retry_e:
                    # retry also failed — log and move on
                    error_msg = (
                        f"Failed to parse order after {MAX_RETRIES} attempt(s). "
                        f"Raw: '{raw[:60]}...'. Final error: {retry_e}"
                    )
                    logger.error(error_msg)
                    errors.append(error_msg)

    logger.info(
        f"Parsing complete — {len(orders)} succeeded, {len(errors)} failed"
    )
    return orders, errors