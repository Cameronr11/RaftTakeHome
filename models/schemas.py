"""
models/schemas.py

Pydantic data models defining the contracts for the agent pipeline.

Classes:
    OrderExtract  -- Lightweight LLM-facing extraction model. Used exclusively
                     with with_structured_output(). No validators, no raw field.
                     Field descriptions are sent to the LLM as part of the prompt.

    Order         -- A single parsed customer order. Validates field types,
                     normalizes state codes, strips currency symbols from totals,
                     and cross-checks the extracted orderId against the raw source
                     string to catch LLM hallucinations.

    FilterSpec    -- Structured representation of a natural language query's intent.
                     Produced by the query planner and consumed by the filter layer
                     to deterministically select and sort matching orders.

    AgentResponse -- Final output envelope. Always returned regardless of success
                     or failure so the caller receives consistent JSON structure.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re


class OrderExtract(BaseModel):
    """
    Intermediate model used exclusively for LLM structured output extraction.
    Simpler than Order — no validators, no raw field, no hallucination check.
    The parser constructs a full Order from this result with raw attached.

    Field descriptions are automatically included in the LLM prompt by
    LangChain's with_structured_output — they are written for the model,
    not just for human readers.
    """
    orderId: str = Field(
        description="Numeric order ID digits only. E.g. '1001' not 'Order 1001'"
    )
    buyer: str = Field(
        description="Full buyer name exactly as written in the text"
    )
    city: Optional[str] = Field(
        default="",
        description="City name only without state. Empty string if not found."
    )
    state: str = Field(
        description="Two-letter uppercase state code only. E.g. 'OH' not 'Ohio'"
    )
    total: float = Field(
        description="Order total as positive float no currency symbols. E.g. 742.10 not $742.10"
    )
    items: list[str] = Field(
        default_factory=list,
        description="List of item name strings. Always a list even if one item. Empty list if not found."
    )


class Order(BaseModel):
    orderId: str
    buyer: str
    city: Optional[str] = None
    state: str
    total: float
    items: Optional[list[str]] = []
    raw: Optional[str] = None

    @field_validator("orderId")
    @classmethod
    def order_id_must_be_numeric_string(cls, v):
        v = v.strip()
        if not v.isdigit():
            raise ValueError(f"orderId '{v}' does not look like a real order ID")
        return v

    @field_validator("state")
    @classmethod
    def state_must_be_two_letters(cls, v):
        v = v.strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(f"state '{v}' is not a valid two-letter code")
        return v

    @field_validator("total", mode="before")
    @classmethod
    def parse_total(cls, v):
        if isinstance(v, str):
            cleaned = re.sub(r"[^\d.]", "", v)
            if not cleaned:
                raise ValueError(f"Cannot parse total from '{v}'")
            return float(cleaned)
        if v <= 0:
            raise ValueError(f"total {v} must be positive")
        return round(float(v), 2)

    @model_validator(mode="after")
    def verify_order_id_in_raw(self):
        if self.raw and self.orderId not in self.raw:
            raise ValueError(
                f"orderId '{self.orderId}' not found in source text — possible hallucination"
            )
        return self

    def to_output(self) -> dict:
        return {
            "orderId": self.orderId,
            "buyer": self.buyer,
            "city": self.city,
            "state": self.state,
            "total": self.total,
            "items": self.items
        }


class FilterSpec(BaseModel):
    state: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    item_keyword: Optional[str] = None
    buyer_name: Optional[str] = None
    limit: Optional[int] = None
    sort_by: Optional[str] = None      # "total", "buyer", "orderId"
    sort_order: Optional[str] = None   # "asc" or "desc", None means unsorted

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v):
        if v is None:
            return v
        return v.strip().upper()


class AgentResponse(BaseModel):
    orders: list[dict]
    query: str
    total_found: int
    error: Optional[str] = None
    flagged_count: int = 0