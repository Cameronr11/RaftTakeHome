"""
models/schemas.py

Pydantic data models defining the contracts for the agent pipeline.

Classes:
    OrderExtract: Forced Contract for 2nd LLM call
    Order: Internal, fully validated record created using the OrderExtract class.
    FilterSpec: Forced Contract for 1st LLM call
    AgentResponse: Final agentic response object
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re


class OrderExtract(BaseModel):
    """
    This OrderExtract class is created to structure the output of the second LLM call in this system. 
    This second LLM response object takes in the raw unstructured order string given by the dummy api. 
    This happens for every order in the api response, essentially to parse the orders and gather
    the relevant order fields into a structured object which get turned into order objects and then compared 
    against the users query intent gathered using the first LLM call and FilterSpec object.
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
    """
    The order class is the product of the LLM's response it created using the OrderExtract class plus the raw api string attached to it. 
    The construction fo this order class is created by the parser. The raw string is attached soley to check for hallucinations using the validator,
    using verify_order_id_in_raw (@model_validator(mode="after")). Raw is then stripped by to_output() and never 
    reaches the response.

    raw string (from API)  →  LLM reads it  →  OrderExtract (LLM's output)  →  Order(fields from OrderExtract + raw=raw string)
    """
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
        if self.raw and not re.search(r'\b' + re.escape(self.orderId) + r'\b', self.raw):
            raise ValueError(
                f"orderId '{self.orderId}' not found as discrete token in source text — possible hallucination"
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
    """
    This is class is used as the structure for the LLM output of the first of two LLM calls in this system.
    This first LLM call is done by the Query Planner. Query Planner node uses _llm.with_structured_output(FilterSpec) 
    to convert the user's query into a FilterSpec object. The FilterSpec class is used to get a snapshot of the user's 
    query intent. Every field is optional but is populated if that an object of that field is mentioned in the query. Ex. Ohio -> state. "
    This FilterSpec gets used by 2 nodes, "api_fetcher_node" and "filter_validate_node". Api Fetcher decides if the user wants 1 specific
    order, or a cluster of orders. The filter validate node worries about all of the different fields the user provided i.e state, total, item, etc.

    """
    state: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    item_keyword: Optional[str] = None
    buyer_name: Optional[str] = None
    limit: Optional[int] = None
    sort_by: Optional[str] = None      # "total", "buyer", "orderId"
    sort_order: Optional[str] = None   # "asc" or "desc", None means unsorted
    order_id: Optional[str] = Field(
        default=None,
        description=(
            "Exact numeric order ID when the user asks for a specific order. "
            "Digits only — e.g. '1003' not 'Order 1003'. "
            "Only populate when the query names a single specific order ID."
        ),
    )

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v):
        if v is None:
            return v
        return v.strip().upper()

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, v):
        if v is None:
            return v
        digits = re.sub(r"\D", "", v)
        if not digits:
            raise ValueError(f"order_id '{v}' contains no digits")
        return digits


class AgentResponse(BaseModel):
    """
    This is the final response object provided by the full agentic LangGraph pipeline. This node is always produced even if a segment
    of the pipelin fails. If the error is set, the response object will be empty and the error will be returned to the user.
    Along with all the orders that have been found and filtered, we also provide a anamoly flag based on the Isolation Forest model.

    """
    orders: list[dict]
    query: str
    total_found: int
    error: Optional[str] = None
    flagged_count: int = 0