# Architecture

**In one sentence:** A LangGraph agent accepts natural-language procurement queries, fetches unstructured order text from a Flask API, uses an LLM to parse into validated Pydantic objects, applies deterministic Python filters, scores each row with an Isolation Forest for procurement-style fraud, and returns clean JSON.

---

## LangGraph flow (six nodes)

```
START
  │
  ▼
query_planner_node
  │ LLM: with_structured_output(FilterSpec)
  │ Extracts: state, min_total, max_total, item_keyword,
  │           buyer_name, limit, sort_by, sort_order, order_id
  │
  ├──(out-of-scope query)──► output_node ──► END
  │
  ▼
api_fetcher_node
  │ HTTP GET → Flask API (localhost:5001)
  │ If order_id set: calls single-order endpoint
  │ Otherwise: fetches full order list
  │
  ├──(API error / order not found)──► output_node ──► END
  │
  ▼
context_guard_node
  │ Token estimate (~4 chars/token); truncate if > 12,000 tokens
  │ (6-order default set does not hit this; extended ~200-order demo set does.)
  │
  ▼
llm_parser_node
  │ LLM: with_structured_output(OrderExtract) per order
  │ Promote → Order with raw source attached
  │ Hallucination check: orderId must appear in source string
  │ Retry once on structural failure; hard reject on hallucination
  │
  ├──(all orders failed)──► output_node ──► END
  │
  ▼
filter_validate_node
  │ Pure Python — no LLM
  │ Apply FilterSpec: state, totals, item keyword, buyer, sort, limit
  │
  ▼
output_node
  │ Isolation Forest scores on each filtered order
  │ Merge scores; build AgentResponse (+ flagged_count)
  │
  ▼
END → JSON
```

---

## Agent state

`AgentState` (TypedDict) is the graph payload:

| Field | Role |
|--------|------|
| `query` | Original NL query; set once, not mutated |
| `filter_spec` | `FilterSpec` from the planner |
| `raw_orders` | Raw strings from the API |
| `parsed_orders` | Validated `Order` instances |
| `filtered_orders` | After filter/sort/limit |
| `parse_errors` | Accumulates across retries (`operator.add`) |
| `retry_count` | Parser retry counter |
| `error` | Fatal condition → short-circuit to `output_node` |

---

## Schema layer (Pydantic)

**OrderExtract** — LLM-facing extraction target for `with_structured_output()`: `orderId`, `buyer`, `city`, `state`, `total`, `items`. Field descriptions ride into the model prompt. No validators: the model should only fill slots, not enforce business rules.

**Order** — Internal, fully validated record. Validators enforce numeric `orderId`, two-letter `state`, currency-stripped `total`, and **hallucination defense**: promoted from `OrderExtract` only if `orderId` appears in the stored raw source string (code-level check, not prompt pleading). `to_output()` strips internals before serialization.

**FilterSpec** — Structured intent from the planner: optional `state`, `min_total`, `max_total`, `item_keyword`, `buyer_name`, `limit`, `sort_by`, `sort_order`, `order_id`. When `order_id` is set the API fetcher calls the single-order endpoint directly. Empty spec means return everything parsed.

**AgentResponse** — Uniform envelope for success and failure; includes `flagged_count` from the anomaly detector.

Illustrative shapes:

```python
from typing import Optional  # illustrative

# OrderExtract — extraction only (see models/schemas.py for full fields)
class OrderExtract(BaseModel):
    orderId: str
    buyer: str
    # city, state, total, items — all described via Field() for the LLM

# FilterSpec — planner output, filter input (all optional)
class FilterSpec(BaseModel):
    state: Optional[str] = None
    min_total: Optional[float] = None
    # item_keyword, buyer_name, limit, sort_by, sort_order, order_id
```

---

## Where the LLM is used (and is not)

| Step | LLM? | Rationale |
|------|------|-----------|
| Query → `FilterSpec` | Yes | NL is ambiguous; structured output pins intent |
| API fetch | No | HTTP client |
| Context guard | No | Token math + truncation |
| Raw string → `Order` | Yes | Unstructured text; resilient to format drift |
| Filter / sort / limit | **No** | Deterministic, testable, no non-reproducible "SQL in prose" |
| Anomaly scoring | No | Loaded sklearn models |

**Why filtering is Python, not LLM:** The user's constraints become executable predicates. Same `FilterSpec` + same `parsed_orders` always yield the same subset and ordering — critical for correctness reviews and regression tests.

**Why `OrderExtract` != `Order`:** The LLM gets a flat, prompt-friendly schema; the app owns validation, normalization, and anti-hallucination rules in one place (`Order`), without polluting the extraction prompt with validator error messages.

---

## ML layer (offline training, runtime inference)

**Isolation Forest (semi-supervised)** — Trained on **normal orders only** (~362 of the training split), mirroring production where fraud labels are scarce. Features: `total`, `num_items`, category one-hot (7 categories). **`price_ratio` is excluded** to avoid circularity with synthetic data generation. Flag when `anomaly_score > 0.6`.

A linear regression was evaluated as an additional "expected total" baseline but was removed — sparse training data for the "other" category produced nonsensical predictions (e.g. $7,600 for a coffee maker). The Isolation Forest score is the sole detection mechanism.

**Offline pipeline (no LLM at train time):** `data_generator.py` → synthetic strings → `save_parsed_orders.py` (through the agent parse path) → JSON → `anomaly_detector.py` (stratified 80/20 split, train on normal only, evaluate on held-out test set).

**Held-out test metrics (no leakage):** precision 0.909, recall 1.000, F1 0.952.

---

## End-to-end data flow

```
User NL query
      │
      ▼
  main.py
      │
      ▼
  agent/graph.py  (compiled StateGraph)
      │
      ├─► query_planner_node ──► LLM (OpenRouter)
      │              │
      │              ▼ FilterSpec
      │
      ├─► api_fetcher_node ──► Flask dummy API (:5001)
      │              │
      │              ▼ list[str] raw orders
      │
      ├─► context_guard_node (Python)
      │              │
      │              ▼ list[str] (maybe truncated)
      │
      ├─► llm_parser_node ──► LLM (OpenRouter), per order
      │              │
      │              ▼ list[Order]
      │
      ├─► filter_validate_node (Python)
      │              │
      │              ▼ list[Order] filtered + sorted
      │
      └─► output_node ──► ml/anomaly_detector.py
                 │
                 ▼
           AgentResponse JSON
           { orders [...scores...], total_found, flagged_count, query, error }
```

---

## Raft spec edge cases

**Context window overflow** — `context_guard_node` estimates tokens before parsing and truncates the order list to a 12,000-token budget. Each order is parsed in its own structured-output call so no single invocation scales with unbounded concatenated context.

**Model hallucination** — After `OrderExtract` is produced, promotion to `Order` runs a **model validator** that requires `orderId` to appear in the original raw string. Mismatches are rejected before filtering. (Example: 4 hallucination attempts observed over a full extended-dataset run; all rejected.)

**Unpredictable API / text formats** — Parsing is LLM-based rather than brittle regex. The extended synthetic set deliberately mixes formats (e.g. standard `Buyer=` lines, pipe-separated rows, `Amount=` / `Products:` variants, `Customer=` instead of `Buyer=`); all were parsed successfully in testing.

---

## Repository layout

```
RaftTakeHome/
├── main.py                          # Entry: python main.py "<query>"
├── dummy_customer_api.py            # Raft-provided API (baseline, 6 orders)
├── dummy_customer_extended_api.py   # Extended API serving demo dataset (~200 orders)
├── requirements.txt                 # Pinned dependencies
├── .env                             # API keys (local, not committed)
├── architecture.md                  # This document
├── README.md                        # Setup and usage
│
├── agent/
│   ├── graph.py     # StateGraph wiring + compile
│   ├── nodes.py     # Six nodes + conditional routing
│   └── state.py     # AgentState TypedDict
│
├── models/
│   └── schemas.py   # OrderExtract, Order, FilterSpec, AgentResponse
│
├── services/
│   ├── llm.py        # Single source of truth for the ChatOpenAI client
│   ├── api_client.py # HTTP client for Flask with schema-drift detection
│   ├── parser.py     # LLM extraction + hallucination checks / retries
│   └── filters.py    # Deterministic filter, sort, limit
│
└── ml/
    ├── data_generator.py      # Synthetic orders (LLM-assisted generation)
    ├── save_parsed_orders.py  # Offline parse → JSON
    ├── anomaly_detector.py    # Isolation Forest training and runtime scoring
    ├── extended_orders.json   # 503 raw strings
    ├── parsed_orders.json     # Pre-parsed orders
    ├── anomaly_labels.json    # Ground truth for evaluation
    └── anomaly_model.pkl      # Trained model — included, no training needed
```

Logging and error handling live at node boundaries and in services so failures (API down, parse exhaustion, planner "out of scope") still terminate in `output_node` with a consistent `AgentResponse` shape.
