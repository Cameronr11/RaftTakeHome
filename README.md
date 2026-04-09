# Procurement order agent — Raft AI engineer challenge

This repository is a submission for Raft’s AI engineer coding challenge: a **LangGraph** agent that accepts natural language, calls a dummy Flask API returning **unstructured** order text, uses an LLM to recover **structured** records, and returns **clean JSON**. The baseline problem is messy-ingestion plus reliable filtering; the submission also adds **semi-supervised anomaly detection** so every result set can surface orders that look like procurement fraud (for example, a mouse pad priced like specialized hardware).

---

## Overview

Government-scale procurement often means millions of legacy records that never agreed on one string format. The agent mirrors that reality: the LLM is responsible only for **interpretation** (query intent and per-order parsing). **Filtering, sorting, and limiting** are ordinary Python on typed models, so the same query yields the same structured outcome regardless of non-determinism in the model. **Anomaly scoring** uses a trained Isolation Forest in the output step (deterministic given the model artifact) and is best-effort if the model is missing.

At runtime the agent talks to the **original** `dummy_customer_api.py` from the challenge. A separate extended dataset and API exist only for training and evaluation of the fraud-detection model; the canonical API file is treated as fixed.

---

## Quick start (under five minutes)

**Prerequisites:** Python 3.10+, an [OpenRouter](https://openrouter.ai/) API key. The specified model `openai/gpt-oss-120b:exacto` works on the free tier; if you need a key, Raft suggests contacting `mlteam@teamraft.com`.

```bash
python -m venv RaftProject
# Windows:
.\RaftProject\Scripts\activate
# macOS/Linux:
# source RaftProject/bin/activate

pip install -r requirements.txt
```

Set `OPENROUTER_API_KEY` in a `.env` file at the repo root (or export it in your shell). Optional: `OPENROUTER_BASE_URL` if you use a custom gateway.

**Terminal 1 — API**

```bash
python dummy_customer_api.py
```

**Terminal 2 — query**

```bash
python main.py "Show me all orders in Ohio over $500"
python main.py "Who bought a laptop?"
python main.py "Show me the top 2 most expensive orders"
```

The Ohio / $500 example is the clearest **anomaly demo**: order **1006** (mouse pad at **$8,750**) should appear in the result set and be flagged for review.

---

## Architecture overview

The agent is a compiled **LangGraph** graph: each transition is named, and important paths log state movement. That matters in regulated settings—you can point to *which* step produced *which* decision. The LLM is constrained with **LangChain `with_structured_output()`** against **Pydantic** schemas; validators enforce plausible fields (for example, `orderId` must appear in the original raw string, which blocks a common hallucination class). A fuller text diagram and node table live in [`architecture.md`](architecture.md); wiring is in `agent/graph.py` and `agent/nodes.py`.

---

## Above and beyond: fraud-oriented anomaly detection

Retrieval answers *what matches the query*; the ML layer asks *which of those matches look wrong*. The scenario is deliberately grounded in procurement fraud patterns—ordinary catalog items with systematically inflated totals. The detector is an **Isolation Forest** trained in a **semi-supervised** way: **only normal orders** were used for fitting, which matches production where labeled fraud is scarce. **`price_ratio` is not used as a feature**, so the model is not tuned to echo how synthetic anomalies were generated. Training uses a **stratified 80/20 train/test split** so the roughly **10%** anomaly rate is preserved in both halves.

**Held-out evaluation** (503 orders: 453 normal, 50 anomalous, **9.9%** rate; train/test stratified; Isolation Forest fit on **362** normal training rows only):

| Metric | Value |
|--------|--------|
| Precision | **0.909** |
| Recall | **1.000** |
| F1 | **0.952** |
| False negatives (test) | **0** |

---

## Design decisions

**LangGraph instead of a single chain** — The graph makes control flow explicit and loggable. For defense-adjacent workflows, “we ran a script” is weaker than “we transitioned through these named steps with this state.”

**Two LLM roles only** — (1) `query_planner_node`: natural language → `FilterSpec`. (2) `llm_parser_node`: raw string → `Order`. Everything downstream is deterministic Python, which is the main lever for **correctness and determinism** in the final JSON.

**Structured output + Pydantic** — Binding the model to schemas prevents free-form replies; validators catch bad IDs, states, and totals before filtering runs.

**Hallucination defense** — Each `Order` retains the source `raw` text; a model validator requires the extracted `orderId` to appear in that string. On a 503-order parse experiment, **four** invented IDs were attempted and **all** were rejected at validation time.

**Deterministic filtering** — `FilterSpec` is applied with typed `Order` objects and set/logic-style operations in Python, not by asking the model to “filter again.”

**Extended data without touching the supplied API** — Synthetic data and labels live under `ml/`; the extended Flask server is optional. The challenge `dummy_customer_api.py` stays the contract for normal runs.

---

## Running the ML pipeline (optional)

Generate the 503-order synthetic set (four string format families to stress the parser), parse it through the same LLM pipeline used by the agent, then train and evaluate the Isolation Forest:

```bash
python ml/data_generator.py
python -m ml.save_parsed_orders
python -m ml.anomaly_detector
```

To serve the extended JSON over HTTP (for experiments only):

```bash
python dummy_customer_extended_api.py
```

---

## Project structure

```
.
├── main.py                      # CLI entry: python main.py "<query>"
├── dummy_customer_api.py        # Original Raft API (unstructured text)
├── dummy_customer_extended_api.py  # Optional: serves ml/extended_orders.json
├── architecture.md              # Text diagram + node summary
├── agent/
│   ├── graph.py                 # LangGraph topology and compilation
│   ├── nodes.py                 # Planner, fetch, parse, filter, output
│   ├── state.py                 # Agent state type
│   └── tools.py                 # Tool definitions used by the graph
├── models/
│   └── schemas.py               # Order, FilterSpec, structured LLM contracts
├── services/
│   ├── api_client.py            # HTTP client for the dummy API
│   ├── filters.py               # Deterministic FilterSpec application
│   └── parser.py                # LLM parsing helpers
└── ml/
    ├── data_generator.py        # Builds extended_orders.json + labels
    ├── extended_orders.json     # Synthetic raw strings (committed)
    ├── anomaly_labels.json      # Ground truth for ML eval (not exposed via API)
    ├── save_parsed_orders.py    # Batch parse → parsed_orders.json
    ├── parsed_orders.json       # Parsed orders for training (generated)
    └── anomaly_detector.py      # Train Isolation Forest, print metrics
```

---

## Original challenge (summary)

Raft’s prompt asked for a real agent: natural language in → call messy API → LLM structuring → JSON out, with attention to context limits, hallucinations, and schema drift; **LangChain or LangGraph**; model **`openai/gpt-oss-120b:exacto`** via OpenRouter; logging and error handling; one-command run (`python main.py`). This implementation meets those constraints and adds the traditional ML layer described above.
