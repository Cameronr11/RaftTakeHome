# Raft Order Intelligence Agent

A LangGraph-powered AI agent that accepts natural-language procurement queries, fetches unstructured order data from a REST API, parses and validates it using an LLM with structured outputs, applies deterministic Python filters, and returns clean JSON — with every result scored by a trained Isolation Forest for procurement fraud detection.

Built by Cameron Rader as a submission for the Raft AI Engineer take-home challenge.

---

## Quick Start

```bash
git clone https://github.com/Cameronr11/RaftTakeHome.git
cd RaftTakeHome
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

That's it. The agent, the API, and the UI all start with one command. A browser window opens automatically at `http://localhost:5000`.

> **Note:** The `.env` file has been excluded from github submission for safety reasons, however during Tuesday meeting can be committed for a brief period of time.
> To create your own, simply use the format below with an openrouter api key and the .env file format below:
  OPENROUTER_API_KEY=your_api_key
  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  MODEL_NAME=openai/gpt-oss-120b:exacto
  API_BASE_URL=http://localhost:5001 

---

## Run Modes

### Web UI — Datasets chosen by toggle (6 orders, 503 orders)
```bash
python main.py
```
Launches the Raft Order Intelligence UI at `http://localhost:5000`. Uses the 6-order baseline dataset provided in the challenge spec.

---

### CLI Mode — Single Query
```bash
python main.py "Show me all orders where the buyer was in Ohio and total was over 500"
python main.py "Who bought a laptop?"
python main.py "Show me the top 3 most expensive orders"
python main.py "Show me order 1003"
```
Runs a single query and prints clean JSON to stdout. No UI required.

---

### CLI Mode — Extended Dataset
```bash
python main.py --extended "Show me orders flagged as anomalous"
python main.py --extended "Find orders over $5000"
```
CLI query against the 503-order dataset with full anomaly scoring.

---

## What the Agent Does

1. **Parses natural language** — Extracts structured filter intent (state, price range, item keyword, buyer name, sort order, etc.) from the user's query using `with_structured_output(FilterSpec)`
2. **Fetches raw data** — Calls the Flask order API, which returns deliberately unstructured text in inconsistent formats
3. **Parses and validates** — LLM extracts each order into a typed Pydantic schema; a model validator cross-checks the extracted order ID against the raw source string to catch hallucinations at the code level
4. **Filters deterministically** — A pure Python layer applies the extracted filters (no LLM involved in filtering, sorting, or limiting)
5. **Scores for fraud** — Every returned order is scored by a pre-trained Isolation Forest. Suspicious orders are flagged with a score, category, and explanation
6. **Returns clean JSON** — Always returns a consistent `AgentResponse` envelope whether the query succeeded, returned zero results, or hit an error

### Example Output

```json
{
  "orders": [
    {
      "orderId": "1003",
      "buyer": "Mike Turner",
      "city": "Cleveland",
      "state": "OH",
      "total": 1299.99,
      "items": ["gaming pc", "mouse"],
      "anomaly_score": 0.312,
      "is_flagged": false,
      "category": "tech_equipment",
      "reason": "Anomaly score 0.312 is within normal range..."
    }
  ],
  "query": "Show me orders in Ohio",
  "total_found": 3,
  "flagged_count": 0,
  "error": null
}
```

---

## ML: Procurement Fraud Detection

The system includes a production-style anomaly detection pipeline built entirely offline — no LLM calls required at training time.

### Model
**Isolation Forest** (scikit-learn) trained semi-supervised on normal orders only. The model learns what legitimate procurement looks like; deviations at inference time produce elevated anomaly scores.

### Data Pipeline
1. `ml/data_generator.py` — Generates 503 synthetic orders via LLM (temp=0.9) in deliberately inconsistent formats, with ~10% containing anomalous price patterns
2. `ml/save_parsed_orders.py` — Runs all strings through the agent's LLM parser and serializes validated `Order` objects to JSON (run once; output committed)
3. `ml/anomaly_detector.py` — Stratified 80/20 train/test split, trains on normal orders only, evaluates on held-out test set

### Training Design Choices
- **Semi-supervised:** Model never sees anomalous orders during training — it learns the normal distribution and flags deviations
- **No leakage:** `StandardScaler` fitted on training data only; test set is strictly held out
- **Features:** `total`, `num_items`, and a 7-class one-hot category encoding (`tech_equipment`, `peripherals`, `networking`, etc.)
- **`price_ratio` excluded:** Deliberately omitted to avoid circular reasoning between how anomalies were generated and how they are detected
- **Threshold (0.60):** Set above the IF decision boundary (0.50) — requires meaningful confidence before flagging. Precision-first: the cost of a false positive is analyst review time

### Held-Out Test Set Performance
| Metric | Score |
|--------|-------|
| Precision | 0.909 |
| Recall | 1.000 |
| F1 | 0.952 |

> The trained model (`ml/anomaly_model.pkl`) is included in this submission. No training step is required to run the demo.

---

## Architecture

The agent is built as a deterministic LangGraph pipeline with six named nodes. For the full architecture diagram, node-by-node data flow, schema design, LLM constraint rationale, and edge case handling details:

**→ See [`architecture.md`](architecture.md)**

---

## Project Structure

```
RaftTakeHome/
├── main.py                          # Single entry point — starts everything
├── dummy_customer_api.py            # Raft-provided baseline API (6 orders)
├── dummy_customer_extended_api.py   # Extended API (503 orders, auto-spawned with --extended)
├── requirements.txt                 # Pinned dependencies
├── architecture.md                  # Full agent architecture and design notes
│
├── agent/
│   ├── graph.py     # StateGraph wiring — purely structural, no business logic
│   ├── nodes.py     # Six node functions + conditional routing
│   └── state.py     # AgentState TypedDict
│
├── models/
│   └── schemas.py   # OrderExtract, Order, FilterSpec, AgentResponse (Pydantic)
│
├── services/
│   ├── llm.py        # Single source of truth for the ChatOpenAI client
│   ├── api_client.py # HTTP wrapper with schema-drift detection
│   ├── parser.py     # LLM extraction, hallucination checks, retry logic
│   └── filters.py    # Deterministic filter, sort, and limit — no LLM
│
├── ml/
│   ├── data_generator.py      # Synthetic dataset generation
│   ├── save_parsed_orders.py  # Offline parse → JSON (run once)
│   ├── anomaly_detector.py    # Isolation Forest training and runtime scoring
│   ├── extended_orders.json   # 503 raw order strings
│   ├── parsed_orders.json     # Pre-parsed Order objects
│   ├── anomaly_labels.json    # Ground truth labels (evaluation only)
│   └── anomaly_model.pkl      # Trained model — included, no training needed
│
└── UI/
    ├── app.py                 # Flask frontend (factory pattern)
    └── templates/
        └── index.html         # Single-page UI
```

---

## Configuration

The `.env` file is included in this submission and is pre-configured. If you need to create it manually:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-oss-120b:exacto
API_BASE_URL=http://localhost:5001
```

---

## Requirements

- Python 3.10+
- All dependencies pinned in `requirements.txt`
- No Docker, no database, no external services beyond the OpenRouter API key
