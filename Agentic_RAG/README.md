# Agentic RAG on AWS Lambda + OpenAI

A production-ready **Agentic Retrieval-Augmented Generation** system that runs on AWS Lambda. The agent uses OpenAI function/tool calling to iteratively search a FAISS knowledge base before synthesising a final answer.

Includes a **hybrid evaluation framework** (RAGAS-style metrics, LLM-as-a-Judge, human review flagging, and user feedback) that runs asynchronously in a separate Lambda — so evaluation never adds latency to the user-facing response.

---

## Agentic Loop

Unlike a fixed pipeline, the LLM **decides** when and how many times to search:

```
START
  │
  ▼
[LLM decides: do I need to search?]
  │
  ├── YES ──► search_knowledge_base(query)
  │                  │
  │           retriever.py → FAISS → top-k chunks
  │                  │
  │           chunks fed back to LLM
  │                  │
  └── (repeat up to 3 rounds)
  │
  ├── LLM has enough context ──► generate final answer ──► END
  │
  └── max rounds reached ──► force final answer ──► END
```

```
Agent state per round
├── messages:         list   # full conversation + tool results
├── tool_rounds_used: int    # how many search rounds were used (1–3)
└── all_documents:    list   # every chunk retrieved across all rounds
                             # → passed to async evaluator
```

The key difference from `RAG_Agent_LangGraph`: **the LLM controls the flow**.
It chooses the search query, decides whether one search is enough, and can
rephrase and search again if the first result was insufficient.

---

## Evaluation Framework

Evaluation is **fully async** — the main Lambda delivers the answer to the user
first, then fires the `EvaluatorFunction` in the background
(`InvocationType=Event`, fire-and-forget). The user never waits for metrics.

### Full system flow

```
User
 │
 │  POST /query
 ▼
AgenticRAGFunction  (main Lambda — fast path)
 │
 ├── Round 1: LLM → search_knowledge_base() → chunks
 ├── Round 2: LLM → search_knowledge_base() → more chunks  (if needed)
 ├── Round 3: LLM → search_knowledge_base() → more chunks  (if needed)
 │
 ├── LLM generates final answer
 │
 ├──► delivers answer + response_id to user immediately  (~5–15 seconds)
 │
 └──► async invoke ──────────────────────────────────────────────────────────┐
                                                                             │
                                                     EvaluatorFunction       │
                                                     (background)            │
                                                                             │
                                               run_evaluation()              │
                                                 │                           │
                                                 ├── Layer 1                 │
                                                 │   faithfulness            │
                                                 │   answer_relevance        │
                                                 │   context_precision       │
                                                 │                           │
                                                 └── Layer 2                 │
                                                     LLM judge (gpt-4o)     │
                                                     correctness             │
                                                     completeness           │
                                                     groundedness           │
                                                     clarity                │
                                                       │                    │
                                                       ▼                    │
                                               + tool_rounds_used           │
                                               (Agentic_RAG extra metric)   │
                                                       │                    │
                                                       ▼                    │
                                               DynamoDB log                 │
                                               (AgenticRAGEvaluationLogs)   │
                                                       │                    │
                                                passes threshold?           │
                                                 │           │              │
                                                YES          NO             │
                                                 │           │              │
                                                 │    flag_for_review       │
                                                 │    (Layer 3)             │
                                                 └───────────┘ ────────────┘

User
 │
 │  POST /feedback
 ▼
AgenticRAGFunction
 │
 └──► store_feedback() ──► AgenticRAGUserFeedback (DynamoDB)   ← Layer 4
                                    │
                             thumbs_down?
                                    │
                             flag_for_review (Layer 3)
```

### The four evaluation layers

#### Layer 1 — RAGAS-style Metrics

Automated metrics that require no ground truth. Computed on every response.
All chunks retrieved across **all tool-call rounds** are passed as context.

| Metric | What it measures | How |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | Extract claims → verify each against all retrieved chunks |
| **Answer Relevance** | Does the answer actually address the question? | LLM rates 0.0–1.0 |
| **Context Precision** | Are the retrieved chunks genuinely useful? | LLM checks each chunk across all rounds |

#### Layer 2 — LLM-as-a-Judge

A stronger model (`gpt-4o`) scores the answer on four dimensions.
Using the same model as the generator (`gpt-4o`) could introduce self-serving
bias — in production, consider using a different model family as judge.

| Dimension | What it measures | Scale |
|---|---|---|
| **Correctness** | Is the answer factually accurate? | 1–5 |
| **Completeness** | Does it fully answer the question? | 1–5 |
| **Groundedness** | Is every claim backed by the retrieved context? | 1–5 |
| **Clarity** | Is it well-written and easy to understand? | 1–5 |

`judge_score` = mean of the four dimensions.

#### Layer 3 — Human Review Flagging

Responses are automatically flagged for human review when:
- Any Layer 1 or Layer 2 metric falls below its configured threshold
- A user submits a thumbs-down (see Layer 4)

Flagged records appear in `AgenticRAGEvaluationLogs` with
`flagged_for_review = true` and a `flag_reason` — giving reviewers a
unified queue without a separate system.

#### Layer 4 — User Feedback

Users submit thumbs-up or thumbs-down via `POST /feedback`.
Thumbs-down automatically triggers a Layer 3 human review flag.

```bash
curl -X POST https://<ApiUrl>/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "<uuid from /query response>",
    "rating": "thumbs_down",
    "comment": "Answer missed the main point"
  }'
```

### Agentic_RAG extra metric: `tool_rounds_used`

Because the agent dynamically decides how many times to search, we track
the number of tool-call rounds per response. This is stored in every
evaluation record alongside the standard metrics.

| Value | Interpretation |
|---|---|
| `1` | Agent found enough context in one search — efficient |
| `2` | Needed a follow-up search — normal |
| `3` | Hit the max rounds — may indicate retrieval quality issues or a complex question |

Monitoring this over time helps you tune `MAX_TOOL_ROUNDS` and `TOP_K`.

### Threshold configuration

All thresholds are environment variables on `EvaluatorFunction` — adjust
without redeploying application code.

| Variable | Default | Meaning |
|---|---|---|
| `THRESHOLD_FAITHFULNESS` | `0.7` | Min fraction of answer claims supported by context |
| `THRESHOLD_ANSWER_RELEVANCE` | `0.6` | Min relevance score (0–1) |
| `THRESHOLD_CONTEXT_PRECISION` | `0.5` | Min fraction of retrieved chunks that are useful |
| `THRESHOLD_JUDGE_SCORE` | `3.0` | Min mean LLM judge score (1–5) |

### DynamoDB schema

**`AgenticRAGEvaluationLogs`** — one record per response

```
response_id           (PK) — UUID
timestamp                  — ISO 8601
question                   — original question
answer                     — delivered answer
documents                  — all chunks retrieved across all rounds
node_path                  — e.g. "agentic-loop (2 tool round(s))"
evaluation
  ├── faithfulness           — 0.0–1.0
  ├── answer_relevance       — 0.0–1.0
  ├── context_precision      — 0.0–1.0
  ├── judge_correctness      — 1–5
  ├── judge_completeness     — 1–5
  ├── judge_groundedness     — 1–5
  ├── judge_clarity          — 1–5
  ├── judge_score            — mean of judge dimensions
  ├── passes_threshold       — bool
  ├── failure_reasons        — list of failed metrics with values
  └── tool_rounds_used       — int (1–3)  ← Agentic_RAG specific
passes_threshold             — bool (top-level, for easy DynamoDB filtering)
flagged_for_review           — bool
flag_reason                  — "threshold_failure" | "user_thumbs_down"
flag_timestamp               — ISO 8601
```

**`AgenticRAGUserFeedback`** — one record per feedback submission

```
response_id  (PK) — UUID
timestamp         — ISO 8601
rating            — "thumbs_up" | "thumbs_down"
comment           — optional free text
```

### Online vs async evaluation trade-off

| | Online (synchronous gate) | Async (this implementation) |
|---|---|---|
| **Blocks response** | Yes — user waits for metrics | No — user gets answer immediately |
| **Latency added** | +8–15 seconds | 0 seconds |
| **Can reject bad answers** | Yes — hard gate before delivery | No — monitoring + flagging only |
| **Best for** | High-stakes domains (medical, legal) | General production workloads |

---

## Project Structure

```
Agentic_RAG/
├── src/
│   ├── lambda_function.py    # Main handler: agentic loop + async evaluator trigger
│   ├── agent.py              # Agentic loop — returns (answer, documents, tool_rounds)
│   ├── retriever.py          # FAISS vector search + S3 index cache
│   ├── indexer.py            # One-off: chunk docs, embed, upload index to S3
│   ├── evaluator.py          # Layer 1 (RAGAS metrics) + Layer 2 (LLM judge)
│   ├── evaluator_handler.py  # Async Lambda handler for background evaluation
│   └── storage.py            # DynamoDB writes for all 4 evaluation layers
├── template.yaml             # SAM: 2 Lambdas + API Gateway + 2 DynamoDB tables
├── requirements.txt
├── .env.example
└── README.md
```

---

## Architecture

```
User ──► API Gateway (HTTP API)
              │
    ┌─────────┴──────────────────────────────────┐
    │   AgenticRAGFunction (main Lambda)          │
    │                                             │
    │   agent.py — OpenAI gpt-4o tool-call loop  │
    │     Round 1: search_knowledge_base()        │
    │     Round 2: search_knowledge_base()        │  → retriever.py
    │     Round 3: search_knowledge_base()        │       │
    │     Final answer generation                 │  FAISS index
    └─────────────────────────────────────────────┘  (cached from S3)
              │                      │
              │ answer +             │ async invoke
              │ response_id          │ (fire-and-forget)
              ▼                      ▼
            User          EvaluatorFunction (background)
                               evaluator.py + storage.py
                               → AgenticRAGEvaluationLogs (DynamoDB)
```

**Key design decisions**

| | Detail |
|---|---|
| **LLM** | `gpt-4o` with function calling |
| **Judge model** | `gpt-4o` (same family — consider different model for stronger bias protection) |
| **Embeddings** | `text-embedding-3-small` (1536 dims) |
| **Vector store** | FAISS `IndexFlatL2` persisted to S3, cached in `/tmp` across warm starts |
| **Agentic pattern** | Up to 3 `search_knowledge_base` calls before forcing a final answer |
| **Evaluation** | Async: RAGAS + LLM judge + human flagging + user feedback |

---

## Prerequisites

- Python 3.12+
- AWS SAM CLI
- AWS credentials configured (`aws configure`)
- An S3 bucket for storing the index and raw documents
- An OpenAI API key

---

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY, S3_BUCKET in .env
```

---

## Step 1 — Build the FAISS Index

```bash
python src/indexer.py
```

This chunks your `.txt` documents from S3, embeds them with
`text-embedding-3-small`, builds a FAISS index, and uploads it back to S3.

---

## Step 2 — Run Locally with SAM

```bash
sam build
sam local start-api

curl -X POST http://127.0.0.1:3000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

---

## Step 3 — Deploy to AWS

```bash
sam deploy --guided
```

SAM will prompt for:
- `OpenAIApiKey` — your OpenAI API key
- `S3Bucket` — the bucket name
- `IndexKey` / `ChunksKey` — accept defaults or customise

This deploys:
- `agentic-rag` — main RAG Lambda (120s timeout, 1024MB)
- `agentic-rag-evaluator` — async evaluator Lambda (120s timeout, 256MB)
- API Gateway with `/query`, `/feedback`, `/health` endpoints
- `AgenticRAGEvaluationLogs` and `AgenticRAGUserFeedback` DynamoDB tables

---

## Usage

```bash
# Query
curl -X POST https://<ApiUrl>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key points about our return policy."}'
# Response: {"answer": "...", "response_id": "<uuid>"}

# Submit feedback (Layer 4)
curl -X POST https://<ApiUrl>/feedback \
  -H "Content-Type: application/json" \
  -d '{"response_id": "<uuid>", "rating": "thumbs_up"}'
# Response: {"status": "ok"}

# Health check
curl https://<ApiUrl>/health
# Response: {"status": "ok"}
```

---

## Environment Variables

### Main Lambda (`agentic-rag`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `S3_BUCKET` | — | S3 bucket for index & docs |
| `INDEX_KEY` | `faiss_index/index.faiss` | S3 key for FAISS index |
| `CHUNKS_KEY` | `faiss_index/chunks.pkl` | S3 key for chunk list |
| `DOCS_PREFIX` | `docs/` | S3 prefix for raw documents |
| `LLM_MODEL` | `gpt-4o` | OpenAI chat model |
| `TOP_K` | `5` | Chunks returned per search |
| `MAX_TOOL_ROUNDS` | `3` | Max agent search iterations |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `EVALUATOR_FUNCTION_NAME` | — | ARN of the async evaluator Lambda |
| `EVAL_TABLE_NAME` | `AgenticRAGEvaluationLogs` | DynamoDB table for eval logs |
| `FEEDBACK_TABLE_NAME` | `AgenticRAGUserFeedback` | DynamoDB table for user feedback |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Evaluator Lambda (`agentic-rag-evaluator`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Model for Layer 1 metric checks |
| `JUDGE_MODEL` | `gpt-4o` | Stronger model for Layer 2 judging |
| `EVAL_TABLE_NAME` | `AgenticRAGEvaluationLogs` | DynamoDB table for eval logs |
| `THRESHOLD_FAITHFULNESS` | `0.7` | Minimum faithfulness score |
| `THRESHOLD_ANSWER_RELEVANCE` | `0.6` | Minimum answer relevance score |
| `THRESHOLD_CONTEXT_PRECISION` | `0.5` | Minimum context precision score |
| `THRESHOLD_JUDGE_SCORE` | `3.0` | Minimum mean LLM judge score |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Security Notes

- **Never commit `.env`** — it contains your API key.
- For production, store `OPENAI_API_KEY` in AWS Secrets Manager or SSM Parameter Store.
- The main Lambda IAM role grants read-only S3 access and only `lambda:InvokeFunction` on the evaluator — principle of least privilege.
- The evaluator Lambda IAM role has no S3 or internet access beyond OpenAI calls.
