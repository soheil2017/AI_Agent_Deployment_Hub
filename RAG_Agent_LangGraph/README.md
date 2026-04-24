# RAG Agent with LangGraph

A Retrieval-Augmented Generation (RAG) agent implemented as an explicit
[LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`, deployed
on AWS Lambda + API Gateway via SAM.

Includes a **hybrid evaluation framework** (RAGAS-style metrics, LLM-as-a-Judge,
human review flagging, and user feedback) that runs asynchronously in a separate
Lambda — so evaluation never adds latency to the user-facing response.

---

## RAG Graph

```
START
  │
  ▼
retrieve          ← embed question, fetch top-k chunks from FAISS
  │
  ▼
grade_documents   ← LLM decides: are these chunks relevant?
  │
  ├── NO  ──► no_answer ──► END    (skip generation entirely)
  │
  └── YES ──► generate  ──► END    (generate answer from context)
```

```
RAGState
├── response_id: str        # UUID for tracking across systems
├── question:    str        # input
├── documents:   list[str]  # populated by retrieve
├── relevant:    bool       # populated by grade_documents
└── answer:      str        # populated by generate or no_answer
```

The graph is compiled **once at module level** and reused across Lambda warm
starts — equivalent to a singleton service object.

### Why `grade_documents`?

Without grading, the generator always runs — even when retrieved chunks are
completely unrelated to the question. The grading node short-circuits that path:

- If the chunks are relevant → generate an answer
- If not → return a clear "no information" message immediately, saving an LLM call

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
RAGAgentFunction  (main Lambda — fast path)
 │
 ├── retrieve → grade_documents → generate / no_answer
 │
 ├──► delivers answer to user immediately  (~3–6 seconds)
 │
 └──► async invoke ──────────────────────────────────────────────────────────┐
                                                                             │
                                                           EvaluatorFunction │
                                                           (background)      │
                                                                             │
                                                    run_evaluation()         │
                                                      │                      │
                                                      ├── Layer 1            │
                                                      │   faithfulness       │
                                                      │   answer_relevance   │
                                                      │   context_precision  │
                                                      │                      │
                                                      └── Layer 2            │
                                                          LLM judge          │
                                                          (gpt-4o)           │
                                                            │                │
                                                            ▼                │
                                                      DynamoDB log           │
                                                      (RAGEvaluationLogs)    │
                                                            │                │
                                                     passes threshold?       │
                                                      │            │         │
                                                     YES           NO        │
                                                      │            │         │
                                                      │      flag_for_review │
                                                      │      (Layer 3)       │
                                                      └────────────┘ ───────┘
```

### The four evaluation layers

#### Layer 1 — RAGAS-style Metrics

Automated metrics that require no ground truth. Computed on every response.

| Metric | What it measures | How |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | Extract claims from answer → verify each against context |
| **Answer Relevance** | Does the answer actually address the question? | LLM rates 0.0–1.0 |
| **Context Precision** | Are the retrieved chunks genuinely useful for the question? | LLM checks each chunk |

#### Layer 2 — LLM-as-a-Judge

A stronger model (`gpt-4o`) scores the answer on four dimensions using a dedicated
judge prompt. Using a different model than the generator avoids self-serving bias.

| Dimension | What it measures | Scale |
|---|---|---|
| **Correctness** | Is the answer factually accurate? | 1–5 |
| **Completeness** | Does it fully answer the question? | 1–5 |
| **Groundedness** | Is every claim backed by the context? | 1–5 |
| **Clarity** | Is it well-written and easy to understand? | 1–5 |

`judge_score` = mean of the four dimensions.

#### Layer 3 — Human Review Flagging

Responses are automatically flagged for human review when:
- Any Layer 1 or Layer 2 metric falls below its configured threshold
- A user submits a thumbs-down (see Layer 4)

Flagged records appear in `RAGEvaluationLogs` with `flagged_for_review = true`
and a `flag_reason` — giving reviewers a unified queue without needing a
separate system.

#### Layer 4 — User Feedback

Users submit thumbs-up or thumbs-down via `POST /feedback`. Thumbs-down
automatically triggers a Layer 3 human review flag.

```bash
curl -X POST https://<ApiUrl>/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "response_id": "<uuid from /query response>",
    "rating": "thumbs_down",
    "comment": "Answer was incorrect"
  }'
```

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

**`RAGEvaluationLogs`** — one record per response

```
response_id         (PK) — UUID
timestamp                — ISO 8601
question                 — original question
answer                   — delivered answer
documents                — retrieved chunks
node_path                — e.g. "retrieve → grade(YES) → generate"
evaluation
  ├── faithfulness        — 0.0–1.0
  ├── answer_relevance    — 0.0–1.0
  ├── context_precision   — 0.0–1.0
  ├── judge_correctness   — 1–5
  ├── judge_completeness  — 1–5
  ├── judge_groundedness  — 1–5
  ├── judge_clarity       — 1–5
  ├── judge_score         — mean of judge dimensions
  ├── passes_threshold    — bool
  └── failure_reasons     — list of failed metrics with values
passes_threshold          — bool (top-level, for easy DynamoDB filtering)
flagged_for_review        — bool
flag_reason               — "threshold_failure" | "user_thumbs_down"
flag_timestamp            — ISO 8601
```

**`RAGUserFeedback`** — one record per feedback submission

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
RAG_Agent_LangGraph/
├── src/
│   ├── lambda_function.py    # Main handler: RAG graph + async evaluator trigger
│   ├── graph.py              # LangGraph StateGraph (retrieve → grade → generate)
│   ├── nodes.py              # RAGState + node functions
│   ├── retriever.py          # FAISS vector search + S3 index cache
│   ├── evaluator.py          # Layer 1 (RAGAS metrics) + Layer 2 (LLM judge)
│   ├── evaluator_handler.py  # Async Lambda handler for background evaluation
│   └── storage.py            # DynamoDB writes for all 4 evaluation layers
├── template.yaml             # SAM: 2 Lambdas + API Gateway + 2 DynamoDB tables
├── requirements.txt
├── .env.example
└── README.md
```

---

## How it compares to Agentic_RAG

| Aspect | Agentic_RAG | RAG_Agent_LangGraph |
|---|---|---|
| Flow control | OpenAI tool-calling loop | LangGraph StateGraph |
| Steps | Dynamic (agent decides) | Conditional: retrieve → grade → generate/no_answer |
| Visualisable | No | Yes (`app.get_graph().draw_mermaid()`) |
| Extensible | Add tools | Add nodes / edges |
| LLM calls (main path) | ≥ 2 (plan + answer) | 2–3 (embed + grade + generate) |
| Evaluation | None | Async: RAGAS + LLM judge + human flagging + user feedback |
| Model | gpt-4o | gpt-4o-mini (generator) + gpt-4o (judge) |

---

## Prerequisites

- Python 3.12
- AWS CLI + SAM CLI configured
- An S3 bucket with a FAISS index built by `Agentic_RAG/src/indexer.py`

## Build the FAISS Index

Reuse the indexer from `Agentic_RAG` — no separate indexer needed:

```bash
cd Agentic_RAG
pip install -r requirements.txt
python src/indexer.py \
  --docs-dir /path/to/documents \
  --bucket my-rag-bucket
```

## Local Development

```bash
cd RAG_Agent_LangGraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and S3_BUCKET in .env

# Test the RAG graph directly
cd src
python -c "
from graph import app
result = app.invoke({'question': 'What is ...?'})
print(result['answer'])
"

# Test the evaluator directly
python -c "
from evaluator import run_evaluation
result = run_evaluation(
    question='What is photosynthesis?',
    documents=['Photosynthesis converts sunlight into energy...'],
    answer='Photosynthesis is the process by which plants convert sunlight into food.',
)
print(result.to_dict())
"
```

## Deploy to AWS

```bash
cd RAG_Agent_LangGraph
sam build
sam deploy --guided
```

SAM will prompt for:
- `OpenAIApiKey` — your OpenAI API key
- `S3Bucket` — bucket containing the FAISS index
- `IndexKey` — S3 key for `index.faiss` (default: `faiss_index/index.faiss`)
- `ChunksKey` — S3 key for `chunks.pkl` (default: `faiss_index/chunks.pkl`)

This deploys:
- `rag-agent-langgraph` — main RAG Lambda (60s timeout)
- `rag-evaluator-langgraph` — async evaluator Lambda (120s timeout)
- API Gateway with `/query`, `/feedback`, `/health` endpoints
- `RAGEvaluationLogs` and `RAGUserFeedback` DynamoDB tables

## Usage

```bash
# Query
curl -X POST https://<ApiUrl>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ...?"}'
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

### Main Lambda (`rag-agent-langgraph`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `S3_BUCKET` | — | S3 bucket with FAISS index |
| `INDEX_KEY` | `faiss_index/index.faiss` | S3 key for FAISS index |
| `CHUNKS_KEY` | `faiss_index/chunks.pkl` | S3 key for chunk list |
| `LLM_MODEL` | `gpt-4o-mini` | Model for grading and generation |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `EVALUATOR_FUNCTION_NAME` | — | ARN of the async evaluator Lambda |
| `EVAL_TABLE_NAME` | `RAGEvaluationLogs` | DynamoDB table for eval logs |
| `FEEDBACK_TABLE_NAME` | `RAGUserFeedback` | DynamoDB table for user feedback |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Evaluator Lambda (`rag-evaluator-langgraph`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Model for Layer 1 metric checks |
| `JUDGE_MODEL` | `gpt-4o` | Stronger model for Layer 2 judging |
| `EVAL_TABLE_NAME` | `RAGEvaluationLogs` | DynamoDB table for eval logs |
| `THRESHOLD_FAITHFULNESS` | `0.7` | Minimum faithfulness score |
| `THRESHOLD_ANSWER_RELEVANCE` | `0.6` | Minimum answer relevance score |
| `THRESHOLD_CONTEXT_PRECISION` | `0.5` | Minimum context precision score |
| `THRESHOLD_JUDGE_SCORE` | `3.0` | Minimum mean LLM judge score |
| `LOG_LEVEL` | `INFO` | Python logging level |
