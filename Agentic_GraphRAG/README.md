# Conduit Graph RAG Agent

An intelligent question-answering agent built for **Conduit** — the dental healthcare interoperability platform by [HeyDonto AI Technology](https://www.heydonto.com). The agent combines a **Knowledge Graph** (Neo4j), a **Vector Database** (ChromaDB), and a **Vision Language Model** (Claude Vision) to answer complex questions about patients, referrals, prior authorizations, FHIR standards, and clinical documents — things that flat vector search alone cannot handle.

---

## Why This Project Exists

Dental data has always been isolated from the rest of healthcare. Conduit solves this by connecting dental practice management systems (PMS) to medical EHRs, payers, patients, and AI platforms through a single FHIR R4 bridge.

Operating at this intersection produces deeply relational, multi-system data:

- A patient has conditions → treated by a dentist → referred to a physician → prior auth submitted to a payer → claim routed → clearance returned
- FHIR resources link together across systems (Patient → Encounter → Condition → Procedure → Claim)
- PMS platforms use incompatible code systems (CDT ↔ ICD-10 ↔ SNOMED ↔ CPT)
- Clinical documents arrive as images — X-rays, insurance cards, faxed referrals, EOBs

A standard RAG system backed by a flat vector database cannot answer questions like:

> *"Which patients have a pending medical-to-dental referral where the payer has not returned prior auth in 7 days?"*

This agent can — because it uses a **Knowledge Graph** to traverse entity relationships, a **Vector DB** to search policies and standards, and a **VLM** to read images. The right tool is chosen automatically for every query.

---

## Production Purpose

In production, this agent serves as the **AI assistant layer on top of Conduit's interoperability platform**. It is built for:

| User | How They Use It |
|---|---|
| **DSO staff** | "Show me all pending referrals for patients with diabetes" |
| **Dental providers** | "What prior auth criteria does Aetna require for D4341?" |
| **Medical providers** | "What is the status of the dental clearance I requested for Alice Johnson?" |
| **Payer operations** | "Which prior auth requests are pending beyond 7 days?" |
| **Compliance teams** | "What does TEFCA require for dental-to-medical data exchange?" |
| **Dental AI platforms** | "Analyze this panoramic X-ray and generate a structured FHIR DiagnosticReport" |

---

## Architecture Overview

```
                         User Query (text or image)
                                    │
                         ┌──────────▼──────────┐
                         │    FastAPI  main.py  │
                         │    POST /query       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   LangGraph Agent    │
                         │                      │
                         │  1. classify_query   │  ← decides which tool to use
                         │  2. run_tools        │  ← calls the right tool(s)
                         │  3. generate         │  ← builds the final answer
                         │  4. no_answer        │  ← fallback if nothing found
                         └──┬──────┬────────┬──┘
                            │      │        │
               ┌────────────▼─┐ ┌──▼───┐ ┌─▼──────────────┐
               │   Neo4j      │ │Chroma│ │  Claude Vision  │
               │  Graph DB    │ │  DB  │ │     (VLM)       │
               │              │ │      │ │                 │
               │  WHO relates │ │ WHAT │ │  WHAT is in     │
               │  to WHO      │ │ docs │ │  this image     │
               └──────────────┘ └──────┘ └─────────────────┘
                            │      │        │
                         ┌──▼──────▼────────▼──┐
                         │     OpenAI LLM       │
                         │   (final answer)     │
                         └─────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │      Langfuse        │
                         │   Observability      │
                         │  traces · spans ·    │
                         │  scores · feedback   │
                         └─────────────────────┘
```

---

## The Four Query Types

The agent classifies every incoming question into one of four types and routes it to the appropriate tool(s):

| Type | Triggered When | Tool Used | Example |
|---|---|---|---|
| **relational** | Question asks about entities and their relationships | Neo4j graph traversal | *"Which patients have a pending prior auth?"* |
| **semantic** | Question asks about policies, standards, or documentation | ChromaDB vector search | *"What does CARIN Blue Button require?"* |
| **visual** | An image is attached to the request | Claude Vision (VLM) | *"Analyze this dental X-ray"* |
| **hybrid** | Question needs both relationship data AND documentation | Neo4j + ChromaDB combined | *"Which patients with periodontitis have pending prior auths, and what are the payer rules?"* |

---

## Knowledge Graph Schema (Neo4j)

The graph captures the full Conduit domain: patients, providers, organizations, clinical data, referrals, and prior authorizations — and the relationships between all of them.

```
Nodes:
  Patient      {id, name, dob, member_id}
  Provider     {npi, name, specialty, type}      ← type: dental | medical
  Organization {id, name, type}                  ← type: DSO | health_system | payer
  Condition    {code, system, description}        ← system: ICD-10 | SNOMED
  Procedure    {code, system, description}        ← system: CDT | CPT
  Referral     {id, type, status, created_date}   ← type: medical_to_dental | dental_to_medical | dental_to_dental
  PriorAuth    {id, status, submitted_date, decision_date, procedure_code}

Relationships:
  (Patient)  -[:HAS_CONDITION]->  (Condition)
  (Patient)  -[:HAD_PROCEDURE]->  (Procedure)
  (Patient)  -[:HAS_REFERRAL]->   (Referral)
  (Provider) -[:SENT_REFERRAL]->  (Referral)
  (Referral) -[:ASSIGNED_TO]->    (Provider)
  (Referral) -[:REQUIRES_AUTH]->  (PriorAuth)
  (PriorAuth)-[:SUBMITTED_TO]->   (Organization)
  (Provider) -[:WORKS_AT]->       (Organization)
```

The graph allows multi-hop queries that flat vector search cannot handle. For example, traversing `Patient → Referral → PriorAuth → Organization` in a single Cypher query answers: *"Which payer is holding up Alice Johnson's prior auth?"*

---

## Vector Store Contents (ChromaDB)

ChromaDB stores policy and specification documents as embeddings for semantic search:

| Document | Topic |
|---|---|
| FHIR R4 overview and key resources | Standards |
| Prior authorization rules and workflow (Da Vinci PAS IG) | Payer policy |
| CARIN Blue Button patient access requirements | Patient access |
| TEFCA exchange framework and exchange purposes | Interoperability |
| Medical-to-dental referral transformation workflow | Referral workflow |
| Dental data isolation problem and Conduit's solution | Product |
| Oral Health Interoperability Alliance (OHIA) specs | Standards |
| PMS vendor integration guide (Dentrix, Eaglesoft, Open Dental) | Integration |

---

## VLM Capabilities (Claude Vision)

The Vision Language Model reads images and converts them to structured text that enters the RAG pipeline as context:

| Image Type | What the VLM Extracts |
|---|---|
| Dental X-ray (periapical, panoramic) | Tooth numbers, caries, bone loss, restorations, implants |
| Insurance card | Payer name, member ID, group number, plan type |
| EOB (Explanation of Benefits) | Procedure codes, service dates, amounts billed/paid, denial codes |
| Faxed referral letter | Referring provider, receiving provider, patient name, diagnosis, reason |

---

## Evaluation Framework (4 Layers)

Every response is evaluated asynchronously — the answer is delivered immediately, and evaluation runs in the background. All scores are visible in the Langfuse dashboard.

| Layer | What It Measures | When It Runs |
|---|---|---|
| **Layer 1 — RAGAS metrics** | Faithfulness, answer relevance, context precision (0.0–1.0) | After every response |
| **Layer 2 — LLM-as-a-Judge** | Correctness, completeness, groundedness, clarity (1–5) | After every response |
| **Layer 3 — Human review flag** | Responses below threshold are flagged in Langfuse | When thresholds fail |
| **Layer 4 — User feedback** | Thumbs-up / thumbs-down from end users (POST /feedback) | On user action |

---

## Project Structure

```
Agentic_GraphRAG/
├── src/
│   ├── main.py               API entry point (FastAPI) — /query, /feedback, /health
│   ├── graph.py              LangGraph StateGraph definition and compilation
│   ├── nodes.py              State type + 4 node functions (classify, run, generate, fallback)
│   ├── tools.py              3 retrieval tools: graph_search, vector_search, vlm_analyze
│   ├── storage.py            Neo4j client + ChromaDB client (module-level singletons)
│   ├── ingestion.py          Load synthetic FHIR data into Neo4j + ChromaDB (run once)
│   ├── evaluator.py          4-layer evaluation logic
│   ├── evaluator_handler.py  Async eval runner + Langfuse score logger
│   └── lambda_function.py    AWS Lambda handler (alternative to main.py for AWS deployment)
├── Dockerfile                Container build for Railway
├── railway.toml              Railway deployment configuration
├── vercel.json               Vercel deployment configuration (alternative)
├── requirements.txt          Python dependencies
└── template.yaml             AWS SAM template (alternative for AWS deployment)
```

---

## Component Responsibilities

### `main.py` — API Layer
The FastAPI application. Handles HTTP requests, attaches the Langfuse tracing callback to every LangGraph invocation, delivers the answer immediately, and fires the evaluation as a background task. Also handles user feedback via `POST /feedback`.

### `graph.py` — Agent Orchestration
Defines and compiles the LangGraph `StateGraph`. Wires the four nodes together and defines the conditional routing logic: if `run_tools` returns documents → `generate`, otherwise → `no_answer`. The compiled graph is cached across container requests.

### `nodes.py` — Agent Logic
Contains the typed state (`GraphRAGState`) and the four node functions:
- `classify_query` — asks the LLM to classify the question type
- `run_tools` — dispatches to the correct tool(s) based on query type
- `generate` — builds the final answer from retrieved context
- `no_answer` — graceful fallback when no context is found

### `tools.py` — Retrieval Tools
The three tools the agent can call:
- `graph_search` — uses the LLM to generate a Cypher query, runs it on Neo4j, returns structured results
- `vector_search` — embeds the query and retrieves the top-k matching policy documents from ChromaDB
- `vlm_analyze` — sends an image to Claude Vision, returns structured clinical/document findings

### `storage.py` — Database Clients
Singleton clients for Neo4j and ChromaDB. Both are initialized once and reused across all requests within the same container instance, avoiding reconnection overhead.

### `ingestion.py` — Data Loader
One-time script that populates Neo4j with synthetic dental/FHIR entities (patients, providers, referrals, prior auths) and loads policy documents into ChromaDB. Re-run to refresh data.

### `evaluator.py` — Evaluation Logic
Implements all four evaluation layers using the OpenAI API. Layer 1 uses RAGAS-style prompting (faithfulness, answer relevance, context precision). Layer 2 uses a stronger judge model (GPT-4o) to score on four quality dimensions.

### `evaluator_handler.py` — Async Eval + Langfuse
Called as a FastAPI `BackgroundTask`. Runs the evaluator, then pushes all scores to Langfuse as named scores on the original query trace. Also exposes `log_user_feedback()` for Layer 4.

---

## Deployment Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    VERCEL  (Option A)                       │
│             or  RAILWAY web service (Option B)              │
│                                                            │
│                    FastAPI  main.py                        │
│             POST /query · POST /feedback · GET /health     │
└─────────────────────────┬──────────────────────────────────┘
                          │  internal Railway service URLs
          ┌───────────────┼────────────────────┐
          │               │                    │
┌─────────▼──────┐ ┌──────▼───────┐  ┌────────▼────────┐
│    RAILWAY     │ │   RAILWAY    │  │    LANGFUSE     │
│                │ │              │  │     CLOUD       │
│  Neo4j         │ │  ChromaDB    │  │                 │
│  bolt://...    │ │  http://...  │  │  Traces, spans, │
│  :7687         │ │  :8000       │  │  scores,        │
│                │ │              │  │  user feedback  │
└────────────────┘ └──────────────┘  └─────────────────┘
```

**Option A — Vercel + Railway (recommended split):**
- Vercel hosts the FastAPI app as serverless functions
- Railway hosts Neo4j and ChromaDB as persistent container services
- Use this when you want Vercel's edge network for the API layer

**Option B — Railway only (simplest):**
- All three services (FastAPI, Neo4j, ChromaDB) run on Railway
- Easier to manage, no cross-platform networking
- Use this for the initial production deployment

---

## Setup

### Prerequisites

- Python 3.12+
- [Neo4j](https://neo4j.com/download/) running locally or on Railway
- [Docker](https://www.docker.com/) (optional, for local ChromaDB)
- API keys for OpenAI, Anthropic, and Langfuse

### 1. Clone and install

```bash
cd Agentic_GraphRAG
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
VLM_MODEL=claude-opus-4-6

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# ChromaDB (local persistent mode by default — no extra config needed)
# Set CHROMA_USE_HTTP=true and CHROMA_HOST/PORT for a remote ChromaDB instance
CHROMA_USE_HTTP=false
CHROMA_COLLECTION=conduit_docs

# Langfuse
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Start Neo4j locally (Docker)

```bash
docker run \
  --name neo4j-conduit \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/your-password \
  -d neo4j:5
```

Neo4j Browser is available at http://localhost:7474

### 4. Load synthetic data

```bash
cd src
python ingestion.py
```

This creates all graph nodes and relationships in Neo4j, and loads 8 policy documents into ChromaDB. You should see output like:

```
INFO: Neo4j: created all nodes
INFO: Neo4j: created all relationships
INFO: ChromaDB: ingested 8 documents
INFO: Ingestion complete.
```

---

## Running Locally

```bash
cd src
uvicorn main:api --reload --port 8000
```

The API is now available at `http://localhost:8000`.

### Example requests

**Relational query** (uses Neo4j):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which patients have a pending prior authorization?"}'
```

**Semantic query** (uses ChromaDB):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does CARIN Blue Button require for patient access to dental claims?"}'
```

**Hybrid query** (uses both):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which patients with periodontitis have pending prior auths, and what are the standard payer rules for this procedure?"}'
```

**Visual query** (uses Claude Vision):
```bash
# Encode your image to base64 first
IMAGE_B64=$(base64 -i xray.jpg)

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Analyze this X-ray\", \"image_base64\": \"$IMAGE_B64\", \"media_type\": \"image/jpeg\"}"
```

**User feedback** (Layer 4):
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"trace_id": "<trace_id_from_query_response>", "rating": "thumbs_up"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

### Example response

```json
{
  "answer": "Two patients currently have pending prior authorizations:\n1. Alice Johnson (P001) — procedure D4341 (periodontal scaling), submitted to United Health Payer on 2026-04-16, decision pending.\n2. ...",
  "trace_id": "3f7a1b2c-...",
  "query_type": "relational"
}
```

---

## Deploying to Railway

### Step 1 — Create a Railway project

```bash
npm install -g @railway/cli
railway login
railway init
```

### Step 2 — Add database services

In the Railway dashboard, add two services from templates:
- **Neo4j** — use the official Neo4j template
- **ChromaDB** — deploy from the `chromadb/chroma` Docker image

Note the internal service URLs Railway assigns to each (they look like `neo4j.railway.internal`).

### Step 3 — Set environment variables

In the Railway dashboard for the `web` service, set all variables from the `.env` file above. Use the Railway-internal URLs for Neo4j and ChromaDB:

```
NEO4J_URI=bolt://neo4j.railway.internal:7687
CHROMA_USE_HTTP=true
CHROMA_HOST=chromadb.railway.internal
CHROMA_PORT=8000
```

### Step 4 — Deploy

```bash
railway up
```

Railway uses the `Dockerfile` to build the container and `railway.toml` for deployment settings (health check path, restart policy).

### Step 5 — Run ingestion on Railway

After the first deploy, run the ingestion script as a one-off Railway job:

```bash
railway run python src/ingestion.py
```

---

## Deploying to Vercel (API layer only)

Use this if you want Vercel to host the FastAPI app while Neo4j and ChromaDB stay on Railway.

### Step 1 — Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2 — Add secrets

```bash
vercel env add OPENAI_API_KEY
vercel env add ANTHROPIC_API_KEY
vercel env add LANGFUSE_PUBLIC_KEY
vercel env add LANGFUSE_SECRET_KEY
vercel env add NEO4J_URI
vercel env add NEO4J_PASSWORD
vercel env add CHROMA_HOST
```

### Step 3 — Deploy

```bash
vercel deploy --prod
```

Vercel uses `vercel.json` to route all traffic to `src/main.py`.

> **Note:** Vercel serverless functions have a 60-second timeout on the Pro plan. This is sufficient for most queries but may be tight for large VLM image analysis. If timeouts occur, move the FastAPI app to Railway instead.

---

## Observability with Langfuse

Every query automatically creates a full trace in Langfuse with:

| Object | What You See |
|---|---|
| **Trace** | The full request — question, final answer, total latency, token cost |
| **Span: classify_query** | Input: question text. Output: query type (relational/semantic/visual/hybrid) |
| **Span: run_tools** | Input: query type. Output: context documents retrieved |
| **Span: generate** | Input: context + question. Output: answer. Token usage and cost |
| **Score: faithfulness** | Are answer claims supported by the retrieved context? (0.0–1.0) |
| **Score: answer_relevance** | Does the answer address the question? (0.0–1.0) |
| **Score: context_precision** | Are the retrieved chunks actually useful? (0.0–1.0) |
| **Score: judge_score** | LLM-as-a-Judge mean score (1.0–5.0) |
| **Score: passes_threshold** | Did this response meet all quality thresholds? (1.0 or 0.0) |
| **Score: user_feedback** | User thumbs-up (1.0) or thumbs-down (0.0) |

Access the Langfuse dashboard at [cloud.langfuse.com](https://cloud.langfuse.com) (or your self-hosted instance).

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key — used for LLM calls and embeddings |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key — used for Claude Vision (VLM) |
| `LLM_MODEL` | No | Text LLM model (default: `gpt-4o-mini`) |
| `JUDGE_MODEL` | No | Evaluator judge model (default: `gpt-4o`) |
| `VLM_MODEL` | No | Vision model (default: `claude-opus-4-6`) |
| `NEO4J_URI` | Yes | Neo4j connection URI (e.g. `bolt://localhost:7687`) |
| `NEO4J_USER` | No | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Yes | Neo4j password |
| `CHROMA_USE_HTTP` | No | Set `true` for remote ChromaDB (default: `false`) |
| `CHROMA_HOST` | If HTTP | ChromaDB hostname |
| `CHROMA_PORT` | No | ChromaDB port (default: `8000`) |
| `CHROMA_COLLECTION` | No | Collection name (default: `conduit_docs`) |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse secret key |
| `LANGFUSE_HOST` | No | Langfuse host (default: `https://cloud.langfuse.com`) |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API framework | FastAPI | HTTP endpoints, background tasks |
| Agent orchestration | LangGraph | Stateful multi-node agent graph |
| Knowledge Graph | Neo4j | Entity relationship storage and traversal |
| Vector Database | ChromaDB | Semantic document search |
| Vision Language Model | Claude Vision (claude-opus-4-6) | Image analysis — X-rays, EOBs, referrals |
| Text LLM | GPT-4o-mini | Query classification, Cypher generation, answer generation |
| Observability | Langfuse | Traces, spans, evaluation scores, user feedback |
| Deployment (app) | Railway / Vercel | Container or serverless hosting |
| Deployment (DBs) | Railway | Neo4j and ChromaDB persistent services |

---

## Part of the AI Agent Deployment Hub

This project is one of several AI agent patterns in the [AI Agent Deployment Hub](../README.md):

| Project | Pattern |
|---|---|
| `RAG_Agent_LangGraph` | Standard RAG with FAISS + LangGraph |
| `Agentic_RAG` | Agentic RAG with self-correction loop |
| `Agentic_GraphRAG` | **This project** — Graph + Vector + VLM |
| `Bedrock_RAG_Agent` | RAG on AWS Bedrock |
| `MCP-Powered_LLM_Agent` | Agent with MCP tool servers |
| `RL_Game_Agent` | Hybrid LLM + Reinforcement Learning |
