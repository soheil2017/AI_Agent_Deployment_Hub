# Agentic RAG on AWS Lambda + OpenAI

A production-ready **Agentic Retrieval-Augmented Generation** system that runs on AWS Lambda. The agent uses OpenAI function/tool calling to iteratively search a FAISS knowledge base before synthesising a final answer.

## Architecture

```
User ──► API Gateway (HTTP API)
              │
              ▼
         Lambda Function
              │
    ┌─────────┴──────────┐
    │   agent.py         │  OpenAI gpt-4o  ◄──► search_knowledge_base()
    │   Agentic loop     │                            │
    └─────────┬──────────┘                    retriever.py
              │                                       │
              │                           FAISS index (cached /tmp)
              │                                       │
              └──────────────────────────────► S3 bucket
```

**Key components**

| File | Role |
|---|---|
| `src/lambda_function.py` | API Gateway handler — parses request, calls agent |
| `src/agent.py` | Agentic loop — iterative tool calling with `gpt-4o` |
| `src/retriever.py` | FAISS vector search — downloads index from S3, caches in `/tmp` |
| `src/indexer.py` | One-off script — chunks docs, embeds with OpenAI, uploads index to S3 |
| `template.yaml` | AWS SAM template — Lambda + HTTP API Gateway + IAM |

**Design decisions**

- **LLM**: `gpt-4o` with function calling
- **Embeddings**: `text-embedding-3-small` (1536 dims)
- **Vector store**: FAISS `IndexFlatL2` persisted to S3; cached in `/tmp` across warm starts
- **Agentic pattern**: up to 3 `search_knowledge_base` calls before forcing a final answer

---

## Prerequisites

- Python 3.12+
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- AWS credentials configured (`aws configure`)
- An S3 bucket for storing the index and raw documents
- An OpenAI API key

---

## Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY, S3_BUCKET, etc.
```

---

## Step 1 — Upload Documents & Build the Index

Place your `.txt` documents in `s3://<S3_BUCKET>/docs/` (or the prefix set by `DOCS_PREFIX`), then run:

```bash
python src/indexer.py
```

This will:
1. List all `.txt` files under `DOCS_PREFIX`
2. Split them into overlapping character chunks
3. Embed each chunk with `text-embedding-3-small`
4. Build a FAISS index and upload it to S3

---

## Step 2 — Run Locally with SAM

```bash
# Build the Lambda package
sam build

# Start a local API (requires Docker)
sam local start-api

# In another terminal, send a query
curl -X POST http://127.0.0.1:3000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Or invoke the function directly:

```bash
sam local invoke AgenticRAGFunction \
  --event '{"body":"{\"question\":\"What is RAG?\"}","requestContext":{"http":{"method":"POST"}}}'
```

---

## Step 3 — Deploy to AWS

```bash
# First-time interactive deploy
sam deploy --guided

# Subsequent deploys
sam deploy
```

During `--guided` you will be prompted for:
- **Stack name** (e.g. `agentic-rag`)
- **AWS Region**
- **OpenAIApiKey** — your OpenAI API key (masked)
- **S3Bucket** — the bucket name
- **IndexKey / ChunksKey** — accept defaults or customise

---

## Step 4 — Call the API

After deployment, SAM prints the `ApiUrl` output. Use it to query your agent:

```bash
API_URL=https://<id>.execute-api.<region>.amazonaws.com

# Ask a question
curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key points about our return policy."}'

# Health check
curl "$API_URL/health"
```

**Response format**

```json
{
  "answer": "According to the knowledge base, ..."
}
```

---

## Configuration Reference

All settings are controlled via environment variables (see `.env.example`):

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
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Project Structure

```
Agentic_RAG/
├── src/
│   ├── lambda_function.py   # Lambda entry point
│   ├── agent.py             # Agentic loop (OpenAI tool calling)
│   ├── retriever.py         # FAISS search + S3 caching
│   └── indexer.py           # Document chunking + indexing
├── template.yaml            # AWS SAM template
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Security Notes

- **Never commit `.env`** — it contains your API key.
- For production, store `OPENAI_API_KEY` in AWS Secrets Manager or SSM Parameter Store and retrieve it at runtime rather than passing it as a plain environment variable.
- The IAM role grants read-only access to the specified S3 bucket only.
