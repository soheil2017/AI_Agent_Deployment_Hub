# Bedrock RAG Agent

A fully managed **Agentic RAG** system built entirely on AWS — no OpenAI key required.

The Bedrock Agent uses Claude 3.5 Sonnet as the reasoning engine and automatically retrieves context from a Bedrock Knowledge Base (backed by OpenSearch Serverless) before generating answers.

## Architecture

```
User
 │
 ▼
API Gateway (HTTP API)
 │
 ▼
Lambda  ──────────────────────► Bedrock Agent (Claude 3.5 Sonnet)
                                        │
                                        │  tool call: search KB
                                        ▼
                               Knowledge Base (managed RAG)
                                        │
                          ┌─────────────┴────────────┐
                          │                          │
                   S3 (documents)     OpenSearch Serverless
                                      (Titan Embeddings v2)
```

**How the agentic loop works (managed by Bedrock):**
1. User sends a question to the Lambda via API Gateway
2. Lambda calls `bedrock-agent-runtime:InvokeAgent`
3. Bedrock Agent decides to call the Knowledge Base tool
4. Knowledge Base embeds the query → searches OpenSearch Serverless → returns top chunks
5. Agent synthesises a final answer grounded in the retrieved context
6. Lambda streams the answer back to the user

**No custom retrieval code, no FAISS, no embedding logic to maintain.**

---

## Stack

| Component | Service |
|---|---|
| LLM / Agent | Amazon Bedrock Agent (Claude 3.5 Sonnet) |
| Embeddings | Amazon Titan Embeddings v2 |
| Vector store | Amazon OpenSearch Serverless (VECTORSEARCH) |
| RAG pipeline | Amazon Bedrock Knowledge Base |
| Documents | Amazon S3 |
| API | API Gateway HTTP API |
| Compute | AWS Lambda (Python 3.12) |
| IaC | AWS SAM |

---

## Prerequisites

- Python 3.12+
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- AWS credentials configured (`aws configure`)
- Bedrock model access enabled for your region:
  - `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - `amazon.titan-embed-text-v2:0`

> Enable model access in the [Bedrock console](https://console.aws.amazon.com/bedrock/) → Model access.

---

## Deploy

```bash
# 1. Install local dependencies (for scripts only)
pip install -r requirements.txt

# 2. Build and deploy
sam build
sam deploy --guided
```

During `--guided` you will be prompted for:
- **Stack name** (e.g. `bedrock-rag-agent`)
- **AWS Region** (must support Bedrock — e.g. `us-east-1`)
- Accept defaults for `CollectionName`

SAM will print outputs including `ApiUrl`, `DocumentsBucket`, `KnowledgeBaseId`, and `DataSourceId`.

---

## Step 1 — Upload Documents

Upload `.txt`, `.pdf`, or `.docx` files to the S3 bucket created by SAM:

```bash
aws s3 cp my-docs/ s3://<DocumentsBucket>/ --recursive
```

Bedrock Knowledge Bases support: plain text, PDF, Word, HTML, Markdown.

---

## Step 2 — Sync the Knowledge Base

After uploading documents, trigger an ingestion job to embed and index them:

```bash
# Copy env template and fill in values from SAM outputs
cp .env.example .env

python scripts/sync_kb.py
```

The script polls until the ingestion job completes and prints progress.

---

## Step 3 — Query the API

```bash
API_URL=https://<id>.execute-api.<region>.amazonaws.com

# Ask a question
curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'

# Maintain a conversation session
curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you elaborate?", "session_id": "my-session-123"}'

# Health check
curl "$API_URL/health"
```

**Response format**

```json
{
  "answer": "According to the knowledge base ...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Pass the returned `session_id` in follow-up requests to maintain conversation context.

---

## Configuration Reference

All runtime values are set as Lambda environment variables by the SAM template.
After deploy, fill in `.env` from the SAM stack outputs for the local sync script.

| Variable | Set by | Description |
|---|---|---|
| `BEDROCK_AGENT_ID` | SAM template | Bedrock Agent ID |
| `BEDROCK_AGENT_ALIAS_ID` | SAM template | Agent Alias ID |
| `BEDROCK_KB_ID` | `.env` | Knowledge Base ID (sync script only) |
| `BEDROCK_DATA_SOURCE_ID` | `.env` | Data Source ID (sync script only) |
| `LOG_LEVEL` | SAM template | Python log level (default: INFO) |

---

## Project Structure

```
Bedrock_RAG_Agent/
├── src/
│   └── lambda_function.py   # Lambda handler — invokes Bedrock Agent
├── scripts/
│   └── sync_kb.py           # Trigger KB ingestion after uploading docs
├── template.yaml            # AWS SAM template (all infrastructure)
├── requirements.txt         # Python deps (boto3, python-dotenv)
├── .env.example             # Environment variable template
└── README.md                # This file
```

---

## Cost Notes

- **OpenSearch Serverless**: minimum ~$700/month (2 OCUs always-on) — consider this for production only
- **Bedrock Agent**: pay per token (input + output)
- **Titan Embeddings**: pay per token at ingestion time
- **Lambda + API Gateway**: effectively free at low volume

For dev/test, consider using the [Bedrock console chat](https://console.aws.amazon.com/bedrock/) to test the agent before deploying the Lambda API.
