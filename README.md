# AI Agent Deployment Hub

A collection of production-ready AI agent architectures deployed on AWS. Each project demonstrates a different agentic pattern, tooling stack, and deployment approach, from custom RAG pipelines to fully managed AWS-native solutions.

## Projects

| Directory | Pattern | LLM | RAG | Deployment |
|---|---|---|---|---|
| [`Agentic_RAG/`](#agentic_rag) | Agentic RAG | OpenAI gpt-4o | Custom FAISS + S3 | Lambda + API Gateway (SAM) |
| [`Bedrock_RAG_Agent/`](#bedrock_rag_agent) | Agentic RAG | Claude 3.5 Sonnet (Bedrock) | Bedrock Knowledge Base | Lambda + API Gateway (SAM) |
| `Knowledge_Graph_Agent/` | Knowledge Graph Agent | — | — | — |
| `MCP-Powered_LLM_Agent/` | MCP Agent | — | — | — |
| `multi-agent/` | Multi-Agent | — | — | — |

---

## Agentic_RAG

**Stack:** OpenAI gpt-4o · FAISS · S3 · AWS Lambda · API Gateway · SAM

A custom agentic RAG system where the agent iteratively calls a `search_knowledge_base` tool (1–3 times) via OpenAI function calling, then synthesises a final answer. The FAISS vector index is built locally, stored in S3, and cached in `/tmp` across Lambda warm starts.

```
User → API Gateway → Lambda
                        │
                    agent.py (OpenAI tool calling loop)
                        │
                    retriever.py ──► FAISS index (cached from S3)
```

**When to use:** You want full control over chunking, embedding, and retrieval logic, or you need to use OpenAI models.

---

## Bedrock_RAG_Agent

**Stack:** Claude 3.5 Sonnet (Bedrock) · Titan Embeddings v2 · OpenSearch Serverless · Bedrock Knowledge Base · AWS Lambda · API Gateway · SAM

A fully managed agentic RAG system. The Bedrock Agent handles the agentic loop natively — deciding when to search the Knowledge Base (the built-in RAG tool), how many times, and when it has enough context to answer. No custom retrieval code.

```
User → API Gateway → Lambda
                        │
                    Bedrock Agent (Claude 3.5 Sonnet)
                        │
                    Knowledge Base (built-in RAG tool)
                        │
              S3 (docs) + OpenSearch Serverless (vectors)
```

**When to use:** You want a managed RAG pipeline with no OpenAI dependency, and you are comfortable with the OpenSearch Serverless cost floor.

---

## Architecture Comparison

### Agentic_RAG vs Bedrock_RAG_Agent

| | Agentic_RAG | Bedrock_RAG_Agent |
|---|---|---|
| **LLM** | OpenAI gpt-4o | Claude 3.5 Sonnet (Bedrock) |
| **Embeddings** | OpenAI text-embedding-3-small | Amazon Titan Embeddings v2 |
| **Vector store** | FAISS (self-managed, cached in /tmp) | OpenSearch Serverless (fully managed) |
| **RAG pipeline** | Custom (retriever.py + agent.py) | Bedrock Knowledge Base (managed) |
| **Agentic loop** | Manual OpenAI function calling loop | Built into Bedrock Agent |
| **Tool calling** | OpenAI `tools` API | Bedrock Agent Action Groups |
| **External API key** | OpenAI API key required | None — IAM only |
| **Code to maintain** | ~300 lines | ~80 lines |
| **Chunking control** | Full (size, overlap, strategy) | Limited (fixed-size or semantic) |
| **Retrieval control** | Full (re-rank, filter, top-k) | Limited |
| **Cold start** | Downloads FAISS index from S3 | No index to load |
| **Index update** | Re-run indexer.py, re-upload | Upload to S3 + trigger sync job |
| **Cost model** | Pay per OpenAI API call | Pay per token + OSS (~$700/mo min) |
| **Best for** | Full control, any LLM, cost-sensitive | Managed infra, AWS-native, production |

### Key Trade-off

> **Agentic_RAG** gives you maximum flexibility — swap the LLM, tune the retriever, change the chunking strategy — at the cost of more code to own and operate.
>
> **Bedrock_RAG_Agent** eliminates the RAG plumbing entirely and is deeply integrated with AWS IAM/security, but the OpenSearch Serverless cost floor makes it unsuitable for experimentation or low-traffic use cases.

---

## Repository Structure

```
AI_Agent_Deployment_Hub/
├── Agentic_RAG/             # Custom FAISS RAG + OpenAI tool calling
│   ├── src/
│   │   ├── lambda_function.py
│   │   ├── agent.py
│   │   ├── retriever.py
│   │   └── indexer.py
│   ├── template.yaml
│   ├── requirements.txt
│   └── .env.example
│
├── Bedrock_RAG_Agent/       # Managed RAG via Bedrock Agent + Knowledge Base
│   ├── src/
│   │   └── lambda_function.py
│   ├── scripts/
│   │   └── sync_kb.py
│   ├── template.yaml
│   ├── requirements.txt
│   └── .env.example
│
├── Knowledge_Graph_Agent/
├── MCP-Powered_LLM_Agent/
└── multi-agent/
```
