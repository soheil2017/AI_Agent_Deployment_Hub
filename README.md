# AI Agent Deployment Hub

A collection of production-ready AI agent architectures deployed on AWS. Each project demonstrates a different agentic pattern, tooling stack, and deployment approach, from custom RAG pipelines to fully managed AWS-native solutions.

## Projects

| Directory | Pattern | LLM | RAG | Deployment |
|---|---|---|---|---|
| [`Agentic_RAG/`](#agentic_rag) | Agentic RAG | OpenAI gpt-4o | Custom FAISS + S3 | Lambda + API Gateway (SAM) |
| [`Bedrock_RAG_Agent/`](#bedrock_rag_agent) | Agentic RAG | Claude 3.5 Sonnet (Bedrock) | Bedrock Knowledge Base | Lambda + API Gateway (SAM) |
| [`LangGraph_RAG/`](#langgraph_rag) | Graph RAG | OpenAI gpt-4o-mini | Custom FAISS + S3 | Lambda + API Gateway (SAM) |
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

## LangGraph_RAG

**Stack:** OpenAI gpt-4o-mini · FAISS · S3 · LangGraph · AWS Lambda · API Gateway · SAM

A RAG service built as an explicit [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`. The graph is compiled once at module level and reused across Lambda warm starts. The flow is fixed (retrieve → generate) but the typed state and graph structure make it trivial to add grading, query rewriting, or multi-agent patterns later.

```
User → API Gateway → Lambda
                        │
                    graph.py (LangGraph StateGraph, compiled at module level)
                        │
              ┌─────────┴──────────┐
           retrieve             generate
              │                     │
          retriever.py ──► FAISS  gpt-4o-mini
                           (cached from S3)
```

**Graph topology:** `START → retrieve → generate → END`

**When to use:** You want the simplicity of a fixed RAG pipeline but with an explicit, visualisable, and extensible graph structure — a natural stepping stone toward CRAG or multi-agent patterns.

---

## Architecture Comparison

### All three projects compared

| | Agentic_RAG | LangGraph_RAG | Bedrock_RAG_Agent |
|---|---|---|---|
| **LLM** | OpenAI gpt-4o | OpenAI gpt-4o-mini | Claude 3.5 Sonnet (Bedrock) |
| **Embeddings** | OpenAI text-embedding-3-small | OpenAI text-embedding-3-small | Amazon Titan Embeddings v2 |
| **Vector store** | FAISS (self-managed, /tmp) | FAISS (self-managed, /tmp) | OpenSearch Serverless (managed) |
| **RAG pipeline** | Custom (retriever.py + agent.py) | LangGraph StateGraph | Bedrock Knowledge Base (managed) |
| **Flow control** | Dynamic OpenAI tool-calling loop | Fixed graph (retrieve → generate) | Built into Bedrock Agent |
| **Flow structure** | Implicit (agent decides) | Explicit typed StateGraph | Opaque (managed) |
| **Extensibility** | Add OpenAI tools | Add graph nodes / edges | Limited to Bedrock features |
| **Visualisable** | No | Yes (`draw_mermaid()`) | No |
| **External API key** | OpenAI | OpenAI | None — IAM only |
| **Lambda memory** | 1024 MB | 512 MB | 512 MB |
| **Lambda timeout** | 120 s | 60 s | 60 s |
| **Cold start** | Downloads FAISS from S3 | Downloads FAISS from S3 | No index to load |
| **Index update** | Re-run indexer.py, re-upload | Re-run indexer.py, re-upload | Upload to S3 + sync job |
| **Cost model** | Pay per OpenAI call | Pay per OpenAI call | Pay per token + OSS (~$700/mo min) |
| **Code to maintain** | ~300 lines | ~200 lines | ~80 lines |
| **Best for** | Full control, multi-turn loops | Explicit flow, easy to extend | Managed infra, AWS-native |

### Key Trade-offs

> **Agentic_RAG** gives maximum flexibility — the agent dynamically decides how many times to retrieve, supports multi-turn tool use, and can be extended with any OpenAI tool. Higher memory and timeout needed.
>
> **LangGraph_RAG** makes the data flow explicit and typed. Same FAISS/S3 stack as Agentic_RAG but the graph structure is self-documenting and easy to extend (add grading, rewriting, routing) without touching the Lambda handler.
>
> **Bedrock_RAG_Agent** eliminates all RAG plumbing and is deeply integrated with AWS IAM/security, but the OpenSearch Serverless cost floor makes it unsuitable for experimentation or low-traffic use cases.

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
├── LangGraph_RAG/           # LangGraph StateGraph RAG + FAISS + S3
│   ├── src/
│   │   ├── lambda_function.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── retriever.py
│   ├── template.yaml
│   ├── requirements.txt
│   └── .env.example
│
├── Knowledge_Graph_Agent/
├── MCP-Powered_LLM_Agent/
└── multi-agent/
```
