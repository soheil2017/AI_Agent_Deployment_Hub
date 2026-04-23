# AI Agent Deployment Hub

A collection of production-ready AI agent architectures. Each project demonstrates a different agentic pattern, tooling stack, and deployment approach — from custom RAG pipelines on AWS to MCP-powered agents on Vercel.

## Projects

| Directory | Pattern | LLM | Tools / RAG | Deployment |
|---|---|---|---|---|
| [`Agentic_RAG/`](#agentic_rag) | Agentic RAG | OpenAI gpt-4o | Custom FAISS + S3 | Lambda + API Gateway (SAM) |
| [`Bedrock_RAG_Agent/`](#bedrock_rag_agent) | Agentic RAG | Claude 3.5 Sonnet (Bedrock) | Bedrock Knowledge Base | Lambda + API Gateway (SAM) |
| [`RAG_Agent_LangGraph/`](#rag_agent_langgraph) | RAG Agent | OpenAI gpt-4o-mini | Custom FAISS + S3 | Lambda + API Gateway (SAM) |
| [`MCP-Powered_LLM_Agent/`](#mcp-powered_llm_agent) | MCP Tool Agent | OpenAI gpt-4o-mini | Web Search · Weather · Calculator (MCP) | Vercel (Next.js) |
| `Agentic_GraphRAG/` | Agentic GraphRAG | — | — | — |
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

## RAG_Agent_LangGraph

**Stack:** OpenAI gpt-4o-mini · FAISS · S3 · LangGraph · AWS Lambda · API Gateway · SAM

A RAG agent built as an explicit [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`. The graph is compiled once at module level and reused across Lambda warm starts. After retrieval, a grading node decides whether the documents are relevant before calling the LLM — skipping generation entirely when retrieval fails.

```
User → API Gateway → Lambda
                        │
                    graph.py (LangGraph StateGraph, compiled at module level)
                        │
              ┌─────────┴──────────┐
           retrieve          grade_documents
              │                     │
          retriever.py ──► FAISS   YES → generate → answer
                           (cached  NO  → no_answer
                           from S3)
```

**Graph topology:** `START → retrieve → grade_documents → [YES] → generate → END`
                                                       `→ [NO]  → no_answer → END`

**When to use:** You want an explicit, visualisable RAG pipeline with conditional routing — a natural stepping stone toward CRAG or multi-agent patterns.

---

## MCP-Powered_LLM_Agent

**Stack:** OpenAI gpt-4o-mini · Model Context Protocol (MCP) · Next.js 15 · Vercel AI SDK · Tailwind CSS · Vercel

A chat-based AI research agent that connects to real-time tools via the **Model Context Protocol (MCP)**. Each tool is an independent MCP server (a Next.js API route) implementing JSON-RPC 2.0. The agent discovers all registered tools at startup and decides which to call — enabling multi-step tool chains in a single response.

```
Browser (Chat UI)
  │
  └──► POST /api/chat  (Vercel AI SDK streamText + OpenAI)
            │
            ├──► POST /api/mcp/search      → Tavily web search
            ├──► POST /api/mcp/weather     → Open-Meteo (free, no key)
            └──► POST /api/mcp/calculator  → mathjs (precise math)
            │
            ◄── streams response with tool call events
```

**Scalability design:** Adding a new tool requires only:
1. Creating `app/api/mcp/<name>/route.ts` (the MCP server)
2. Adding one entry to `config/mcp.config.ts` (the registry)

The agent, chat logic, and UI require zero changes.

**Memory:** Session-based conversation history stored server-side (in-memory Map in dev, Vercel KV in production). Session ID persists in `localStorage` — conversations survive page refreshes.

| | Detail |
|---|---|
| **LLM** | OpenAI gpt-4o-mini |
| **Protocol** | MCP (JSON-RPC 2.0 over HTTP POST) |
| **Tools** | Web Search (Tavily) · Weather (Open-Meteo) · Calculator (mathjs) |
| **Memory** | In-memory Map → Vercel KV (drop-in upgrade) |
| **Deployment** | Vercel serverless (Next.js App Router) |
| **Max tool chain** | 5 steps per response (`maxSteps: 5`) |

**When to use:** You want a chat agent that can call real-world APIs with full tool-use transparency, and you need it deployable to Vercel (not AWS) with a clean path to adding new tools over time.

---

## Architecture Comparison

### All projects compared

| | Agentic_RAG | RAG_Agent_LangGraph | Bedrock_RAG_Agent | MCP-Powered_LLM_Agent |
|---|---|---|---|---|
| **LLM** | OpenAI gpt-4o | OpenAI gpt-4o-mini | Claude 3.5 Sonnet (Bedrock) | OpenAI gpt-4o-mini |
| **Tool / RAG mechanism** | OpenAI function calling | LangGraph StateGraph | Bedrock Knowledge Base | MCP (JSON-RPC 2.0) |
| **Vector store** | FAISS (S3-cached) | FAISS (S3-cached) | OpenSearch Serverless | None |
| **Flow control** | Dynamic tool-calling loop | Conditional graph (retrieve → grade → generate/no_answer) | Managed by Bedrock Agent | Dynamic tool-calling loop |
| **Flow structure** | Implicit (agent decides) | Explicit typed StateGraph | Opaque (managed) | Implicit (agent decides) |
| **Extensibility** | Add OpenAI tools | Add graph nodes / edges | Limited to Bedrock features | Add 1 file + 1 config line |
| **Memory** | None | None | None | Session-based (→ Vercel KV) |
| **External API keys** | OpenAI | OpenAI | None — IAM only | OpenAI + Tavily |
| **Deployment** | AWS Lambda + API Gateway | AWS Lambda + API Gateway | AWS Lambda + API Gateway | Vercel (Next.js) |
| **Cold start** | Downloads FAISS from S3 | Downloads FAISS from S3 | No index to load | No index to load |
| **Cost model** | Pay per OpenAI call | Pay per OpenAI call | Pay per token + OSS (~$700/mo min) | Pay per OpenAI call; Vercel free tier |
| **Code to maintain** | ~300 lines | ~200 lines | ~80 lines | ~500 lines |
| **Best for** | Full control, multi-turn loops | Explicit flow, conditional routing | Managed infra, AWS-native | Real-time tools, fast iteration, Vercel |

### Key Trade-offs

> **Agentic_RAG** gives maximum flexibility — the agent dynamically decides how many times to retrieve, supports multi-turn tool use, and can be extended with any OpenAI tool. Higher memory and timeout needed.
>
> **RAG_Agent_LangGraph** makes the data flow explicit and typed. Same FAISS/S3 stack as Agentic_RAG but with conditional routing — a grading node skips generation when retrieved documents are irrelevant, saving LLM cost and returning a clear fallback message.
>
> **Bedrock_RAG_Agent** eliminates all RAG plumbing and is deeply integrated with AWS IAM/security, but the OpenSearch Serverless cost floor makes it unsuitable for experimentation or low-traffic use cases.
>
> **MCP-Powered_LLM_Agent** focuses on real-time external tools (not RAG over a static corpus). The MCP architecture makes adding new tools trivially easy — no changes to agent logic. Conversation memory and a streaming chat UI are included out of the box. Best choice when you want Vercel deployment and a growing tool set.

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
├── RAG_Agent_LangGraph/     # LangGraph StateGraph RAG agent + FAISS + S3
│   ├── src/
│   │   ├── lambda_function.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── retriever.py
│   ├── template.yaml
│   ├── requirements.txt
│   └── .env.example
│
│
├── MCP-Powered_LLM_Agent/   # MCP tool agent + chat UI (Next.js + Vercel)
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat/route.ts         # Agent endpoint (streamText + MCP tools)
│   │   │   ├── memory/route.ts       # Conversation persistence API
│   │   │   └── mcp/
│   │   │       ├── search/route.ts   # MCP server: Tavily web search
│   │   │       ├── weather/route.ts  # MCP server: Open-Meteo weather
│   │   │       └── calculator/route.ts # MCP server: mathjs calculator
│   │   └── page.tsx                  # Chat UI with tool call badges
│   ├── config/mcp.config.ts          # MCP server registry (add tools here)
│   ├── lib/
│   │   ├── memory.ts                 # Session memory (in-memory → Vercel KV)
│   │   └── mcp-client.ts             # MCP client factory
│   └── vercel.json
│
├── Agentic_GraphRAG/
└── multi-agent/
```
