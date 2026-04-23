# RAG Agent with LangGraph

A Retrieval-Augmented Generation (RAG) agent implemented as an explicit
[LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`, deployed
on AWS Lambda + API Gateway via SAM.

## Graph

```
START → retrieve → grade_documents → [relevant?] → YES → generate → END
                                                  → NO  → no_answer → END
```

```
RAGState
├── question:  str        # input
├── documents: list[str]  # populated by retrieve
├── relevant:  bool       # populated by grade_documents
└── answer:    str        # populated by generate or no_answer
```

The graph is compiled **once at module level** and reused across Lambda warm
starts — equivalent to a singleton service object.

## How it compares to Agentic_RAG

| Aspect | Agentic_RAG | RAG_Agent_LangGraph |
|---|---|---|
| Flow control | OpenAI tool-calling loop | LangGraph StateGraph |
| Steps | Dynamic (agent decides) | Conditional: retrieve → grade → generate/no_answer |
| Visualisable | No | Yes (`app.get_graph().draw_mermaid()`) |
| Extensible | Add tools | Add nodes / edges |
| LLM calls | ≥ 2 (plan + answer) | 2-3 (embed + grade + generate) |
| Model | gpt-4o | gpt-4o-mini |

Both projects produce equivalent answers for a single-turn RAG query. The
LangGraph version makes the data flow explicit and typed, with conditional
routing to skip generation when retrieved documents are not relevant.

## Project Structure

```
RAG_Agent_LangGraph/
├── src/
│   ├── lambda_function.py   # Lambda handler
│   ├── graph.py             # StateGraph compiled at module level
│   ├── nodes.py             # RAGState, retrieve(), grade_documents(), generate(), no_answer()
│   └── retriever.py         # FAISS + S3 (same as Agentic_RAG)
├── template.yaml            # SAM: Lambda + HTTP API Gateway
├── requirements.txt
├── .env.example
└── README.md
```

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

# Test the graph directly
cd src
python -c "
from graph import app
result = app.invoke({'question': 'What is ...?', 'documents': [], 'answer': ''})
print(result['answer'])
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

## Usage

```bash
# Query
curl -X POST https://<ApiUrl>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ...?"}'

# Health check
curl https://<ApiUrl>/health
```

Response:
```json
{"answer": "..."}
```

## Extending the Graph

Add a **grading** node that re-retrieves if documents are irrelevant:

```python
from langgraph.graph import StateGraph, START, END
from nodes import RAGState, retrieve, generate

def grade_documents(state: RAGState) -> str:
    # return "generate" if relevant, "retrieve" if not
    ...

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", grade_documents)
builder.add_edge("generate", END)
app = builder.compile()
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `S3_BUCKET` | — | S3 bucket name |
| `INDEX_KEY` | `faiss_index/index.faiss` | S3 key for FAISS index |
| `CHUNKS_KEY` | `faiss_index/chunks.pkl` | S3 key for chunk list |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `LOG_LEVEL` | `INFO` | Python logging level |
