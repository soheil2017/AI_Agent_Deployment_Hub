"""
main.py — FastAPI application entry point.

Deployed on Vercel (serverless) or Railway (container).

Endpoints:
  POST /query     — run the Graph RAG agent, return answer immediately,
                    then run evaluation as a FastAPI background task
  POST /feedback  — store user rating in Langfuse (Layer 4)
  GET  /health    — health check

Observability:
  Every /query invocation creates a Langfuse trace.
  The Langfuse CallbackHandler is passed to LangGraph — it automatically
  creates spans for each node (classify_query, run_tools, generate).
  Evaluation scores (faithfulness, judge_score, etc.) are logged to the
  same trace after the response is delivered.

Query request (text):
  { "question": "Which patients have a pending prior auth?" }

Query request (with image for VLM):
  { "question": "Analyze this X-ray",
    "image_base64": "<base64-string>",
    "media_type": "image/jpeg" }

Query response:
  { "answer": "...", "trace_id": "<uuid>", "query_type": "relational" }
"""

import logging
import os
import uuid

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langfuse.callback import CallbackHandler as LangfuseCallback

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from graph import app as rag_app
import evaluator_handler

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────

api = FastAPI(
    title="Conduit Graph RAG Agent",
    description="Dental healthcare interoperability assistant — Neo4j + ChromaDB + Claude Vision",
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:     str
    image_base64: str | None = None   # base64 image for VLM queries
    media_type:   str        = "image/jpeg"

class QueryResponse(BaseModel):
    answer:     str
    trace_id:   str
    query_type: str

class FeedbackRequest(BaseModel):
    trace_id: str
    rating:   str    # "thumbs_up" | "thumbs_down"
    comment:  str = ""

class FeedbackResponse(BaseModel):
    status: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Run the Graph RAG agent and return the answer immediately.
    Evaluation (Layers 1–3) runs as a background task after the response.

    Langfuse tracing is automatic — the CallbackHandler instruments every
    LangGraph node and logs token usage, latency, and inputs/outputs.
    """
    trace_id = str(uuid.uuid4())

    # Langfuse callback — auto-traces every LangGraph node
    langfuse_handler = LangfuseCallback(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        trace_id=trace_id,
        session_id=trace_id,
        user_id="conduit-api",
        tags=["production", "dental-rag"],
        metadata={"question": request.question},
    )

    # Build initial LangGraph state
    initial_state: dict = {"response_id": trace_id, "question": request.question}
    if request.image_base64:
        initial_state["image_base64"] = request.image_base64
        initial_state["media_type"]   = request.media_type

    # Invoke the graph — Langfuse automatically records each node as a span
    final_state = rag_app.invoke(
        initial_state,
        config={"callbacks": [langfuse_handler]},
    )

    answer     = final_state.get("answer", "")
    query_type = final_state.get("query_type", "")
    documents  = final_state.get("documents", [])

    # Flush Langfuse spans before firing the background task
    langfuse_handler.flush()

    # Run evaluation asynchronously — scores are sent to Langfuse after response
    background_tasks.add_task(
        evaluator_handler.run,
        trace_id=trace_id,
        question=request.question,
        answer=answer,
        documents=documents,
        query_type=query_type,
    )

    logger.info("query: trace_id=%s query_type=%s", trace_id, query_type)
    return QueryResponse(answer=answer, trace_id=trace_id, query_type=query_type)


@api.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """
    Layer 4 — store user thumbs-up/thumbs-down as a Langfuse score.
    Thumbs-down automatically adds a 'needs_review' comment in Langfuse.
    """
    if request.rating not in ("thumbs_up", "thumbs_down"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="rating must be 'thumbs_up' or 'thumbs_down'")

    evaluator_handler.log_user_feedback(
        trace_id=request.trace_id,
        rating=request.rating,
        comment=request.comment,
    )
    logger.info("feedback: trace_id=%s rating=%s", request.trace_id, request.rating)
    return FeedbackResponse(status="ok")
