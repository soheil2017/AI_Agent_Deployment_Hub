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
  { "answer": "...", "trace_id": "<uuid>", "query_type": "relational",
    "low_confidence": false }

  low_confidence=True means the sync faithfulness gate fired before delivery —
  the answer may not be fully grounded in retrieved context.
"""

import logging
import os
import uuid

# Configure logging FIRST so all module-level init messages are visible
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallback
    _langfuse_enabled = True
except Exception:
    _langfuse_enabled = False

from graph import app as rag_app
from evaluator import check_faithfulness, THRESHOLD_FAITHFULNESS
import evaluator_handler

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
    answer:         str
    trace_id:       str
    query_type:     str
    low_confidence: bool = False

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
    trace_id = str(uuid.uuid4())
    try:
        # Build initial LangGraph state
        initial_state: dict = {"response_id": trace_id, "question": request.question}
        if request.image_base64:
            initial_state["image_base64"] = request.image_base64
            initial_state["media_type"]   = request.media_type

        # Attach Langfuse callback if available — pass our trace_id so that
        # create_score() in evaluator_handler uses the same trace.
        callbacks = []
        if _langfuse_enabled:
            try:
                callbacks = [LangfuseCallback(trace_id=trace_id)]
            except Exception as e:
                logger.warning("Langfuse callback init failed: %s", e)

        # Invoke the graph
        final_state = rag_app.invoke(
            initial_state,
            config={"callbacks": callbacks} if callbacks else {},
        )

        answer     = final_state.get("answer", "")
        query_type = final_state.get("query_type", "")
        documents  = final_state.get("documents", [])

        # Layer 0 — sync faithfulness gate (runs before the response is sent)
        # Skip for visual queries: the VLM answer isn't grounded in text chunks.
        faith_score    = 1.0
        low_confidence = False
        if query_type != "visual" and documents and answer:
            faith_score    = check_faithfulness(documents=documents, answer=answer)
            low_confidence = faith_score < THRESHOLD_FAITHFULNESS
            if low_confidence:
                logger.warning(
                    "query: low_confidence trace_id=%s faithfulness=%.2f",
                    trace_id, faith_score,
                )

        # Full evaluation (Layers 1–3) runs async — no user latency impact.
        # Pass the pre-computed faith_score so it isn't computed twice.
        background_tasks.add_task(
            evaluator_handler.run,
            trace_id=trace_id,
            question=request.question,
            answer=answer,
            documents=documents,
            query_type=query_type,
            precomputed_faithfulness=faith_score,
        )

        logger.info(
            "query: trace_id=%s query_type=%s low_confidence=%s",
            trace_id, query_type, low_confidence,
        )
        return QueryResponse(
            answer=answer,
            trace_id=trace_id,
            query_type=query_type,
            low_confidence=low_confidence,
        )

    except Exception as exc:
        logger.exception("query failed: %s", exc)
        raise


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
