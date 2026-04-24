"""
evaluator_handler.py — Lambda handler for async evaluation (Layers 1, 2 & 3).

Invoked asynchronously (InvocationType=Event) by the main Agentic RAG Lambda
immediately after delivering the response to the user — so evaluation
never adds latency to the user-facing request.

Expected event payload:
  {
    "response_id":      "<uuid>",
    "question":         "What is ...?",
    "answer":           "...",
    "documents":        ["chunk1", "chunk2", ...],   ← all chunks across all rounds
    "tool_rounds_used": 2,
    "node_path":        "agentic-loop (2 tool round(s))"
  }

What this handler does:
  Layer 1 — RAGAS-style metrics (faithfulness, answer_relevance, context_precision)
  Layer 2 — LLM-as-a-Judge (correctness, completeness, groundedness, clarity)
  Layer 3 — flags low-quality responses for human review in DynamoDB

Agentic_RAG extra metric:
  tool_rounds_used — number of search rounds the agent used (1–3).
  Stored in the evaluation record for cost and efficiency monitoring.
  High rounds on simple questions may indicate retrieval quality issues.
"""

import json
import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from evaluator import run_evaluation
import storage

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def handler(event: dict, context) -> dict:
    """
    Async evaluation handler.
    Return value is ignored by the caller (InvocationType=Event).
    """
    if isinstance(event, str):
        event = json.loads(event)

    response_id      = event.get("response_id",      "unknown")
    question         = event.get("question",          "")
    answer           = event.get("answer",            "")
    documents        = event.get("documents",         [])
    tool_rounds_used = event.get("tool_rounds_used",  0)
    node_path        = event.get("node_path",         "")

    logger.info(
        "evaluator_handler: start response_id=%s tool_rounds=%d",
        response_id, tool_rounds_used,
    )

    # ── Layers 1 & 2: run all metrics ────────────────────────────────────────
    result = run_evaluation(question=question, documents=documents, answer=answer)

    # ── Add Agentic_RAG-specific metric to the evaluation record ─────────────
    eval_dict = result.to_dict()
    eval_dict["tool_rounds_used"] = tool_rounds_used   # extra metric unique to this project

    # ── Persist evaluation record to DynamoDB ─────────────────────────────────
    storage.log_evaluation(
        response_id=response_id,
        question=question,
        answer=answer,
        documents=documents,
        evaluation=eval_dict,
        node_path=node_path,
    )

    # ── Layer 3: flag for human review if below threshold ─────────────────────
    if not result.passes_threshold:
        storage.flag_for_review(response_id, reason="threshold_failure")
        logger.warning(
            "evaluator_handler: response_id=%s flagged — %s",
            response_id, result.failure_reasons,
        )

    logger.info(
        "evaluator_handler: done response_id=%s passes=%s tool_rounds=%d scores=%s",
        response_id,
        result.passes_threshold,
        tool_rounds_used,
        {
            "faithfulness":      result.faithfulness,
            "answer_relevance":  result.answer_relevance,
            "context_precision": result.context_precision,
            "judge_score":       result.judge_score,
            "tool_rounds_used":  tool_rounds_used,
        },
    )

    return {"status": "ok", "passes_threshold": result.passes_threshold}
