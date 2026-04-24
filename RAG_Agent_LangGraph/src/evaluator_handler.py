"""
evaluator_handler.py — Lambda handler for async evaluation (Layers 1, 2 & 3).

Invoked asynchronously (InvocationType=Event) by the main RAG Lambda
immediately after delivering the response to the user — so evaluation
never adds latency to the user-facing request.

Expected event payload:
  {
    "response_id": "<uuid>",
    "question":    "What is ...?",
    "answer":      "...",
    "documents":   ["chunk1", "chunk2", ...],
    "node_path":   "retrieve → grade(YES) → generate"
  }

What this handler does:
  Layer 1 — computes RAGAS-style metrics (faithfulness, answer_relevance,
             context_precision) via evaluator.run_evaluation()
  Layer 2 — runs LLM-as-a-Judge scoring (correctness, completeness,
             groundedness, clarity) via evaluator.run_evaluation()
  Layer 3 — persists results to DynamoDB and flags low-quality responses
             for human review via storage.flag_for_review()
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
    # When invoked via InvocationType=Event the payload may be a raw dict
    # or a JSON string depending on how boto3 serialised it.
    if isinstance(event, str):
        event = json.loads(event)

    response_id = event.get("response_id", "unknown")
    question    = event.get("question",    "")
    answer      = event.get("answer",      "")
    documents   = event.get("documents",   [])
    node_path   = event.get("node_path",   "")

    logger.info("evaluator_handler: start response_id=%s", response_id)

    # ── Layers 1 & 2: run all metrics ────────────────────────────────────────
    result = run_evaluation(question=question, documents=documents, answer=answer)

    # ── Persist evaluation record to DynamoDB ─────────────────────────────────
    storage.log_evaluation(
        response_id=response_id,
        question=question,
        answer=answer,
        documents=documents,
        evaluation=result.to_dict(),
        node_path=node_path,
    )

    # ── Layer 3: flag for human review if below threshold ─────────────────────
    if not result.passes_threshold:
        storage.flag_for_review(response_id, reason="threshold_failure")
        logger.warning(
            "evaluator_handler: response_id=%s flagged — %s",
            response_id,
            result.failure_reasons,
        )

    logger.info(
        "evaluator_handler: done response_id=%s passes=%s scores=%s",
        response_id,
        result.passes_threshold,
        {
            "faithfulness":      result.faithfulness,
            "answer_relevance":  result.answer_relevance,
            "context_precision": result.context_precision,
            "judge_score":       result.judge_score,
        },
    )

    return {"status": "ok", "passes_threshold": result.passes_threshold}
