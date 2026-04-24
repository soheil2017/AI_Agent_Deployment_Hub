"""
storage.py — DynamoDB persistence for the hybrid evaluation framework.

Tables:
  RAGEvaluationLogs  — one record per response: Layer 1 + 2 scores, node path,
                       and human-review flag (Layer 3).
  RAGUserFeedback    — user thumbs-up / thumbs-down per response (Layer 4).

All writes are best-effort: a DynamoDB failure is logged but never raises,
so it cannot block the user-facing response.
"""

import logging
import os
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

EVAL_TABLE_NAME     = os.environ.get("EVAL_TABLE_NAME",     "RAGEvaluationLogs")
FEEDBACK_TABLE_NAME = os.environ.get("FEEDBACK_TABLE_NAME", "RAGUserFeedback")

_dynamodb = boto3.resource("dynamodb")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Layer 1 + 2: log evaluation results ──────────────────────────────────────

def log_evaluation(
    response_id: str,
    question: str,
    answer: str,
    documents: list[str],
    evaluation: dict,
    node_path: str = "",
) -> None:
    """
    Persist a full evaluation record for a response.
    Called after the graph completes, regardless of which path was taken.
    """
    try:
        _dynamodb.Table(EVAL_TABLE_NAME).put_item(Item={
            "response_id":        response_id,
            "timestamp":          _now(),
            "question":           question,
            "answer":             answer,
            "documents":          documents,
            "evaluation":         evaluation,
            "passes_threshold":   evaluation.get("passes_threshold", False),
            "node_path":          node_path,
            "flagged_for_review": False,
        })
        logger.info("log_evaluation: stored response_id=%s path=%s", response_id, node_path)
    except Exception as exc:
        logger.warning("log_evaluation: DynamoDB write failed — %s", exc)


# ── Layer 3: flag for human review ───────────────────────────────────────────

def flag_for_review(response_id: str, reason: str) -> None:
    """
    Mark a response for human review.  Triggered when:
      - The threshold gate rejects an answer  (reason = "threshold_failure")
      - A user submits a thumbs-down          (reason = "user_thumbs_down")
    """
    try:
        _dynamodb.Table(EVAL_TABLE_NAME).update_item(
            Key={"response_id": response_id},
            UpdateExpression=(
                "SET flagged_for_review = :t, "
                "    flag_reason        = :r, "
                "    flag_timestamp     = :ts"
            ),
            ExpressionAttributeValues={
                ":t":  True,
                ":r":  reason,
                ":ts": _now(),
            },
        )
        logger.info("flag_for_review: response_id=%s reason=%s", response_id, reason)
    except Exception as exc:
        logger.warning("flag_for_review: DynamoDB update failed — %s", exc)


# ── Layer 4: user feedback ────────────────────────────────────────────────────

def store_feedback(
    response_id: str,
    rating: str,       # "thumbs_up" | "thumbs_down"
    comment: str = "",
) -> None:
    """
    Persist user feedback.  Thumbs-down automatically triggers a human-review flag
    so reviewers have a unified queue in the EvaluationLogs table.
    """
    try:
        _dynamodb.Table(FEEDBACK_TABLE_NAME).put_item(Item={
            "response_id": response_id,
            "timestamp":   _now(),
            "rating":      rating,
            "comment":     comment,
        })
        logger.info("store_feedback: response_id=%s rating=%s", response_id, rating)

        if rating == "thumbs_down":
            flag_for_review(response_id, reason="user_thumbs_down")

    except Exception as exc:
        logger.warning("store_feedback: DynamoDB write failed — %s", exc)
