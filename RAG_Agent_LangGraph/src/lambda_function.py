"""
lambda_function.py — AWS Lambda entry point / API Gateway HTTP API handler.

Endpoints:
  POST /query     — run the RAG graph, deliver answer immediately, then fire
                    async evaluation (no blocking on metrics)
  POST /feedback  — store user thumbs-up/thumbs-down (Layer 4)
  GET  /health    — health check

Query request:   { "question": "What is ...?" }
Query response:  { "answer": "...", "response_id": "<uuid>" }

Feedback request:
  { "response_id": "<uuid>", "rating": "thumbs_up" | "thumbs_down", "comment": "" }
Feedback response: { "status": "ok" }

Async evaluation flow:
  After delivering the response, this handler invokes EvaluatorFunction
  asynchronously (InvocationType=Event — fire-and-forget).
  The evaluator runs Layers 1 & 2, stores results to DynamoDB, and flags
  low-quality responses for human review (Layer 3).
"""

import json
import logging
import os
import uuid

import boto3

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from graph import app
import storage

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

EVALUATOR_FUNCTION_NAME = os.environ.get("EVALUATOR_FUNCTION_NAME", "")

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}

_lambda_client = boto3.client("lambda")


def _respond(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def _node_path(state: dict) -> str:
    """Infer the graph path taken from the final state."""
    if not state.get("relevant"):
        return "retrieve → grade(NO) → no_answer"
    return "retrieve → grade(YES) → generate"


def _fire_evaluator(
    response_id: str,
    question:    str,
    answer:      str,
    documents:   list[str],
    node_path:   str,
) -> None:
    """
    Asynchronously invoke the EvaluatorFunction (fire-and-forget).
    The main Lambda returns its response before evaluation completes.
    Silently skips if EVALUATOR_FUNCTION_NAME is not configured.
    """
    if not EVALUATOR_FUNCTION_NAME:
        logger.warning("_fire_evaluator: EVALUATOR_FUNCTION_NAME not set — skipping")
        return
    try:
        _lambda_client.invoke(
            FunctionName=EVALUATOR_FUNCTION_NAME,
            InvocationType="Event",   # async — does not wait for result
            Payload=json.dumps({
                "response_id": response_id,
                "question":    question,
                "answer":      answer,
                "documents":   documents,
                "node_path":   node_path,
            }).encode(),
        )
        logger.info("_fire_evaluator: async invoke sent for response_id=%s", response_id)
    except Exception as exc:
        logger.warning("_fire_evaluator: invoke failed — %s", exc)


# ── Endpoint handlers ─────────────────────────────────────────────────────────

def _handle_query(body: dict) -> dict:
    question = body.get("question", "").strip()
    if not question:
        return _respond(400, {"error": "Missing required field: 'question'."})

    response_id = str(uuid.uuid4())

    try:
        initial_state = {"response_id": response_id, "question": question}
        final_state   = app.invoke(initial_state)
        answer        = final_state.get("answer", "")
        node_path     = _node_path(final_state)

        # ── Deliver response immediately ──────────────────────────────────────
        response = _respond(200, {"answer": answer, "response_id": response_id})

        # ── Fire async evaluation (Layers 1, 2, 3) — does not block ──────────
        _fire_evaluator(
            response_id=response_id,
            question=question,
            answer=answer,
            documents=final_state.get("documents", []),
            node_path=node_path,
        )

        return response

    except Exception as exc:
        logger.exception("Graph error: %s", exc)
        return _respond(500, {"error": "Internal server error. Please try again."})


def _handle_feedback(body: dict) -> dict:
    """Layer 4 — store user feedback and auto-flag thumbs-down for human review."""
    response_id = body.get("response_id", "").strip()
    rating      = body.get("rating", "").strip()
    comment     = body.get("comment", "").strip()

    if not response_id:
        return _respond(400, {"error": "Missing required field: 'response_id'."})
    if rating not in ("thumbs_up", "thumbs_down"):
        return _respond(400, {"error": "Field 'rating' must be 'thumbs_up' or 'thumbs_down'."})

    storage.store_feedback(response_id=response_id, rating=rating, comment=comment)
    return _respond(200, {"status": "ok"})


# ── Lambda entry point ────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """Lambda handler for API Gateway HTTP API (payload format 2.0)."""
    logger.info("Event: %s", json.dumps(event))

    http   = event.get("requestContext", {}).get("http", {})
    method = http.get("method", "")
    path   = event.get("rawPath", "/query")

    if method == "GET":
        return _respond(200, {"status": "ok"})

    if method == "OPTIONS":
        return _respond(200, {})

    raw_body = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        import base64
        raw_body = base64.b64decode(raw_body).decode("utf-8")

    try:
        body = json.loads(raw_body or "{}")
    except json.JSONDecodeError:
        return _respond(400, {"error": "Request body must be valid JSON."})

    if method == "POST" and path == "/feedback":
        return _handle_feedback(body)

    if method == "POST" and path == "/query":
        return _handle_query(body)

    return _respond(404, {"error": "Not found."})
