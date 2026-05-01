"""
lambda_function.py — AWS Lambda entry point / API Gateway HTTP API handler.

Endpoints:
  POST /query     — run the Graph RAG agent, return answer immediately,
                    then fire async evaluation (no blocking on metrics)
  POST /feedback  — store user thumbs-up/thumbs-down (Layer 4)
  GET  /health    — health check

Query request (text):
  { "question": "Which patients have a pending prior auth?" }

Query request (with image for VLM):
  { "question": "Analyze this X-ray", "image_base64": "<base64>", "media_type": "image/jpeg" }

Query response:
  { "answer": "...", "response_id": "<uuid>", "query_type": "relational" }

Feedback request:
  { "response_id": "<uuid>", "rating": "thumbs_up" | "thumbs_down", "comment": "" }
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
    "Content-Type":                "application/json",
    "Access-Control-Allow-Origin": "*",
}

_lambda_client = boto3.client("lambda")


def _respond(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers":    CORS_HEADERS,
        "body":       json.dumps(body),
    }


def _fire_evaluator(
    response_id: str,
    question:    str,
    answer:      str,
    documents:   list[str],
    query_type:  str,
) -> None:
    """Fire-and-forget async evaluation Lambda invocation."""
    if not EVALUATOR_FUNCTION_NAME:
        logger.warning("_fire_evaluator: EVALUATOR_FUNCTION_NAME not set — skipping")
        return
    try:
        _lambda_client.invoke(
            FunctionName=EVALUATOR_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps({
                "response_id": response_id,
                "question":    question,
                "answer":      answer,
                "documents":   documents,
                "query_type":  query_type,
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

    response_id  = str(uuid.uuid4())
    image_base64 = body.get("image_base64")
    media_type   = body.get("media_type", "image/jpeg")

    try:
        initial_state: dict = {"response_id": response_id, "question": question}
        if image_base64:
            initial_state["image_base64"] = image_base64
            initial_state["media_type"]   = media_type

        final_state = app.invoke(initial_state)
        answer      = final_state.get("answer", "")
        query_type  = final_state.get("query_type", "")

        # Deliver response immediately
        response = _respond(200, {
            "answer":      answer,
            "response_id": response_id,
            "query_type":  query_type,
        })

        # Fire async evaluation (Layers 1, 2, 3) — does not block
        _fire_evaluator(
            response_id=response_id,
            question=question,
            answer=answer,
            documents=final_state.get("documents", []),
            query_type=query_type,
        )

        return response

    except Exception as exc:
        logger.exception("Graph error: %s", exc)
        return _respond(500, {"error": "Internal server error. Please try again."})


def _handle_feedback(body: dict) -> dict:
    """Layer 4 — store user feedback."""
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
