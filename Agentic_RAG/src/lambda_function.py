"""
lambda_function.py — AWS Lambda entry point / API Gateway HTTP API handler.

Expected request body (JSON):
    { "question": "What is ...?" }

Response body (JSON):
    { "answer": "..." }
    or
    { "error": "..." }  with a non-200 status code
"""

import json
import logging
import os

# Load .env when running locally (no-op in Lambda)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import run

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
}


def _respond(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def handler(event: dict, context) -> dict:
    """Lambda handler for API Gateway HTTP API (payload format 2.0)."""
    logger.info("Event: %s", json.dumps(event))

    # Handle CORS pre-flight
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return _respond(200, {})

    # Parse request body
    raw_body = event.get("body", "{}")
    if event.get("isBase64Encoded"):
        import base64
        raw_body = base64.b64decode(raw_body).decode("utf-8")

    try:
        body = json.loads(raw_body or "{}")
    except json.JSONDecodeError:
        return _respond(400, {"error": "Request body must be valid JSON."})

    question = body.get("question", "").strip()
    if not question:
        return _respond(400, {"error": "Missing required field: 'question'."})

    try:
        answer = run(question)
        return _respond(200, {"answer": answer})
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        return _respond(500, {"error": "Internal server error. Please try again."})
