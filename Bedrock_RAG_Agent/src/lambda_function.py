"""
lambda_function.py — AWS Lambda entry point for the Bedrock RAG Agent.

Receives an API Gateway HTTP API event, invokes the Bedrock Agent (which
internally queries the Knowledge Base), and returns the final answer.

Expected request body (JSON):
    { "question": "What is ...?", "session_id": "<optional-uuid>" }

Response body (JSON):
    { "answer": "...", "session_id": "..." }
    or
    { "error": "..." }  with a non-200 status code
"""

import json
import logging
import os
import uuid

import boto3

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

AGENT_ID = os.environ["BEDROCK_AGENT_ID"]
AGENT_ALIAS_ID = os.environ["BEDROCK_AGENT_ALIAS_ID"]

_runtime = boto3.client("bedrock-agent-runtime")

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


def _invoke_agent(question: str, session_id: str) -> str:
    """Invoke the Bedrock Agent and collect the streamed response."""
    response = _runtime.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=question,
    )

    answer_parts: list[str] = []
    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            answer_parts.append(chunk["bytes"].decode("utf-8"))

    return "".join(answer_parts)


def handler(event: dict, context) -> dict:
    """Lambda handler — API Gateway HTTP API payload format 2.0."""
    logger.info("Event: %s", json.dumps(event))

    # CORS pre-flight
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return _respond(200, {})

    # Health check
    raw_path = event.get("rawPath", "")
    if raw_path == "/health":
        return _respond(200, {"status": "ok"})

    # Parse body
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

    session_id = body.get("session_id") or str(uuid.uuid4())

    try:
        answer = _invoke_agent(question, session_id)
        return _respond(200, {"answer": answer, "session_id": session_id})
    except Exception as exc:
        logger.exception("Agent invocation error: %s", exc)
        return _respond(500, {"error": "Internal server error. Please try again."})
