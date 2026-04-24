"""
agent.py — Agentic loop using OpenAI tool/function calling.

The agent iteratively calls `search_knowledge_base` (1–3 times) until it
has gathered enough context, then synthesises a final answer.

Returns a tuple (answer, documents, tool_rounds_used) so the Lambda handler
can pass all retrieved context to the async evaluator.
"""

import json
import logging
import os
from typing import Any

from openai import OpenAI

from retriever import search

logger = logging.getLogger(__name__)

LLM_MODEL      = os.environ.get("LLM_MODEL",       "gpt-4o")
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "3"))

_openai = OpenAI()

# ── Tool schema ──────────────────────────────────────────────────────────────
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the knowledge base for information relevant to a query. "
                "Call this tool one or more times to gather context before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the knowledge base.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.
When answering questions:
1. Use the search_knowledge_base tool to retrieve relevant information.
2. You may call the tool multiple times with different queries to gather enough context.
3. After retrieving sufficient information, synthesise a clear, accurate answer.
4. If the knowledge base does not contain relevant information, say so honestly.
"""


def _handle_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    all_documents: list[str],
) -> str:
    """Execute a tool call and append retrieved chunks to all_documents."""
    if tool_name == "search_knowledge_base":
        chunks = search(arguments["query"])
        if not chunks:
            return "No relevant information found in the knowledge base."
        all_documents.extend(chunks)   # ← collect across all rounds
        return "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )
    return f"Unknown tool: {tool_name}"


def run(question: str) -> tuple[str, list[str], int]:
    """
    Run the agentic loop and return:
      answer           — the final answer string
      all_documents    — every chunk retrieved across all tool call rounds
      tool_rounds_used — number of rounds that involved a tool call
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    all_documents: list[str] = []
    tool_rounds_used: int    = 0

    for round_num in range(MAX_TOOL_ROUNDS + 1):
        logger.info("Agent round %d/%d", round_num, MAX_TOOL_ROUNDS)

        response = _openai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        # No tool calls → model produced a final answer
        if not message.tool_calls:
            return message.content or "", all_documents, tool_rounds_used

        tool_rounds_used += 1

        if round_num == MAX_TOOL_ROUNDS:
            logger.info("Max tool rounds reached — requesting final answer")
            messages.append({
                "role":    "user",
                "content": (
                    "You have gathered enough context. "
                    "Please provide your final answer now."
                ),
            })
            final = _openai.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
            )
            return final.choices[0].message.content or "", all_documents, tool_rounds_used

        # Execute each tool call and feed results back
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            logger.info("Tool call: %s(%s)", fn_name, fn_args)

            tool_result = _handle_tool_call(fn_name, fn_args, all_documents)

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      tool_result,
            })

    return "Unable to generate an answer.", all_documents, tool_rounds_used
