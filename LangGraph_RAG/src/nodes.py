"""
nodes.py — RAG graph state definition and node functions.

Nodes:
  retrieve  — embed the question, fetch top-k FAISS chunks
  generate  — build a prompt from the chunks, call gpt-4o-mini
"""

import os
import logging
from typing import TypedDict

from openai import OpenAI
import retriever

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_openai = OpenAI()


# ── Typed state ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    documents: list[str]
    answer: str


# ── Node functions ────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> dict:
    """Fetch the most relevant chunks for the question."""
    logger.info("retrieve: question=%r", state["question"])
    chunks = retriever.search(state["question"])
    logger.info("retrieve: found %d chunks", len(chunks))
    return {"documents": chunks}


def generate(state: RAGState) -> dict:
    """Generate an answer from the retrieved chunks."""
    context = "\n\n---\n\n".join(state["documents"]) if state["documents"] else "No context available."
    logger.info("generate: context_length=%d chars", len(context))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the provided context. If the context does not contain enough information, "
                "say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {state['question']}",
        },
    ]

    response = _openai.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content
    logger.info("generate: answer_length=%d chars", len(answer))
    return {"answer": answer}
