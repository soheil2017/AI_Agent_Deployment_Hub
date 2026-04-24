"""
nodes.py — RAG graph state definition and node functions.

Nodes:
  retrieve        — embed the question, fetch top-k FAISS chunks
  grade_documents — decide whether retrieved chunks are relevant (yes/no)
  generate        — build a prompt from the chunks, call gpt-4o-mini
  no_answer       — return a fallback when documents are not relevant

Evaluation (Layers 1–3) runs asynchronously in a separate Lambda after the
response is delivered — see evaluator_handler.py and evaluator.py.
"""

import logging
import os
from typing import TypedDict

from openai import OpenAI

import retriever

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_openai = OpenAI()


# ── Typed state ──────────────────────────────────────────────────────────────

class RAGState(TypedDict, total=False):
    response_id: str         # UUID set by lambda_function, forwarded to evaluator
    question:    str         # input (always present)
    documents:   list[str]   # populated by retrieve
    relevant:    bool        # populated by grade_documents
    answer:      str         # populated by generate or no_answer


# ── Node functions ────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> dict:
    """Fetch the most relevant chunks for the question."""
    logger.info("retrieve: question=%r", state["question"])
    chunks = retriever.search(state["question"])
    logger.info("retrieve: found %d chunks", len(chunks))
    return {"documents": chunks}


def grade_documents(state: RAGState) -> dict:
    """Ask the LLM whether the retrieved chunks actually answer the question."""
    question  = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.info("grade_documents: no documents retrieved — marking irrelevant")
        return {"relevant": False}

    context  = "\n\n---\n\n".join(documents)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a relevance grader. Given a question and a set of retrieved "
                "documents, answer with a single word — 'yes' if the documents contain "
                "enough information to answer the question, or 'no' if they do not."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nDocuments:\n{context}",
        },
    ]

    response = _openai.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0,
    )
    verdict  = response.choices[0].message.content.strip().lower()
    relevant = verdict.startswith("yes")
    logger.info("grade_documents: verdict=%r relevant=%s", verdict, relevant)
    return {"relevant": relevant}


def generate(state: RAGState) -> dict:
    """Generate an answer from the retrieved chunks."""
    context = "\n\n---\n\n".join(state["documents"])
    logger.info("generate: context_length=%d chars", len(context))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using ONLY "
                "the provided context."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {state['question']}",
        },
    ]

    response = _openai.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0,
    )
    answer = response.choices[0].message.content
    logger.info("generate: answer_length=%d chars", len(answer))
    return {"answer": answer}


def no_answer(state: RAGState) -> dict:
    """Return a fallback answer when retrieved documents are not relevant."""
    logger.info("no_answer: documents were not relevant to the question")
    return {
        "answer": (
            "I don't have relevant information in my knowledge base "
            "to answer this question."
        )
    }
