"""
nodes.py — LangGraph state definition and node functions.

State flows through 4 nodes:
  1. classify_query  — decides which tool(s) to use
  2. run_tools       — calls the right tool (graph, vector, vlm, or both)
  3. generate        — builds the final answer from collected context
  4. no_answer       — fallback when no context was found

Interview explanation of the flow:
  "Every query is first classified into a type, then routed to the right
   retrieval tool. The results become the context for the final LLM answer."
"""

import logging
import os
from typing import TypedDict

from openai import OpenAI

import tools

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_openai = OpenAI()

# ── Typed state ───────────────────────────────────────────────────────────────

class GraphRAGState(TypedDict, total=False):
    response_id:   str             # UUID set by lambda_function
    question:      str             # text query (always present)
    image_base64:  str | None      # base64 image for VLM queries
    media_type:    str | None      # e.g. "image/jpeg"
    query_type:    str             # "relational" | "semantic" | "visual" | "hybrid"
    documents:     list[str]       # context chunks assembled by run_tools
    answer:        str             # final answer from generate or no_answer


# ── Node 1: Classify Query ────────────────────────────────────────────────────

def classify_query(state: GraphRAGState) -> dict:
    """
    Decide which retrieval path to take:
      relational — needs graph traversal (who, which, how many, relationships)
      semantic   — needs document search (what is, explain, what does X say)
      visual     — has an image attached
      hybrid     — needs both graph + vector (complex questions)

    If an image is present, always classify as "visual".
    """
    # Image present → always visual
    if state.get("image_base64"):
        logger.info("classify_query: image detected → visual")
        return {"query_type": "visual"}

    question = state["question"]

    prompt = f"""You are a query router for a dental healthcare interoperability system.

Classify the question into exactly one category:
  relational — asks about entities and their relationships
               (e.g. patients, referrals, providers, prior auths, who/which/how many)
  semantic   — asks about policies, standards, or documentation
               (e.g. FHIR, CARIN, TEFCA, prior auth rules, what does X mean)
  hybrid     — needs BOTH relationships AND documentation to answer fully

Question: {question}

Return ONLY one word: relational, semantic, or hybrid."""

    response = _openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip().lower()

    # Normalize to valid type
    if raw not in ("relational", "semantic", "hybrid"):
        raw = "semantic"

    logger.info("classify_query: question=%r → type=%s", question[:60], raw)
    return {"query_type": raw}


# ── Node 2: Run Tools ─────────────────────────────────────────────────────────

def run_tools(state: GraphRAGState) -> dict:
    """
    Dispatch to the right tool(s) based on query_type.

      relational → graph_search
      semantic   → vector_search
      visual     → vlm_analyze
      hybrid     → graph_search + vector_search (results combined)

    All results are merged into `documents` — a flat list of context strings.
    """
    query_type = state.get("query_type", "semantic")
    question   = state["question"]
    documents: list[str] = []

    if query_type == "relational":
        logger.info("run_tools: calling graph_search")
        documents = tools.graph_search(question)

    elif query_type == "semantic":
        logger.info("run_tools: calling vector_search")
        documents = tools.vector_search(question)

    elif query_type == "visual":
        logger.info("run_tools: calling vlm_analyze")
        image_b64  = state.get("image_base64", "")
        media_type = state.get("media_type", "image/jpeg")
        documents  = tools.vlm_analyze(image_b64, media_type)

    elif query_type == "hybrid":
        logger.info("run_tools: calling graph_search + vector_search")
        graph_results  = tools.graph_search(question)
        vector_results = tools.vector_search(question)
        documents      = graph_results + vector_results

    logger.info("run_tools: collected %d context chunks", len(documents))
    return {"documents": documents}


# ── Node 3: Generate ──────────────────────────────────────────────────────────

def generate(state: GraphRAGState) -> dict:
    """
    Assemble context from `documents` and generate a final answer using the LLM.
    Includes the query_type so the LLM knows what kind of context it received.
    """
    context    = "\n\n---\n\n".join(state["documents"])
    question   = state["question"]
    query_type = state.get("query_type", "")

    system_prompt = (
        "You are Conduit Assistant — an expert in dental healthcare interoperability, "
        "FHIR R4, dental data exchange, and Conduit's platform. "
        "Answer the question using ONLY the provided context. "
        "Be concise and specific. If the context contains structured data (from graph or VLM), "
        "present it in a clear, organized format."
    )

    user_prompt = (
        f"Context type: {query_type}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = _openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content
    logger.info("generate: answer_length=%d chars", len(answer))
    return {"answer": answer}


# ── Node 4: No Answer ─────────────────────────────────────────────────────────

def no_answer(state: GraphRAGState) -> dict:
    """Fallback when no context was retrieved from any tool."""
    logger.info("no_answer: no documents found for question=%r", state["question"][:60])
    return {
        "answer": (
            "I could not find relevant information in the knowledge base or graph "
            "to answer this question. Please try rephrasing, or check that the "
            "relevant data has been ingested."
        )
    }
