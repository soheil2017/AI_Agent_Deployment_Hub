"""
tools.py — Three retrieval tools used by the LangGraph agent.

  graph_search(question)              → query Neo4j using LLM-generated Cypher
  vector_search(question)             → semantic search in ChromaDB
  vlm_analyze(image_base64, prompt)   → Claude Vision analyzes a dental image

Each tool returns a list of plain text strings (the "context chunks") that
the generate node assembles into a final answer.

Interview explanation:
  - graph_search  → "who relates to whom" (relational questions)
  - vector_search → "what does this document say" (semantic questions)
  - vlm_analyze   → "what's in this image" (visual questions)
"""

import logging
import os

import anthropic
from openai import OpenAI

from storage import get_neo4j, get_chroma

logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
VLM_MODEL = os.environ.get("VLM_MODEL", "claude-opus-4-6")   # Claude Vision

_openai    = OpenAI()
_anthropic = anthropic.Anthropic()


# ── Tool 1: Graph Search ──────────────────────────────────────────────────────

# Neo4j schema hint injected into every Cypher-generation prompt.
# This tells the LLM exactly what nodes and relationships exist.
_NEO4J_SCHEMA = """
Node labels and key properties:
  Patient      {id, name, dob, member_id}
  Provider     {npi, name, specialty, type}   -- type: "dental" or "medical"
  Organization {id, name, type}               -- type: DSO / health_system / payer
  Condition    {code, system, description}    -- system: ICD-10 or SNOMED
  Procedure    {code, system, description}    -- system: CDT or CPT
  Referral     {id, type, status, created_date, reason}
               -- type: medical_to_dental / dental_to_medical / dental_to_dental
               -- status: pending / completed
  PriorAuth    {id, status, submitted_date, decision_date, procedure_code}
               -- status: pending / approved / denied

Relationships:
  (Patient)-[:HAS_CONDITION]->(Condition)
  (Patient)-[:HAD_PROCEDURE]->(Procedure)
  (Patient)-[:HAS_REFERRAL]->(Referral)
  (Provider)-[:SENT_REFERRAL]->(Referral)
  (Referral)-[:ASSIGNED_TO]->(Provider)
  (Referral)-[:REQUIRES_AUTH]->(PriorAuth)
  (PriorAuth)-[:SUBMITTED_TO]->(Organization)
  (Provider)-[:WORKS_AT]->(Organization)
"""


def graph_search(question: str) -> list[str]:
    """
    Step 1: Ask the LLM to write a Cypher query for the question.
    Step 2: Run that Cypher on Neo4j.
    Step 3: Return results as readable text strings.

    This handles multi-hop relational questions like:
    "Which patients have a pending referral where prior auth is also pending?"
    """
    # Ask LLM to generate Cypher
    prompt = f"""You are a Neo4j expert. Given the schema below, write a Cypher READ query
to answer the question. Return ONLY the Cypher query — no explanation, no markdown fences.

Schema:
{_NEO4J_SCHEMA}

Question: {question}"""

    response = _openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    cypher = response.choices[0].message.content.strip()

    # Strip markdown fences if the LLM added them anyway
    if cypher.startswith("```"):
        cypher = cypher.split("```")[1].lstrip("cypher").strip()

    logger.info("graph_search: generated Cypher: %s", cypher)

    # Run Cypher on Neo4j
    try:
        records = get_neo4j().query(cypher)
    except Exception as exc:
        logger.warning("graph_search: Cypher execution failed — %s", exc)
        return []

    if not records:
        return []

    # Convert each record dict into a readable string
    results = []
    for record in records:
        line = " | ".join(f"{k}: {v}" for k, v in record.items() if v is not None)
        results.append(line)

    logger.info("graph_search: returned %d records", len(results))
    return results


# ── Tool 2: Vector Search ─────────────────────────────────────────────────────

def vector_search(question: str, top_k: int = 4) -> list[str]:
    """
    Semantic search over policy and spec documents stored in ChromaDB.

    Handles questions like:
    "What does CARIN Blue Button require for patient access?"
    "What is the TEFCA exchange purpose for referrals?"
    """
    hits = get_chroma().search(question, top_k=top_k)
    results = [hit["text"] for hit in hits]
    logger.info("vector_search: returned %d chunks", len(results))
    return results


# ── Tool 3: VLM Analyze ───────────────────────────────────────────────────────

def vlm_analyze(image_base64: str, media_type: str = "image/jpeg") -> list[str]:
    """
    Send a dental image to Claude Vision and return structured findings.

    Supported image types:
      - Dental X-rays    → detect caries, bone loss, implants
      - Insurance cards  → extract payer name, member ID, group number
      - EOB documents    → parse claim lines, procedure codes, amounts
      - Faxed referrals  → extract provider, patient, diagnosis details

    Returns a list with one string: the VLM's structured analysis.
    """
    prompt = (
        "You are a clinical document analyst specializing in dental healthcare. "
        "Analyze this image and extract structured information. "
        "If it is a dental X-ray: identify tooth numbers, caries, bone loss, restorations, implants. "
        "If it is an insurance card: extract payer name, member ID, group number, plan type. "
        "If it is an EOB: extract procedure codes, service dates, amounts billed, amounts paid, denial codes. "
        "If it is a referral letter: extract referring provider, receiving provider, patient name, diagnosis, reason for referral. "
        "Format your response as clear labeled fields."
    )

    response = _anthropic.messages.create(
        model=VLM_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_base64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    analysis = response.content[0].text
    logger.info("vlm_analyze: extracted %d chars of findings", len(analysis))
    return [f"[VLM Analysis]\n{analysis}"]
