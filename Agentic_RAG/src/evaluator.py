"""
evaluator.py — Hybrid evaluation framework (Layers 1 & 2) + threshold gate.

Layer 1 — RAGAS-style metrics (no ground truth required):
  faithfulness       — are answer claims supported by retrieved context?
  answer_relevance   — does the answer address the question?
  context_precision  — are the retrieved chunks relevant to the question?

Layer 2 — LLM-as-a-Judge (stronger model than the generator):
  correctness, completeness, groundedness, clarity  (each scored 1–5)

Threshold gate:
  passes_threshold = True only when ALL metrics clear their configured threshold.
  All thresholds are configurable via environment variables.
"""

import json
import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

LLM_MODEL   = os.environ.get("LLM_MODEL",   "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL",  "gpt-4o")   # stronger model — avoids self-serving bias

THRESHOLD_FAITHFULNESS      = float(os.environ.get("THRESHOLD_FAITHFULNESS",      "0.7"))
THRESHOLD_ANSWER_RELEVANCE  = float(os.environ.get("THRESHOLD_ANSWER_RELEVANCE",  "0.6"))
THRESHOLD_CONTEXT_PRECISION = float(os.environ.get("THRESHOLD_CONTEXT_PRECISION", "0.5"))
THRESHOLD_JUDGE_SCORE       = float(os.environ.get("THRESHOLD_JUDGE_SCORE",       "3.0"))

_openai = OpenAI()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chat(model: str, prompt: str) -> str:
    response = _openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _parse_json(text: str, default):
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("JSON parse failed for evaluator output: %r", text[:200])
        return default


# ── Layer 1: RAGAS-style metrics ─────────────────────────────────────────────

def _faithfulness(context: str, answer: str) -> float:
    """
    Extract all factual claims from the answer and verify each against
    the retrieved context.  Score = supported_claims / total_claims.
    """
    prompt = f"""Extract every factual claim made in the Answer and decide whether
each is directly supported by the Context.

Context:
{context}

Answer:
{answer}

Return ONLY a JSON array like:
[{{"claim": "...", "supported": true}}, ...]"""

    claims = _parse_json(_chat(LLM_MODEL, prompt), default=[])
    if not claims:
        return 1.0  # no claims → vacuously faithful
    supported = sum(1 for c in claims if c.get("supported", False))
    score = round(supported / len(claims), 3)
    logger.info("faithfulness: %d/%d claims supported → %.3f", supported, len(claims), score)
    return score


def _answer_relevance(question: str, answer: str) -> float:
    """
    Ask the LLM how well the answer addresses the question.
    Score = 0.0 – 1.0.
    """
    prompt = f"""Rate how well the Answer addresses the Question on a scale of 0.0 to 1.0.
0.0 = completely irrelevant, 1.0 = perfectly and fully addresses the question.

Question: {question}
Answer:   {answer}

Return ONLY a float between 0.0 and 1.0, no other text."""

    raw = _chat(LLM_MODEL, prompt).strip()
    try:
        score = round(max(0.0, min(1.0, float(raw))), 3)
    except ValueError:
        logger.warning("answer_relevance parse failed: %r", raw)
        score = 0.5
    logger.info("answer_relevance: %.3f", score)
    return score


def _context_precision(question: str, documents: list[str]) -> float:
    """
    Check how many retrieved chunks are actually useful for answering
    the question.  Score = relevant_chunks / total_chunks.
    """
    if not documents:
        return 0.0

    chunks_text = "\n\n".join(
        f"[Chunk {i}]: {chunk}" for i, chunk in enumerate(documents)
    )
    prompt = f"""For each chunk below, decide whether it contains information
useful for answering the Question.

Question: {question}

Chunks:
{chunks_text}

Return ONLY a JSON array like:
[{{"chunk_index": 0, "relevant": true}}, ...]"""

    results = _parse_json(_chat(LLM_MODEL, prompt), default=[])
    if not results:
        return 0.5
    relevant = sum(1 for r in results if r.get("relevant", False))
    score = round(relevant / len(documents), 3)
    logger.info("context_precision: %d/%d chunks relevant → %.3f", relevant, len(documents), score)
    return score


# ── Layer 2: LLM-as-a-Judge ──────────────────────────────────────────────────

def _llm_judge(question: str, context: str, answer: str) -> dict[str, float]:
    """
    Score the answer on four quality dimensions using a stronger judge model.
    Returns dict with keys: correctness, completeness, groundedness, clarity (1–5).
    """
    prompt = f"""You are an expert answer quality evaluator.
Score the Answer on each dimension from 1 (worst) to 5 (best).

Question: {question}

Context (retrieved documents):
{context}

Answer:
{answer}

Dimensions:
  correctness  — is the answer factually accurate based on the context?
  completeness — does it fully address all aspects of the question?
  groundedness — is every claim supported by the context (no hallucination)?
  clarity      — is it well-written and easy to understand?

Return ONLY a JSON object like:
{{"correctness": 4, "completeness": 3, "groundedness": 5, "clarity": 4}}"""

    raw  = _chat(JUDGE_MODEL, prompt)
    data = _parse_json(raw, default={})

    scores: dict[str, float] = {}
    for dim in ("correctness", "completeness", "groundedness", "clarity"):
        try:
            scores[dim] = float(max(1, min(5, data.get(dim, 3))))
        except (TypeError, ValueError):
            scores[dim] = 3.0

    logger.info("llm_judge: %s", scores)
    return scores


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    # Layer 1
    faithfulness:       float
    answer_relevance:   float
    context_precision:  float
    # Layer 2
    judge_correctness:  float
    judge_completeness: float
    judge_groundedness: float
    judge_clarity:      float
    judge_score:        float          # mean of the four judge dimensions
    # Gate
    passes_threshold:   bool
    failure_reasons:    list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "faithfulness":       self.faithfulness,
            "answer_relevance":   self.answer_relevance,
            "context_precision":  self.context_precision,
            "judge_correctness":  self.judge_correctness,
            "judge_completeness": self.judge_completeness,
            "judge_groundedness": self.judge_groundedness,
            "judge_clarity":      self.judge_clarity,
            "judge_score":        self.judge_score,
            "passes_threshold":   self.passes_threshold,
            "failure_reasons":    self.failure_reasons,
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_evaluation(
    question: str,
    documents: list[str],
    answer: str,
) -> EvaluationResult:
    """
    Run all evaluation layers and apply the threshold gate.
    Never raises — returns a conservative result on unexpected errors.
    """
    try:
        context = "\n\n---\n\n".join(documents) if documents else ""

        # Layer 1
        faith = _faithfulness(context, answer)
        rel   = _answer_relevance(question, answer)
        prec  = _context_precision(question, documents)

        # Layer 2
        judge   = _llm_judge(question, context, answer)
        j_score = round(sum(judge.values()) / len(judge), 3)

        # Threshold gate
        failures: list[str] = []
        if faith < THRESHOLD_FAITHFULNESS:
            failures.append(f"faithfulness={faith:.2f} < {THRESHOLD_FAITHFULNESS}")
        if rel < THRESHOLD_ANSWER_RELEVANCE:
            failures.append(f"answer_relevance={rel:.2f} < {THRESHOLD_ANSWER_RELEVANCE}")
        if prec < THRESHOLD_CONTEXT_PRECISION:
            failures.append(f"context_precision={prec:.2f} < {THRESHOLD_CONTEXT_PRECISION}")
        if j_score < THRESHOLD_JUDGE_SCORE:
            failures.append(f"judge_score={j_score:.2f} < {THRESHOLD_JUDGE_SCORE}")

        passes = len(failures) == 0
        logger.info("evaluation gate: passes=%s failures=%s", passes, failures)

        return EvaluationResult(
            faithfulness=faith,
            answer_relevance=rel,
            context_precision=prec,
            judge_correctness=judge["correctness"],
            judge_completeness=judge["completeness"],
            judge_groundedness=judge["groundedness"],
            judge_clarity=judge["clarity"],
            judge_score=j_score,
            passes_threshold=passes,
            failure_reasons=failures,
        )

    except Exception as exc:
        # Fail open — do not block the user if the evaluator itself errors
        logger.exception("run_evaluation: unexpected error — %s", exc)
        return EvaluationResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_precision=0.0,
            judge_correctness=3.0,
            judge_completeness=3.0,
            judge_groundedness=3.0,
            judge_clarity=3.0,
            judge_score=3.0,
            passes_threshold=True,   # fail open
            failure_reasons=["evaluator_error"],
        )
