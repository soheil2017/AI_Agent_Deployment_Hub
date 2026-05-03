"""
evaluator_handler.py — Async evaluation + Langfuse score logging.

Called as a FastAPI BackgroundTask after every /query response is delivered.
Evaluation never adds latency to the user — it runs after the answer is sent.

What this does:
  Layers 1 & 2 — runs RAGAS + LLM-judge metrics via evaluator.run_evaluation()
  Layer 3       — logs all scores to Langfuse on the same trace as the query
                  (so you see scores alongside the node spans in Langfuse UI)
  Layer 4       — log_user_feedback() logs thumbs-up/down as a Langfuse score

Langfuse score names:
  faithfulness, answer_relevance, context_precision  (Layer 1 — 0.0 to 1.0)
  judge_correctness, judge_completeness,
  judge_groundedness, judge_clarity, judge_score      (Layer 2 — 1.0 to 5.0)
  user_feedback                                       (Layer 4 — 1=up, 0=down)
"""

import logging
import os

from langfuse import Langfuse

from evaluator import run_evaluation

logger = logging.getLogger(__name__)

try:
    _langfuse = Langfuse()
    _langfuse_enabled = True
except Exception as e:
    _langfuse = None
    _langfuse_enabled = False
    logger.warning("Langfuse init failed: %s", e)


def run(
    trace_id:   str,
    question:   str,
    answer:     str,
    documents:  list[str],
    query_type: str,
) -> None:
    """
    Run evaluation and push all scores to Langfuse.

    This function is called by FastAPI's BackgroundTasks — it runs after
    the HTTP response has already been sent to the client.
    """
    logger.info("evaluator_handler.run: start trace_id=%s query_type=%s", trace_id, query_type)

    # Layers 1 & 2 — compute all metrics
    result = run_evaluation(question=question, documents=documents, answer=answer)

    # Layer 3 — log every metric as a named score on the Langfuse trace
    scores = {
        # Layer 1: RAGAS-style (0.0 – 1.0)
        "faithfulness":       result.faithfulness,
        "answer_relevance":   result.answer_relevance,
        "context_precision":  result.context_precision,
        # Layer 2: LLM-judge (1.0 – 5.0)
        "judge_correctness":  result.judge_correctness,
        "judge_completeness": result.judge_completeness,
        "judge_groundedness": result.judge_groundedness,
        "judge_clarity":      result.judge_clarity,
        "judge_score":        result.judge_score,
    }

    if _langfuse_enabled and _langfuse:
        try:
            for name, value in scores.items():
                _langfuse.create_score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=f"query_type={query_type}",
                )
            passes = result.passes_threshold
            _langfuse.create_score(
                trace_id=trace_id,
                name="passes_threshold",
                value=1.0 if passes else 0.0,
                comment="; ".join(result.failure_reasons) if result.failure_reasons else "all thresholds met",
            )
            _langfuse.flush()
            logger.info("Langfuse: scores flushed for trace_id=%s", trace_id)
        except Exception as e:
            logger.warning("Langfuse score logging failed: %s", e)

    passes = result.passes_threshold

    logger.info(
        "evaluator_handler.run: done trace_id=%s passes=%s judge_score=%.2f",
        trace_id, passes, result.judge_score,
    )


def log_user_feedback(trace_id: str, rating: str, comment: str = "") -> None:
    """
    Layer 4 — store user thumbs-up/down as a Langfuse score.
    Value: 1.0 = thumbs_up, 0.0 = thumbs_down.
    """
    value = 1.0 if rating == "thumbs_up" else 0.0
    if _langfuse_enabled and _langfuse:
        try:
            _langfuse.create_score(
                trace_id=trace_id,
                name="user_feedback",
                value=value,
                comment=comment or rating,
            )
            _langfuse.flush()
        except Exception as e:
            logger.warning("Langfuse feedback logging failed: %s", e)
    logger.info("log_user_feedback: trace_id=%s rating=%s", trace_id, rating)
