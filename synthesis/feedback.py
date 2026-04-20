"""Synthesis module: feedback agent — adjusts scoring weights and tracks prompt quality.

This module also provides the conditional edge function that decides whether the
pipeline should loop back through the scoring stage or terminate.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from llm.ollama_client import OllamaClient
from synthesis.prompts import FEEDBACK_TEMPLATE
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)

# Maximum number of feedback-loop iterations before unconditional termination.
_MAX_ITERATIONS: int = 2

# Minimum prompt-quality score (0–1) required to exit the feedback loop early.
_QUALITY_THRESHOLD: float = 0.8

# Maximum absolute adjustment accepted from LLM feedback.
_MAX_WEIGHT_ADJUSTMENT: float = 1.0


def _coerce_weight_adjustments(raw_adjustments: Any) -> dict[str, float]:
    """Normalize weight adjustments to string keys and bounded float values."""
    if not isinstance(raw_adjustments, dict):
        return {}

    adjustments: dict[str, float] = {}
    for key, value in raw_adjustments.items():
        try:
            delta = float(value)
        except (ValueError, TypeError):
            continue

        delta = max(-_MAX_WEIGHT_ADJUSTMENT, min(_MAX_WEIGHT_ADJUSTMENT, delta))
        adjustments[str(key)] = delta

    return adjustments


def _parse_feedback(raw: str) -> dict[str, Any]:
    """Extract feedback fields from LLM JSON, with safe defaults."""
    try:
        data = json.loads(raw.strip())
        return {
            "weight_adjustments": _coerce_weight_adjustments(data.get("weight_adjustments", {})),
            "prompt_quality": float(data.get("prompt_quality", 0.7)),
            "prompt_improvement": str(data.get("prompt_improvement", "")),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"weight_adjustments": {}, "prompt_quality": 0.7, "prompt_improvement": ""}


def apply_feedback(state: ResearchState) -> dict[str, Any]:
    """Evaluate the digest and compute scoring/prompt adjustments for the next iteration.

    Node inputs:  ``digest``, ``iteration`` (optional)
    Node outputs: ``feedback``, ``iteration``
    """
    digest = state.get("digest", "")
    iteration = state.get("iteration", 0) + 1

    if not digest:
        return {"feedback": {}, "iteration": iteration}

    prompt = FEEDBACK_TEMPLATE.format(digest=digest)
    llm = OllamaClient.get_llm()
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        feedback = _parse_feedback(raw)
    except Exception:
        logger.warning("Feedback generation failed")
        feedback = {"weight_adjustments": {}, "prompt_quality": 0.7, "prompt_improvement": ""}

    logger.debug(
        "Feedback iteration %d: quality=%.2f",
        iteration,
        feedback.get("prompt_quality", 0.0),
    )
    return {"feedback": feedback, "iteration": iteration}


def should_continue(state: ResearchState) -> str:
    """Conditional edge: decide whether to loop back to scoring or terminate.

    Returns:
        ``"score_chunks"`` — re-score with updated weights for another iteration.
        ``"end"``          — pipeline is complete.
    """
    iteration = state.get("iteration", 0)
    prompt_quality = state.get("feedback", {}).get("prompt_quality", 1.0)

    if iteration >= _MAX_ITERATIONS or prompt_quality >= _QUALITY_THRESHOLD:
        return "end"
    return "score_chunks"
