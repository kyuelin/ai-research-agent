"""Synthesis module: scoring agent — ranks chunks by novelty, practicality, adoption, relevance."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm.ollama_client import OllamaClient
from synthesis.prompts import SCORE_CHUNKS_TEMPLATE
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "novelty": 0.25,
    "practicality": 0.25,
    "adoption": 0.25,
    "relevance": 0.25,
}


def _parse_scores(raw: str) -> dict[str, float]:
    """Extract numeric dimension scores from LLM JSON, falling back to 0.5 defaults."""
    try:
        data = json.loads(raw.strip())
        return {
            "novelty": float(data.get("novelty", 0.5)),
            "practicality": float(data.get("practicality", 0.5)),
            "adoption": float(data.get("adoption", 0.5)),
            "relevance": float(data.get("relevance", 0.5)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"novelty": 0.5, "practicality": 0.5, "adoption": 0.5, "relevance": 0.5}


def _weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return sum(scores[k] * weights.get(k, 0.25) for k in scores)


def score_chunks(state: ResearchState) -> dict[str, Any]:
    """Score each normalised chunk and sort by composite weighted score.

    Weight adjustments from the feedback loop are applied on top of the
    default equal weights so that subsequent iterations can reprioritise
    dimensions based on digest quality.

    Node inputs:  ``query``, ``normalized_chunks``, ``feedback`` (optional)
    Node outputs: ``scores``
    """
    query = state.get("query", "")
    chunks = state.get("normalized_chunks", [])
    feedback = state.get("feedback", {})

    weight_adjustments: dict[str, float] = feedback.get("weight_adjustments", {})
    weights = {
        k: max(0.0, min(1.0, _DEFAULT_WEIGHTS[k] + weight_adjustments.get(k, 0.0)))
        for k in _DEFAULT_WEIGHTS
    }

    llm = OllamaClient.get_llm()
    scored: list[dict[str, Any]] = []

    for chunk in chunks:
        prompt = SCORE_CHUNKS_TEMPLATE.format(query=query, chunk=chunk["text"])
        try:
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            dim_scores = _parse_scores(raw)
        except Exception:
            logger.warning("Scoring failed for chunk %d; using defaults", chunk["index"])
            dim_scores = {"novelty": 0.5, "practicality": 0.5, "adoption": 0.5, "relevance": 0.5}

        composite = _weighted_score(dim_scores, weights)
        scored.append({**chunk, "scores": dim_scores, "composite_score": composite})

    scored.sort(key=lambda c: c["composite_score"], reverse=True)
    logger.debug("Scored and ranked %d chunks", len(scored))
    return {"scores": scored}
