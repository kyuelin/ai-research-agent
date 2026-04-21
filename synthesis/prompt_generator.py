"""Synthesis module: prompt generator agent — produces code-generation prompts."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm.ollama_client import OllamaClient
from synthesis.prompts import GENERATE_PROMPTS_TEMPLATE
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def _parse_prompts(raw: str) -> list[str]:
    """Extract a list of prompt strings from LLM JSON output."""
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return [str(p) for p in data if p]
    except (json.JSONDecodeError, ValueError):
        pass
    return [raw.strip()] if raw.strip() else []


def generate_prompts(state: ResearchState) -> dict[str, Any]:
    """Generate 3–5 actionable code-generation prompts from the implementation plan.

    Node inputs:  ``implementation_plan``
    Node outputs: ``code_prompts``
    """
    implementation_plan = state.get("implementation_plan", "")
    if not implementation_plan:
        return {"code_prompts": []}

    prompt = GENERATE_PROMPTS_TEMPLATE.format(implementation_plan=implementation_plan)
    llm = OllamaClient.get_llm()
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        code_prompts = _parse_prompts(raw)
    except Exception:
        logger.warning("Prompt generation failed; returning empty list")
        code_prompts = []

    logger.debug("Generated %d code prompts", len(code_prompts))
    return {"code_prompts": code_prompts}
