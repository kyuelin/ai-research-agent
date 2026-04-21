"""Synthesis module: digest agent — produces a concise weekly research summary."""

from __future__ import annotations

import logging
from typing import Any

from llm.ollama_client import OllamaClient
from synthesis.prompts import DIGEST_TEMPLATE
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def create_digest(state: ResearchState) -> dict[str, Any]:
    """Produce a concise weekly research digest from the synthesis results.

    Node inputs:  ``query``, ``synthesis``, ``implementation_plan``
    Node outputs: ``digest``
    """
    query = state.get("query", "")
    synthesis = state.get("synthesis", "")
    implementation_plan = state.get("implementation_plan", "")

    prompt = DIGEST_TEMPLATE.format(
        query=query,
        synthesis=synthesis,
        implementation_plan=implementation_plan,
    )
    llm = OllamaClient.get_llm()
    try:
        response = llm.invoke(prompt)
        digest = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.warning("Digest generation failed")
        digest = ""

    logger.debug("Created digest (%d chars)", len(digest))
    return {"digest": digest}
