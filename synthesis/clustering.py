"""Synthesis module: clustering agent — groups related chunks into thematic clusters."""

from __future__ import annotations

import json
import logging
from typing import Any

from llm.ollama_client import OllamaClient
from synthesis.prompts import CLUSTER_CHUNKS_TEMPLATE
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def _parse_clusters(raw: str, num_chunks: int) -> dict[str, list[int]]:
    """Parse LLM cluster JSON; return a mapping of cluster name → list of chunk indices."""
    try:
        data = json.loads(raw.strip())
        return {
            str(k): [int(i) for i in v if str(i).isdigit()]
            for k, v in data.items()
            if isinstance(v, list)
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"general": list(range(num_chunks))}


def cluster_chunks(state: ResearchState) -> dict[str, Any]:
    """Group scored chunks into thematic clusters.

    Node inputs:  ``scores``
    Node outputs: ``clusters`` (cluster_label → list of chunk texts)
    """
    scored_chunks = state.get("scores", [])
    if not scored_chunks:
        return {"clusters": {}}

    excerpts = "\n".join(
        f"[{i}] {chunk['text'][:300]}" for i, chunk in enumerate(scored_chunks)
    )
    prompt = CLUSTER_CHUNKS_TEMPLATE.format(excerpts=excerpts)

    llm = OllamaClient.get_llm()
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        index_clusters = _parse_clusters(raw, len(scored_chunks))
    except Exception:
        logger.warning("Clustering failed; placing all chunks in 'general' cluster")
        index_clusters = {"general": list(range(len(scored_chunks)))}

    clusters: dict[str, list[str]] = {}
    for label, indices in index_clusters.items():
        texts: list[str] = []
        for idx in indices:
            try:
                texts.append(scored_chunks[idx]["text"])
            except IndexError:
                pass
        if texts:
            clusters[label] = texts

    logger.debug("Produced %d clusters from %d chunks", len(clusters), len(scored_chunks))
    return {"clusters": clusters}
