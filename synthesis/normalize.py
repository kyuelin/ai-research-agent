"""Synthesis module: normalize agent — enforces consistent dated metadata on chunks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def normalize(state: ResearchState) -> dict[str, Any]:
    """Enforce consistent metadata on every retrieved chunk.

    Each normalized chunk is a dict containing at minimum:
    ``index``, ``text``, ``source``, ``source_type``, and ``ingested_at``
    (ISO-8601 UTC timestamp).

    Node inputs:  ``retrieved_chunks``
    Node outputs: ``normalized_chunks``
    """
    raw_chunks = state.get("retrieved_chunks", [])
    now = datetime.now(tz=timezone.utc).isoformat()
    normalized: list[dict[str, Any]] = []

    for i, chunk in enumerate(raw_chunks):
        if isinstance(chunk, dict):
            normalized_chunk = {
                "index": i,
                "text": chunk.get("text", ""),
                "source": chunk.get("source") or "unknown",
                "source_type": chunk.get("source_type") or "retrieved",
                "ingested_at": chunk.get("ingested_at") or now,
            }
        else:
            normalized_chunk = {
                "index": i,
                "text": chunk,
                "source": "unknown",
                "source_type": "retrieved",
                "ingested_at": now,
            }
        normalized.append(normalized_chunk)
    logger.debug("Normalized %d chunks", len(normalized))
    return {"normalized_chunks": normalized}
