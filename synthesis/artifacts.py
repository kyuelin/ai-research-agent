"""Synthesis module: artifacts agent — records paper-to-solution lineage."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def track_artifacts(state: ResearchState) -> dict[str, Any]:
    """Build a lineage record mapping source papers to generated artefacts.

    Creates one ``implementation_plan`` artefact entry and one entry per
    generated code prompt so that the provenance chain from raw papers to
    concrete engineering outputs is fully traceable.

    Node inputs:  ``scores``, ``implementation_plan``, ``code_prompts`` (optional)
    Node outputs: ``artifacts``
    """
    scored_chunks = state.get("scores", [])
    implementation_plan = state.get("implementation_plan", "")
    code_prompts = state.get("code_prompts", [])
    now = datetime.now(tz=timezone.utc).isoformat()

    sources: list[str] = list({chunk.get("source", "unknown") for chunk in scored_chunks})

    artifacts: list[dict[str, Any]] = [
        {
            "created_at": now,
            "source_papers": sources,
            "artefact_type": "implementation_plan",
            "content_preview": implementation_plan[:200],
        }
    ]

    for i, prompt in enumerate(code_prompts):
        artifacts.append(
            {
                "created_at": now,
                "source_papers": sources,
                "artefact_type": "code_prompt",
                "artefact_index": i,
                "content_preview": prompt[:200],
            }
        )

    logger.debug("Tracked %d artifacts from %d sources", len(artifacts), len(sources))
    return {"artifacts": artifacts}
