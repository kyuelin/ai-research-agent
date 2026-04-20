"""Synthesis module: shared state schema for the LangGraph workflow.

``ResearchState`` is the single mutable object that flows through every node
in the graph.  Using ``total=False`` makes all fields optional so nodes only
need to populate the fields they produce.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """State passed between nodes in the research synthesis graph."""

    # --- core inputs ---
    query: str
    max_results: int

    # --- retrieval & normalisation ---
    retrieved_chunks: list[str]
    normalized_chunks: list[dict[str, Any]]

    # --- scoring & clustering ---
    scores: list[dict[str, Any]]
    clusters: dict[str, list[str]]

    # --- analysis & synthesis ---
    analysis: str
    synthesis: str
    implementation_plan: str

    # --- downstream artefacts ---
    code_prompts: list[str]
    artifacts: list[dict[str, Any]]

    # --- digest & feedback loop ---
    digest: str
    feedback: dict[str, Any]
    iteration: int
