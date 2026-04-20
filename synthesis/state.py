"""Synthesis module: shared state schema for the LangGraph workflow.

``ResearchState`` is the single mutable object that flows through every node
in the graph.  Using ``total=False`` makes all fields optional so nodes only
need to populate the fields they produce.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """State passed between nodes in the research synthesis graph."""

    query: str
    retrieved_chunks: list[str]
    analysis: str
    synthesis: str
    implementation_plan: str
