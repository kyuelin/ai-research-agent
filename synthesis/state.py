"""Synthesis module: shared state schema for the LangGraph workflow.

``ResearchState`` is the single mutable object that flows through every node
in the graph.  Using ``total=False`` makes all fields optional so nodes only
need to populate the fields they produce.
"""

from __future__ import annotations

from typing import List

from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """State passed between nodes in the research synthesis graph.

    Fields
    ------
    query:
        The original research question submitted by the user.
    retrieved_chunks:
        Raw text excerpts retrieved from the vector store.
    analysis:
        Key concepts, methods, and findings extracted by the analysis agent.
    synthesis:
        Coherent narrative synthesising the analysis across papers.
    implementation_plan:
        Concrete, production-ready engineering plan derived from the synthesis.
    """

    query: str
    retrieved_chunks: List[str]
    analysis: str
    synthesis: str
    implementation_plan: str
