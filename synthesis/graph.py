"""Synthesis module: LangGraph multi-agent workflow.

Pipeline
--------
START → retrieve_context → analyze_papers → synthesize_findings
      → generate_implementation → END

Each edge is unconditional; the graph runs every node in sequence.
The compiled graph is safe to reuse across requests.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from synthesis.agents import (
    analyze_papers,
    generate_implementation,
    retrieve_context,
    synthesize_findings,
)
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def build_graph() -> Any:
    """Build and compile the research synthesis ``StateGraph``.

    Returns:
        A compiled LangGraph runnable that accepts a ``ResearchState`` dict
        containing at minimum a ``query`` key and returns a fully populated
        ``ResearchState``.
    """
    graph: StateGraph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("analyze_papers", analyze_papers)
    graph.add_node("synthesize_findings", synthesize_findings)
    graph.add_node("generate_implementation", generate_implementation)

    # Wire edges
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_papers")
    graph.add_edge("analyze_papers", "synthesize_findings")
    graph.add_edge("synthesize_findings", "generate_implementation")
    graph.add_edge("generate_implementation", END)

    return graph.compile()


def get_compiled_graph() -> Any:
    """Return a fresh compiled LangGraph research synthesis workflow.

    A new graph is compiled on each call so the module remains import-safe and
    there are no shared-state concerns between concurrent requests when the
    API layer uses separate threads/processes.
    """
    return build_graph()
