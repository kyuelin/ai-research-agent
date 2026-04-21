"""Synthesis module: LangGraph multi-agent workflow with feedback loop.

Pipeline
--------
START → retrieve_context → normalize → score_chunks ←──────────────────┐
  → cluster_chunks → analyze_papers → synthesize_findings               │
  → generate_implementation → generate_prompts → track_artifacts        │
  → create_digest → apply_feedback ──(quality < 0.8 & iter < 2)────────┘
                              └──(otherwise)──→ END

The compiled graph is cached at module level; compiled LangGraph runnables
are stateless and safe to share across concurrent requests.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from synthesis.agents import (
    analyze_papers,
    generate_implementation,
    retrieve_context,
    synthesize_findings,
)
from synthesis.artifacts import track_artifacts
from synthesis.clustering import cluster_chunks
from synthesis.digest import create_digest
from synthesis.feedback import apply_feedback, should_continue
from synthesis.normalize import normalize
from synthesis.prompt_generator import generate_prompts
from synthesis.scoring import score_chunks
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
    graph.add_node("normalize", normalize)
    graph.add_node("score_chunks", score_chunks)
    graph.add_node("cluster_chunks", cluster_chunks)
    graph.add_node("analyze_papers", analyze_papers)
    graph.add_node("synthesize_findings", synthesize_findings)
    graph.add_node("generate_implementation", generate_implementation)
    graph.add_node("generate_prompts", generate_prompts)
    graph.add_node("track_artifacts", track_artifacts)
    graph.add_node("create_digest", create_digest)
    graph.add_node("apply_feedback", apply_feedback)

    # Linear edges
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "normalize")
    graph.add_edge("normalize", "score_chunks")
    graph.add_edge("score_chunks", "cluster_chunks")
    graph.add_edge("cluster_chunks", "analyze_papers")
    graph.add_edge("analyze_papers", "synthesize_findings")
    graph.add_edge("synthesize_findings", "generate_implementation")
    graph.add_edge("generate_implementation", "generate_prompts")
    graph.add_edge("generate_prompts", "track_artifacts")
    graph.add_edge("track_artifacts", "create_digest")
    graph.add_edge("create_digest", "apply_feedback")

    # Conditional feedback edge: loop back to score_chunks or terminate
    graph.add_conditional_edges(
        "apply_feedback",
        should_continue,
        {"score_chunks": "score_chunks", "end": END},
    )

    return graph.compile()


@functools.lru_cache(maxsize=1)
def get_compiled_graph() -> Any:
    """Return the cached compiled LangGraph research synthesis workflow.

    The graph is compiled once on first call and reused across all subsequent
    requests.  Compiled LangGraph runnables are stateless, so this is safe
    under concurrent access.
    """
    return build_graph()
