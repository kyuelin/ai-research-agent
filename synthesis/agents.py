"""Synthesis module: individual LangGraph agent node functions.

Each function has the signature ``(state: ResearchState) -> dict`` and returns
only the state keys it produces.  LangGraph merges the returned dict into the
running state automatically.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from llm.ollama_client import OllamaClient
from memory.vector_store import get_default_store
from synthesis.prompts import (
    ANALYZE_PAPERS_TEMPLATE,
    GENERATE_IMPLEMENTATION_TEMPLATE,
    SYNTHESIZE_FINDINGS_TEMPLATE,
)
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)


def retrieve_context(state: ResearchState) -> dict[str, Any]:
    """Retrieve relevant document chunks from ChromaDB for the query.

    Node inputs:  ``query``, ``max_results`` (optional)
    Node outputs: ``retrieved_chunks``
    """
    query = state["query"]
    k = state.get("max_results", settings.MAX_RETRIEVAL_RESULTS)
    store = get_default_store()
    results = store.similarity_search(query, k=k)
    chunks = [doc.page_content for doc in results]
    logger.debug("Retrieved %d chunks for query: %s", len(chunks), query)
    return {"retrieved_chunks": chunks}


def analyze_papers(state: ResearchState) -> dict[str, Any]:
    """Analyze retrieved chunks and extract key concepts and findings.

    Uses clustered chunks when available; falls back to scored chunks or raw
    retrieved chunks so the node works at any point in the pipeline.

    Node inputs:  ``query``, ``clusters`` | ``scores`` | ``retrieved_chunks``
    Node outputs: ``analysis``
    """
    query = state["query"]
    clusters = state.get("clusters", {})
    scored = state.get("scores", [])

    if clusters:
        parts = [
            f"[Cluster: {label}]\n" + "\n---\n".join(texts)
            for label, texts in clusters.items()
        ]
        context = "\n\n===\n\n".join(parts)
    elif scored:
        context = "\n\n---\n\n".join(c["text"] for c in scored)
    else:
        raw_chunks = state.get("retrieved_chunks", [])
        context = "\n\n---\n\n".join(raw_chunks) if raw_chunks else "No context available."

    prompt = ANALYZE_PAPERS_TEMPLATE.format(query=query, context=context)
    response = OllamaClient.get_llm().invoke(prompt)
    analysis = response.content if hasattr(response, "content") else str(response)
    return {"analysis": analysis}


def synthesize_findings(state: ResearchState) -> dict[str, Any]:
    """Synthesize the analysis into a coherent cross-paper narrative.

    Node inputs:  ``query``, ``analysis``
    Node outputs: ``synthesis``
    """
    query = state["query"]
    analysis = state.get("analysis", "")

    prompt = SYNTHESIZE_FINDINGS_TEMPLATE.format(query=query, analysis=analysis)
    response = OllamaClient.get_llm().invoke(prompt)
    synthesis = response.content if hasattr(response, "content") else str(response)
    return {"synthesis": synthesis}


def generate_implementation(state: ResearchState) -> dict[str, Any]:
    """Generate a concrete engineering implementation plan from the synthesis.

    Node inputs:  ``query``, ``synthesis``
    Node outputs: ``implementation_plan``
    """
    query = state["query"]
    synthesis = state.get("synthesis", "")

    prompt = GENERATE_IMPLEMENTATION_TEMPLATE.format(query=query, synthesis=synthesis)
    response = OllamaClient.get_llm().invoke(prompt)
    plan = response.content if hasattr(response, "content") else str(response)
    return {"implementation_plan": plan}
