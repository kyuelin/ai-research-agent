"""Synthesis module: individual LangGraph agent node functions.

Each function has the signature ``(state: ResearchState) -> dict`` and returns
only the state keys it produces.  LangGraph merges the returned dict into the
running state automatically.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_ollama import ChatOllama

from config import settings
from memory.vector_store import ResearchVectorStore
from synthesis.state import ResearchState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM — constructed once at import time (import-safe: no network call
# happens until `.invoke()` is called).
# ---------------------------------------------------------------------------
_llm = ChatOllama(
    model=settings.OLLAMA_LLM_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
)


def _get_vector_store() -> ResearchVectorStore:
    """Return a ``ResearchVectorStore`` instance with default settings."""
    return ResearchVectorStore()


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def retrieve_context(state: ResearchState) -> Dict[str, Any]:
    """Retrieve relevant document chunks from ChromaDB for the query.

    Node inputs:  ``query``
    Node outputs: ``retrieved_chunks``
    """
    query = state["query"]
    store = _get_vector_store()
    results = store.similarity_search(query, k=settings.MAX_RETRIEVAL_RESULTS)
    chunks = [doc.page_content for doc in results]
    logger.debug("Retrieved %d chunks for query: %s", len(chunks), query)
    return {"retrieved_chunks": chunks}


def analyze_papers(state: ResearchState) -> Dict[str, Any]:
    """Analyse retrieved chunks and extract key concepts and findings.

    Node inputs:  ``query``, ``retrieved_chunks``
    Node outputs: ``analysis``
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n---\n\n".join(chunks) if chunks else "No context available."

    prompt = (
        f"You are a research analyst. The user asks: '{query}'\n\n"
        f"Below are excerpts from relevant research papers:\n\n{context}\n\n"
        "Identify and explain the key concepts, methods, and findings from these papers "
        "that are most relevant to the query. Be concise and technical."
    )
    response = _llm.invoke(prompt)
    analysis = response.content if hasattr(response, "content") else str(response)
    return {"analysis": analysis}


def synthesize_findings(state: ResearchState) -> Dict[str, Any]:
    """Synthesise the analysis into a coherent cross-paper narrative.

    Node inputs:  ``query``, ``analysis``
    Node outputs: ``synthesis``
    """
    query = state["query"]
    analysis = state.get("analysis", "")

    prompt = (
        f"You are a research synthesis expert. The user asks: '{query}'\n\n"
        f"Based on the following analysis of research papers:\n\n{analysis}\n\n"
        "Synthesize the key findings into a coherent narrative. "
        "Highlight agreements, contradictions, and research gaps. "
        "Conclude with the current state of the art."
    )
    response = _llm.invoke(prompt)
    synthesis = response.content if hasattr(response, "content") else str(response)
    return {"synthesis": synthesis}


def generate_implementation(state: ResearchState) -> Dict[str, Any]:
    """Generate a concrete engineering implementation plan from the synthesis.

    Node inputs:  ``query``, ``synthesis``
    Node outputs: ``implementation_plan``
    """
    query = state["query"]
    synthesis = state.get("synthesis", "")

    prompt = (
        f"You are a senior AI systems engineer. The user asks: '{query}'\n\n"
        f"Based on this research synthesis:\n\n{synthesis}\n\n"
        "Generate a concrete, production-ready implementation plan. Include:\n"
        "1. System architecture (modules, data flow)\n"
        "2. Technology stack and rationale\n"
        "3. Key algorithms or model choices\n"
        "4. Data pipeline design\n"
        "5. API design (endpoints, schemas)\n"
        "6. Testing strategy\n\n"
        "Be specific and actionable. Prefer Python, LangChain, FastAPI, and ChromaDB."
    )
    response = _llm.invoke(prompt)
    plan = response.content if hasattr(response, "content") else str(response)
    return {"implementation_plan": plan}
