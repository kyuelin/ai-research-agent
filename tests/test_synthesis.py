"""Tests for the synthesis agents and LangGraph graph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synthesis.state import ResearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_response(text: str) -> MagicMock:
    m = MagicMock()
    m.content = text
    return m


# ---------------------------------------------------------------------------
# Individual agent node tests
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def test_returns_chunks(self) -> None:
        from synthesis.agents import retrieve_context

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            MagicMock(page_content="chunk A"),
            MagicMock(page_content="chunk B"),
        ]

        with patch("synthesis.agents._get_vector_store", return_value=mock_store):
            result = retrieve_context({"query": "transformers"})

        assert result["retrieved_chunks"] == ["chunk A", "chunk B"]

    def test_empty_store_returns_empty_list(self) -> None:
        from synthesis.agents import retrieve_context

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        with patch("synthesis.agents._get_vector_store", return_value=mock_store):
            result = retrieve_context({"query": "obscure topic"})

        assert result["retrieved_chunks"] == []


class TestAnalyzePapers:
    def test_returns_analysis_string(self) -> None:
        from synthesis.agents import analyze_papers

        with patch("synthesis.agents._llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response("Key concept: attention")
            result = analyze_papers(
                {"query": "attention mechanisms", "retrieved_chunks": ["chunk 1"]}
            )

        assert "analysis" in result
        assert result["analysis"] == "Key concept: attention"

    def test_handles_missing_chunks(self) -> None:
        from synthesis.agents import analyze_papers

        with patch("synthesis.agents._llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response("No context found.")
            result = analyze_papers({"query": "test"})

        assert result["analysis"] == "No context found."


class TestSynthesizeFindings:
    def test_returns_synthesis_string(self) -> None:
        from synthesis.agents import synthesize_findings

        with patch("synthesis.agents._llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response("Synthesis narrative here.")
            result = synthesize_findings(
                {"query": "LLMs", "analysis": "Analysis text"}
            )

        assert result["synthesis"] == "Synthesis narrative here."


class TestGenerateImplementation:
    def test_returns_implementation_plan(self) -> None:
        from synthesis.agents import generate_implementation

        with patch("synthesis.agents._llm") as mock_llm:
            mock_llm.invoke.return_value = _llm_response("1. Build FastAPI app...")
            result = generate_implementation(
                {"query": "build RAG system", "synthesis": "Synthesis text"}
            )

        assert result["implementation_plan"] == "1. Build FastAPI app..."


# ---------------------------------------------------------------------------
# Graph integration test (all nodes mocked)
# ---------------------------------------------------------------------------


class TestResearchGraph:
    def test_full_pipeline_produces_all_keys(self) -> None:
        """The compiled graph should populate all four output keys."""
        from synthesis.graph import build_graph

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [MagicMock(page_content="ctx")]

        with patch("synthesis.agents._get_vector_store", return_value=mock_store):
            with patch("synthesis.agents._llm") as mock_llm:
                mock_llm.invoke.side_effect = [
                    _llm_response("analysis result"),
                    _llm_response("synthesis result"),
                    _llm_response("implementation result"),
                ]
                graph = build_graph()
                state = graph.invoke({"query": "test query"})

        assert state["query"] == "test query"
        assert state["retrieved_chunks"] == ["ctx"]
        assert state["analysis"] == "analysis result"
        assert state["synthesis"] == "synthesis result"
        assert state["implementation_plan"] == "implementation result"
