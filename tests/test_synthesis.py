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


def _make_chunk(idx: int = 0, text: str = "chunk text") -> dict:
    return {
        "index": idx,
        "text": text,
        "source": "test",
        "source_type": "retrieved",
        "ingested_at": "2026-01-01T00:00:00+00:00",
        "scores": {"novelty": 0.5, "practicality": 0.5, "adoption": 0.5, "relevance": 0.5},
        "composite_score": 0.5,
    }


# ---------------------------------------------------------------------------
# retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def test_returns_chunks(self) -> None:
        from synthesis.agents import retrieve_context

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            MagicMock(page_content="chunk A"),
            MagicMock(page_content="chunk B"),
        ]

        with patch("synthesis.agents.get_default_store", return_value=mock_store):
            result = retrieve_context({"query": "transformers"})

        assert result["retrieved_chunks"] == ["chunk A", "chunk B"]

    def test_empty_store_returns_empty_list(self) -> None:
        from synthesis.agents import retrieve_context

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        with patch("synthesis.agents.get_default_store", return_value=mock_store):
            result = retrieve_context({"query": "obscure topic"})

        assert result["retrieved_chunks"] == []

    def test_max_results_forwarded(self) -> None:
        from synthesis.agents import retrieve_context

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        with patch("synthesis.agents.get_default_store", return_value=mock_store):
            retrieve_context({"query": "q", "max_results": 3})

        mock_store.similarity_search.assert_called_once_with("q", k=3)


# ---------------------------------------------------------------------------
# analyze_papers
# ---------------------------------------------------------------------------


class TestAnalyzePapers:
    def test_returns_analysis_string(self) -> None:
        from synthesis.agents import analyze_papers

        with patch("synthesis.agents.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response("Key concept: attention")
            result = analyze_papers(
                {"query": "attention mechanisms", "retrieved_chunks": ["chunk 1"]}
            )

        assert "analysis" in result
        assert result["analysis"] == "Key concept: attention"

    def test_uses_clusters_when_available(self) -> None:
        from synthesis.agents import analyze_papers

        with patch("synthesis.agents.OllamaClient.get_llm") as mock_get_llm:
            mock_llm = mock_get_llm.return_value
            mock_llm.invoke.return_value = _llm_response("cluster analysis")
            result = analyze_papers(
                {
                    "query": "test",
                    "clusters": {"topic_a": ["text 1"], "topic_b": ["text 2"]},
                }
            )

        assert result["analysis"] == "cluster analysis"
        # Verify the prompt contained cluster labels
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "Cluster: topic_a" in prompt_arg

    def test_handles_missing_chunks(self) -> None:
        from synthesis.agents import analyze_papers

        with patch("synthesis.agents.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response("No context found.")
            result = analyze_papers({"query": "test"})

        assert result["analysis"] == "No context found."


# ---------------------------------------------------------------------------
# synthesize_findings
# ---------------------------------------------------------------------------


class TestSynthesizeFindings:
    def test_returns_synthesis_string(self) -> None:
        from synthesis.agents import synthesize_findings

        with patch("synthesis.agents.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response("Synthesis narrative here.")
            result = synthesize_findings(
                {"query": "LLMs", "analysis": "Analysis text"}
            )

        assert result["synthesis"] == "Synthesis narrative here."


# ---------------------------------------------------------------------------
# generate_implementation
# ---------------------------------------------------------------------------


class TestGenerateImplementation:
    def test_returns_implementation_plan(self) -> None:
        from synthesis.agents import generate_implementation

        with patch("synthesis.agents.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response("1. Build FastAPI app...")
            result = generate_implementation(
                {"query": "build RAG system", "synthesis": "Synthesis text"}
            )

        assert result["implementation_plan"] == "1. Build FastAPI app..."


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_produces_normalized_chunks(self) -> None:
        from synthesis.normalize import normalize

        result = normalize({"retrieved_chunks": ["text A", "text B"]})
        chunks = result["normalized_chunks"]
        assert len(chunks) == 2
        assert chunks[0]["text"] == "text A"
        assert chunks[0]["index"] == 0
        assert "ingested_at" in chunks[0]
        assert chunks[1]["index"] == 1

    def test_empty_input(self) -> None:
        from synthesis.normalize import normalize

        result = normalize({"retrieved_chunks": []})
        assert result["normalized_chunks"] == []


# ---------------------------------------------------------------------------
# score_chunks
# ---------------------------------------------------------------------------


class TestScoreChunks:
    def test_returns_sorted_scored_chunks(self) -> None:
        from synthesis.scoring import score_chunks

        chunks = [_make_chunk(0, "text A"), _make_chunk(1, "text B")]
        state: ResearchState = {"query": "q", "normalized_chunks": chunks}

        with patch("synthesis.scoring.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response(
                '{"novelty": 0.8, "practicality": 0.7, "adoption": 0.6, "relevance": 0.9}'
            )
            result = score_chunks(state)

        scored = result["scores"]
        assert len(scored) == 2
        assert "composite_score" in scored[0]
        assert scored[0]["composite_score"] >= scored[1]["composite_score"]

    def test_applies_feedback_weight_adjustments(self) -> None:
        from synthesis.scoring import score_chunks

        chunks = [_make_chunk(0)]
        state: ResearchState = {
            "query": "q",
            "normalized_chunks": chunks,
            "feedback": {"weight_adjustments": {"novelty": 0.2}},
        }

        with patch("synthesis.scoring.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response(
                '{"novelty": 1.0, "practicality": 0.0, "adoption": 0.0, "relevance": 0.0}'
            )
            result = score_chunks(state)

        # With adjusted novelty weight (0.45), composite > 0.25 baseline
        assert result["scores"][0]["composite_score"] > 0.25


# ---------------------------------------------------------------------------
# cluster_chunks
# ---------------------------------------------------------------------------


class TestClusterChunks:
    def test_returns_clusters(self) -> None:
        from synthesis.clustering import cluster_chunks

        scored = [_make_chunk(0, "attention mechanism"), _make_chunk(1, "transformer model")]
        state: ResearchState = {"scores": scored}

        with patch("synthesis.clustering.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response(
                '{"architectures": [0, 1]}'
            )
            result = cluster_chunks(state)

        assert "architectures" in result["clusters"]
        assert len(result["clusters"]["architectures"]) == 2

    def test_empty_scores_returns_empty_clusters(self) -> None:
        from synthesis.clustering import cluster_chunks

        result = cluster_chunks({"scores": []})
        assert result["clusters"] == {}


# ---------------------------------------------------------------------------
# track_artifacts
# ---------------------------------------------------------------------------


class TestTrackArtifacts:
    def test_creates_implementation_plan_artifact(self) -> None:
        from synthesis.artifacts import track_artifacts

        scored = [_make_chunk(0)]
        result = track_artifacts(
            {"scores": scored, "implementation_plan": "build X", "code_prompts": []}
        )
        artifacts = result["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["artefact_type"] == "implementation_plan"

    def test_creates_code_prompt_artifacts(self) -> None:
        from synthesis.artifacts import track_artifacts

        scored = [_make_chunk(0)]
        result = track_artifacts(
            {
                "scores": scored,
                "implementation_plan": "plan",
                "code_prompts": ["write unit tests", "build FastAPI endpoint"],
            }
        )
        types = [a["artefact_type"] for a in result["artifacts"]]
        assert types.count("code_prompt") == 2


# ---------------------------------------------------------------------------
# generate_prompts
# ---------------------------------------------------------------------------


class TestGeneratePrompts:
    def test_returns_list_of_prompts(self) -> None:
        from synthesis.prompt_generator import generate_prompts

        with patch("synthesis.prompt_generator.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response(
                '["Write a FastAPI endpoint", "Implement ChromaDB indexing"]'
            )
            result = generate_prompts({"implementation_plan": "some plan"})

        assert len(result["code_prompts"]) == 2

    def test_empty_plan_returns_empty(self) -> None:
        from synthesis.prompt_generator import generate_prompts

        result = generate_prompts({"implementation_plan": ""})
        assert result["code_prompts"] == []


# ---------------------------------------------------------------------------
# create_digest
# ---------------------------------------------------------------------------


class TestCreateDigest:
    def test_returns_digest_string(self) -> None:
        from synthesis.digest import create_digest

        with patch("synthesis.digest.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response("Weekly digest here.")
            result = create_digest(
                {"query": "LLMs", "synthesis": "...", "implementation_plan": "..."}
            )

        assert result["digest"] == "Weekly digest here."


# ---------------------------------------------------------------------------
# apply_feedback / should_continue
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_returns_feedback_and_increments_iteration(self) -> None:
        from synthesis.feedback import apply_feedback

        with patch("synthesis.feedback.OllamaClient.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _llm_response(
                '{"weight_adjustments": {"novelty": 0.1}, "prompt_quality": 0.6, "prompt_improvement": "add examples"}'
            )
            result = apply_feedback({"digest": "digest text", "iteration": 0})

        assert result["iteration"] == 1
        assert result["feedback"]["prompt_quality"] == 0.6

    def test_should_continue_loops_on_low_quality(self) -> None:
        from synthesis.feedback import should_continue

        state: ResearchState = {"iteration": 0, "feedback": {"prompt_quality": 0.5}}
        assert should_continue(state) == "score_chunks"

    def test_should_continue_ends_on_high_quality(self) -> None:
        from synthesis.feedback import should_continue

        state: ResearchState = {"iteration": 0, "feedback": {"prompt_quality": 0.9}}
        assert should_continue(state) == "end"

    def test_should_continue_ends_on_max_iterations(self) -> None:
        from synthesis.feedback import should_continue

        state: ResearchState = {"iteration": 2, "feedback": {"prompt_quality": 0.5}}
        assert should_continue(state) == "end"


# ---------------------------------------------------------------------------
# Graph integration test (all nodes mocked)
# ---------------------------------------------------------------------------


class TestResearchGraph:
    def test_full_pipeline_produces_all_keys(self) -> None:
        """The compiled graph should populate all output keys."""
        from synthesis.graph import build_graph
        from llm.ollama_client import OllamaClient

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [MagicMock(page_content="ctx")]

        # All synthesis modules share the same OllamaClient class; patch once.
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            # score_chunks (1 chunk × 1 call)
            _llm_response('{"novelty": 0.5, "practicality": 0.5, "adoption": 0.5, "relevance": 0.5}'),
            # cluster_chunks
            _llm_response('{"general": [0]}'),
            # analyze_papers
            _llm_response("analysis result"),
            # synthesize_findings
            _llm_response("synthesis result"),
            # generate_implementation
            _llm_response("implementation result"),
            # generate_prompts
            _llm_response('["prompt 1"]'),
            # create_digest
            _llm_response("digest text"),
            # apply_feedback — high quality → terminates
            _llm_response(
                '{"weight_adjustments": {}, "prompt_quality": 0.9, "prompt_improvement": ""}'
            ),
        ]

        with patch("synthesis.agents.get_default_store", return_value=mock_store):
            with patch.object(OllamaClient, "get_llm", return_value=mock_llm):
                graph = build_graph()
                state = graph.invoke({"query": "test query"})

        assert state["query"] == "test query"
        assert state["analysis"] == "analysis result"
        assert state["synthesis"] == "synthesis result"
        assert state["implementation_plan"] == "implementation result"
        assert state["digest"] == "digest text"
        assert "artifacts" in state
        assert "code_prompts" in state
