"""Tests for the FastAPI application (api.main)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str = "sample text") -> Document:
    return Document(page_content=text, metadata={"source": "test"})


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_ok(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ---------------------------------------------------------------------------
# /ingest/pdf
# ---------------------------------------------------------------------------


class TestIngestPdf:
    def _post_pdf(self, filename: str = "paper.pdf"):
        return client.post(
            "/ingest/pdf",
            files={"file": (filename, BytesIO(b"%PDF fake"), "application/pdf")},
        )

    def test_successful_ingest(self) -> None:
        docs = [_make_doc()]
        chunks = [_make_doc("chunk")]

        with patch("api.main._loader.load_bytes", return_value=docs):
            with patch("api.main._chunker.chunk", return_value=chunks):
                with patch("api.main._get_store") as mock_store_factory:
                    mock_store_factory.return_value.add_documents.return_value = ["id1"]
                    resp = self._post_pdf()

        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_added"] == 1
        assert data["source"] == "paper.pdf"

    def test_rejects_non_pdf(self) -> None:
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("doc.txt", BytesIO(b"text"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_returns_500_on_loader_error(self) -> None:
        with patch("api.main._loader.load_bytes", side_effect=RuntimeError("boom")):
            resp = self._post_pdf()
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /ingest/arxiv
# ---------------------------------------------------------------------------


class TestIngestArxiv:
    def test_successful_arxiv_ingest(self) -> None:
        docs = [_make_doc()]
        chunks = [_make_doc("chunk")]

        with patch("api.main._loader.load_arxiv", return_value=docs):
            with patch("api.main._chunker.chunk", return_value=chunks):
                with patch("api.main._get_store") as mock_store_factory:
                    mock_store_factory.return_value.add_documents.return_value = ["id1"]
                    resp = client.post(
                        "/ingest/arxiv", json={"arxiv_id": "2310.06825"}
                    )

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "arxiv:2310.06825"
        assert data["chunks_added"] == 1

    def test_returns_500_on_loader_error(self) -> None:
        with patch("api.main._loader.load_arxiv", side_effect=ValueError("bad id")):
            resp = client.post("/ingest/arxiv", json={"arxiv_id": "bad"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /ingest/url
# ---------------------------------------------------------------------------


class TestIngestUrl:
    def test_successful_url_ingest(self) -> None:
        docs = [_make_doc()]
        chunks = [_make_doc("chunk")]

        with patch("api.main._loader.load_url", return_value=docs):
            with patch("api.main._chunker.chunk", return_value=chunks):
                with patch("api.main._get_store") as mock_store_factory:
                    mock_store_factory.return_value.add_documents.return_value = ["id1"]
                    resp = client.post(
                        "/ingest/url",
                        json={"url": "https://example.com/paper.pdf"},
                    )

        assert resp.status_code == 200
        assert resp.json()["source"] == "https://example.com/paper.pdf"

    def test_returns_500_on_download_error(self) -> None:
        with patch("api.main._loader.load_url", side_effect=IOError("download fail")):
            resp = client.post(
                "/ingest/url", json={"url": "https://example.com/paper.pdf"}
            )
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_successful_synthesis(self) -> None:
        fake_result = {
            "query": "attention mechanisms",
            "retrieved_chunks": ["chunk"],
            "analysis": "analysis text",
            "synthesis": "synthesis text",
            "implementation_plan": "plan text",
        }

        with patch("api.main.get_compiled_graph") as mock_graph_factory:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = fake_result
            mock_graph_factory.return_value = mock_graph

            resp = client.post(
                "/synthesize", json={"query": "attention mechanisms"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "attention mechanisms"
        assert data["analysis"] == "analysis text"
        assert data["synthesis"] == "synthesis text"
        assert data["implementation_plan"] == "plan text"

    def test_returns_500_on_graph_error(self) -> None:
        with patch(
            "api.main.get_compiled_graph",
            side_effect=RuntimeError("LLM offline"),
        ):
            resp = client.post("/synthesize", json={"query": "test"})
        assert resp.status_code == 500
