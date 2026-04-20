"""Tests for ingestion.paper_loader.PaperLoader."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from ingestion.paper_loader import PaperLoader


@pytest.fixture()
def loader() -> PaperLoader:
    return PaperLoader()


@pytest.fixture()
def sample_docs() -> list[Document]:
    return [Document(page_content="Hello research world", metadata={})]


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------


class TestLoadPdf:
    def test_raises_when_file_missing(self, loader: PaperLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("/nonexistent/path/paper.pdf")

    def test_returns_documents_with_metadata(
        self, loader: PaperLoader, sample_docs: list[Document]
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        try:
            with patch(
                "ingestion.paper_loader.PyPDFLoader"
            ) as MockLoader:
                MockLoader.return_value.load.return_value = sample_docs
                docs = loader.load_pdf(tmp_path)

            assert len(docs) == 1
            assert docs[0].metadata["source"] == tmp_path
            assert docs[0].metadata["source_type"] == "pdf"
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# load_arxiv
# ---------------------------------------------------------------------------


class TestLoadArxiv:
    def test_sets_arxiv_metadata(
        self, loader: PaperLoader, sample_docs: list[Document]
    ) -> None:
        with patch("ingestion.paper_loader.ArxivLoader") as MockLoader:
            MockLoader.return_value.load.return_value = sample_docs
            docs = loader.load_arxiv("2310.06825")

        assert docs[0].metadata["source"] == "arxiv:2310.06825"
        assert docs[0].metadata["source_type"] == "arxiv"


# ---------------------------------------------------------------------------
# load_url
# ---------------------------------------------------------------------------


class TestLoadUrl:
    def test_downloads_and_loads_pdf(
        self, loader: PaperLoader, sample_docs: list[Document]
    ) -> None:
        fake_response = MagicMock()
        fake_response.content = b"%PDF-1.4 fake content"
        fake_response.raise_for_status = MagicMock()

        # Patch _validate_url to skip the real DNS lookup, and requests.get
        with patch("ingestion.paper_loader._validate_url"):
            with patch(
                "ingestion.paper_loader.requests.get", return_value=fake_response
            ):
                with patch.object(loader, "load_pdf", return_value=sample_docs):
                    docs = loader.load_url("https://example.com/paper.pdf")

        assert docs[0].metadata["source"] == "https://example.com/paper.pdf"
        assert docs[0].metadata["source_type"] == "url"

    def test_raises_on_http_error(self, loader: PaperLoader) -> None:
        import requests as req_lib

        with patch("ingestion.paper_loader._validate_url"):
            with patch(
                "ingestion.paper_loader.requests.get",
                side_effect=req_lib.HTTPError("404"),
            ):
                with pytest.raises(req_lib.HTTPError):
                    loader.load_url("https://example.com/missing.pdf")

    def test_rejects_private_ip(self, loader: PaperLoader) -> None:
        """load_url must block requests to private/internal addresses."""
        with pytest.raises(ValueError, match="not permitted"):
            loader.load_url("http://192.168.1.1/paper.pdf")

    def test_rejects_loopback(self, loader: PaperLoader) -> None:
        with pytest.raises(ValueError, match="not permitted"):
            loader.load_url("http://127.0.0.1/paper.pdf")

    def test_rejects_non_http_scheme(self, loader: PaperLoader) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            loader.load_url("ftp://example.com/paper.pdf")


# ---------------------------------------------------------------------------
# load_bytes
# ---------------------------------------------------------------------------


class TestLoadBytes:
    def test_sets_upload_metadata(
        self, loader: PaperLoader, sample_docs: list[Document]
    ) -> None:
        with patch.object(loader, "load_pdf", return_value=sample_docs):
            docs = loader.load_bytes(b"%PDF fake", source_name="my_paper.pdf")

        assert docs[0].metadata["source"] == "my_paper.pdf"
        assert docs[0].metadata["source_type"] == "upload"
