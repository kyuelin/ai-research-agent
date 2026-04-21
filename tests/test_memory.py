"""Tests for memory.vector_store.ResearchVectorStore."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from memory.vector_store import ResearchVectorStore


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


@pytest.fixture()
def mock_chroma():
    """Return a MagicMock standing in for langchain_chroma.Chroma."""
    return MagicMock()


@pytest.fixture()
def store(mock_chroma) -> ResearchVectorStore:
    """ResearchVectorStore with all external I/O replaced by mocks."""
    mock_embedder = MagicMock()
    mock_embedder.embeddings = MagicMock()

    with patch("memory.vector_store.Chroma", return_value=mock_chroma):
        vs = ResearchVectorStore(embedder=mock_embedder)
    return vs


class TestResearchVectorStore:
    def test_add_documents_delegates_to_chroma(
        self, store: ResearchVectorStore, mock_chroma: MagicMock
    ) -> None:
        docs = [_make_doc("chunk text")]
        mock_chroma.add_documents.return_value = ["id-1"]

        ids = store.add_documents(docs)

        mock_chroma.add_documents.assert_called_once_with(docs)
        assert ids == ["id-1"]

    def test_similarity_search_returns_documents(
        self, store: ResearchVectorStore, mock_chroma: MagicMock
    ) -> None:
        expected = [_make_doc("result")]
        mock_chroma.similarity_search.return_value = expected

        results = store.similarity_search("test query", k=3)

        mock_chroma.similarity_search.assert_called_once_with("test query", k=3)
        assert results == expected

    def test_similarity_search_with_score(
        self, store: ResearchVectorStore, mock_chroma: MagicMock
    ) -> None:
        doc = _make_doc("result")
        mock_chroma.similarity_search_with_score.return_value = [(doc, 0.9)]

        results = store.similarity_search_with_score("query", k=1)

        assert results[0][1] == 0.9

    def test_delete_collection_calls_chroma(
        self, store: ResearchVectorStore, mock_chroma: MagicMock
    ) -> None:
        store.delete_collection()
        mock_chroma.delete_collection.assert_called_once()

    def test_get_retriever_delegates_to_chroma(
        self, store: ResearchVectorStore, mock_chroma: MagicMock
    ) -> None:
        mock_chroma.as_retriever.return_value = MagicMock()
        retriever = store.get_retriever(k=4)
        mock_chroma.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
        assert retriever is not None
