"""Tests for processing.chunker.DocumentChunker."""

from __future__ import annotations

from langchain_core.documents import Document

from processing.chunker import DocumentChunker


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test"})


class TestDocumentChunker:
    def test_returns_list_of_documents(self) -> None:
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        docs = [_make_doc("word " * 50)]  # ~250 chars
        chunks = chunker.chunk(docs)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Document)

    def test_chunks_long_text(self) -> None:
        """A long document should produce more than one chunk."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        long_text = "The quick brown fox jumps over the lazy dog. " * 20  # ~900 chars
        docs = [_make_doc(long_text)]
        chunks = chunker.chunk(docs)
        assert len(chunks) > 1

    def test_short_text_stays_as_single_chunk(self) -> None:
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)
        docs = [_make_doc("Short text.")]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 1

    def test_metadata_is_preserved(self) -> None:
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        docs = [_make_doc("word " * 40)]
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test"

    def test_empty_input_returns_empty_list(self) -> None:
        chunker = DocumentChunker()
        assert chunker.chunk([]) == []
