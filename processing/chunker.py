"""Processing module: splits ``Document`` objects into overlapping text chunks.

Uses ``RecursiveCharacterTextSplitter`` from LangChain so that chunks respect
natural text boundaries (paragraphs → sentences → words).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


class DocumentChunker:
    """Splits documents into chunks suitable for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters shared between consecutive chunks.
        """
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into overlapping text chunks.

        Args:
            documents: Source ``Document`` objects (e.g. from ``PaperLoader``).

        Returns:
            List of chunked ``Document`` objects with preserved metadata and an
            added ``start_index`` metadata field indicating the byte offset.
        """
        return self._splitter.split_documents(documents)
