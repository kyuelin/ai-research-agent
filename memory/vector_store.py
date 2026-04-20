"""Memory module: persistent ChromaDB-backed vector store.

Wraps ``langchain_chroma.Chroma`` so that the rest of the application never
imports ChromaDB directly — all vector-store operations go through this class.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import settings
from processing.embedder import OllamaEmbedder

# ---------------------------------------------------------------------------
# Module-level singleton — one ChromaDB connection per process
# ---------------------------------------------------------------------------
_default_store: Optional["ResearchVectorStore"] = None


def get_default_store() -> "ResearchVectorStore":
    """Return the process-wide singleton ``ResearchVectorStore``.

    Lazily initialised on first call; reused on subsequent calls to avoid
    opening redundant ChromaDB connections.
    """
    global _default_store
    if _default_store is None:
        _default_store = ResearchVectorStore()
    return _default_store


class ResearchVectorStore:
    """Persistent vector store for research paper chunks (ChromaDB backend)."""

    def __init__(
        self,
        collection_name: str = settings.CHROMA_COLLECTION,
        persist_directory: str = settings.CHROMA_PERSIST_DIR,
        embedder: Optional[OllamaEmbedder] = None,
    ) -> None:
        self._embedder = embedder or OllamaEmbedder()
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embedder.embeddings,
            persist_directory=persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add document chunks to the vector store."""
        return self._store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = settings.MAX_RETRIEVAL_RESULTS,
    ) -> list[Document]:
        """Return the *k* most similar documents for *query*."""
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = settings.MAX_RETRIEVAL_RESULTS,
    ) -> list[tuple[Document, float]]:
        """Return documents together with their similarity scores."""
        return self._store.similarity_search_with_score(query, k=k)

    def delete_collection(self) -> None:
        """Delete all documents from the collection."""
        self._store.delete_collection()

    def get_retriever(self, k: int = settings.MAX_RETRIEVAL_RESULTS):
        """Return a LangChain ``BaseRetriever`` interface for this store."""
        return self._store.as_retriever(search_kwargs={"k": k})
