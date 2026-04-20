"""Memory module: persistent ChromaDB-backed vector store.

Wraps ``langchain_chroma.Chroma`` so that the rest of the application never
imports ChromaDB directly — all vector-store operations go through this class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import settings
from processing.embedder import OllamaEmbedder


class ResearchVectorStore:
    """Persistent vector store for research paper chunks (ChromaDB backend).

    The store is created automatically when the object is instantiated.  Data
    is persisted to *persist_directory* so it survives process restarts.
    """

    def __init__(
        self,
        collection_name: str = settings.CHROMA_COLLECTION,
        persist_directory: str = settings.CHROMA_PERSIST_DIR,
        embedder: Optional[OllamaEmbedder] = None,
    ) -> None:
        """Initialise the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory where ChromaDB persists its data.
            embedder: ``OllamaEmbedder`` to use.  A default instance is created
                when *None* is passed.
        """
        self._embedder = embedder or OllamaEmbedder()
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embedder.embeddings,
            persist_directory=persist_directory,
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add document chunks to the vector store.

        Args:
            documents: Chunked ``Document`` objects produced by
                ``DocumentChunker``.

        Returns:
            List of ChromaDB document IDs assigned to each chunk.
        """
        return self._store.add_documents(documents)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = settings.MAX_RETRIEVAL_RESULTS,
    ) -> List[Document]:
        """Return the *k* most similar documents for *query*.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.

        Returns:
            List of ``Document`` objects ordered by similarity (most similar
            first).
        """
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = settings.MAX_RETRIEVAL_RESULTS,
    ) -> List[Tuple[Document, float]]:
        """Return documents together with their similarity scores.

        Args:
            query: Natural-language search query.
            k: Maximum number of results to return.

        Returns:
            List of ``(Document, score)`` tuples ordered by similarity.
        """
        return self._store.similarity_search_with_score(query, k=k)

    def delete_collection(self) -> None:
        """Delete all documents from the collection.

        Primarily useful for tests and administrative resets.
        """
        self._store.delete_collection()

    def get_retriever(self, k: int = settings.MAX_RETRIEVAL_RESULTS):
        """Return a LangChain ``BaseRetriever`` interface for this store.

        Args:
            k: Number of documents to retrieve per query.

        Returns:
            A ``VectorStoreRetriever`` compatible with LangChain chains.
        """
        return self._store.as_retriever(search_kwargs={"k": k})
