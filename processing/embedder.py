"""Processing module: generates text embeddings via an Ollama model.

Wraps ``langchain_ollama.OllamaEmbeddings`` and exposes the underlying
LangChain embeddings object so it can be passed directly to LangChain
integrations such as ``langchain_chroma.Chroma``.
"""

from __future__ import annotations

from typing import List

from langchain_ollama import OllamaEmbeddings

from config import settings


class OllamaEmbedder:
    """Generates dense vector embeddings using an Ollama embedding model."""

    def __init__(
        self,
        model: str = settings.OLLAMA_EMBED_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ) -> None:
        """Initialise the embedder.

        Args:
            model: Name of the Ollama embedding model (e.g. ``nomic-embed-text``).
            base_url: Base URL of the Ollama server.
        """
        self._embeddings = OllamaEmbeddings(model=model, base_url=base_url)

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """The underlying LangChain ``Embeddings`` instance.

        This is the object expected by ``langchain_chroma.Chroma`` and other
        LangChain vector-store integrations.
        """
        return self._embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of plain text strings.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors, one per input string.
        """
        return self._embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.

        Args:
            query: The search query.

        Returns:
            A single embedding vector.
        """
        return self._embeddings.embed_query(query)
