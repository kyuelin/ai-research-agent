"""LLM module: singleton wrappers for Ollama chat and embedding models.

All application code should obtain LLM/embeddings instances via this module
so that configuration is centralised and model objects are reused across calls.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import settings


class OllamaClient:
    """Lazy-initialised singleton accessors for Ollama chat and embedding models.

    Models are constructed at most once per process; subsequent calls return the
    cached instance.  No network I/O occurs until ``.invoke()``/``.embed_*()``
    is called on the returned object.
    """

    _llm: ChatOllama | None = None
    _embeddings: OllamaEmbeddings | None = None

    @classmethod
    def get_llm(
        cls,
        model: str = settings.OLLAMA_LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ) -> ChatOllama:
        """Return the shared ``ChatOllama`` instance.

        Args:
            model: Ollama model name (default: ``settings.OLLAMA_LLM_MODEL``).
            base_url: Ollama server base URL.

        Returns:
            A ``ChatOllama`` instance ready for ``.invoke()``.
        """
        if cls._llm is None:
            cls._llm = ChatOllama(model=model, base_url=base_url)
        return cls._llm

    @classmethod
    def get_embeddings(
        cls,
        model: str = settings.OLLAMA_EMBED_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
    ) -> OllamaEmbeddings:
        """Return the shared ``OllamaEmbeddings`` instance.

        Args:
            model: Ollama embedding model name (default: ``settings.OLLAMA_EMBED_MODEL``).
            base_url: Ollama server base URL.

        Returns:
            An ``OllamaEmbeddings`` instance compatible with LangChain integrations.
        """
        if cls._embeddings is None:
            cls._embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        return cls._embeddings
