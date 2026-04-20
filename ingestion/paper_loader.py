"""Ingestion module: loads research papers from PDF files, URLs, or arXiv.

Each public method returns a list of ``langchain.schema.Document`` objects
preserving the original text and enriched metadata.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader


class PaperLoader:
    """Loads AI research papers from PDF files, URLs, or arXiv IDs."""

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a research paper from a local PDF file.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of ``Document`` objects, one per page.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata.setdefault("source_type", "pdf")
        return docs

    def load_arxiv(self, arxiv_id: str) -> List[Document]:
        """Load a research paper from arXiv by its identifier.

        Args:
            arxiv_id: The arXiv paper ID, e.g. ``"2310.06825"``.

        Returns:
            List of ``Document`` objects.
        """
        loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", f"arxiv:{arxiv_id}")
            doc.metadata.setdefault("source_type", "arxiv")
        return docs

    def load_url(self, url: str) -> List[Document]:
        """Download a PDF from *url* and load it.

        Args:
            url: A direct HTTP/HTTPS link to a PDF file.

        Returns:
            List of ``Document`` objects.

        Raises:
            requests.HTTPError: If the download fails.
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        try:
            docs = self.load_pdf(tmp_path)
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["source_type"] = "url"
            return docs
        finally:
            os.unlink(tmp_path)

    def load_bytes(self, data: bytes, source_name: str = "upload") -> List[Document]:
        """Load a PDF from raw bytes (e.g. from an HTTP file upload).

        Args:
            data: Raw PDF bytes.
            source_name: Human-readable label for the ``source`` metadata field.

        Returns:
            List of ``Document`` objects.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            docs = self.load_pdf(tmp_path)
            for doc in docs:
                doc.metadata["source"] = source_name
                doc.metadata["source_type"] = "upload"
            return docs
        finally:
            os.unlink(tmp_path)
