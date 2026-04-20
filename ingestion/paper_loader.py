"""Ingestion module: loads research papers from PDF files, URLs, or arXiv.

Each public method returns a list of ``langchain_core.documents.Document``
objects preserving the original text and enriched metadata.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader

# Allowed URL schemes for load_url
_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> None:
    """Raise ``ValueError`` for URLs that could trigger SSRF.

    Checks:
    * Only ``http`` and ``https`` schemes are permitted.
    * The resolved host must not be a loopback, link-local, or private address.

    Args:
        url: The URL to validate.

    Raises:
        ValueError: If the URL is considered unsafe.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Only {_ALLOWED_SCHEMES} are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname.")

    try:
        addr = ipaddress.ip_address(socket.gethostbyname(hostname))
    except (socket.gaierror, ValueError) as exc:
        raise ValueError(f"Could not resolve hostname '{hostname}': {exc}") from exc

    if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_reserved:
        raise ValueError(
            f"Requests to '{hostname}' ({addr}) are not permitted "
            "(loopback / private / reserved address)."
        )


class PaperLoader:
    """Loads AI research papers from PDF files, URLs, or arXiv IDs."""

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_pdf(self, file_path: str) -> list[Document]:
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

    def load_arxiv(self, arxiv_id: str) -> list[Document]:
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

    def load_url(self, url: str) -> list[Document]:
        """Download a PDF from *url* and load it.

        Only public HTTP/HTTPS URLs are accepted.  Requests to loopback,
        private, link-local, or reserved addresses are blocked to prevent
        Server-Side Request Forgery (SSRF).

        Args:
            url: A direct HTTP/HTTPS link to a PDF file.

        Returns:
            List of ``Document`` objects.

        Raises:
            ValueError: If the URL scheme is not allowed or the host resolves
                to a private/internal address.
            requests.HTTPError: If the download fails.
        """
        _validate_url(url)
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

    def load_bytes(self, data: bytes, source_name: str = "upload") -> list[Document]:
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

