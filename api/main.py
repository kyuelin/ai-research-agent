"""API module: FastAPI application exposing ingestion and synthesis endpoints.

Endpoints
---------
GET  /health              — liveness probe
POST /ingest/pdf          — ingest a PDF uploaded as multipart form-data
POST /ingest/arxiv        — ingest a paper from arXiv by ID
POST /ingest/url          — ingest a PDF from a direct URL
POST /synthesize          — run the multi-agent synthesis workflow
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config import settings
from ingestion.paper_loader import PaperLoader
from memory.vector_store import get_default_store
from processing.chunker import DocumentChunker
from synthesis.graph import get_compiled_graph

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Agent",
    description=(
        "Multi-agent system that converts AI research papers into "
        "engineered systems via LangGraph, ChromaDB, and Ollama."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Module-level helpers (one instance per worker process)
# ---------------------------------------------------------------------------
_loader = PaperLoader()
_chunker = DocumentChunker()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class IngestArxivRequest(BaseModel):
    arxiv_id: str = Field(..., description="arXiv paper ID, e.g. '2310.06825'")


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="Direct HTTP/HTTPS URL pointing to a PDF file")


class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    source: str


class SynthesizeRequest(BaseModel):
    query: str = Field(..., description="Research question or topic to synthesize")
    max_results: int = Field(default=5, ge=1, le=20)


class SynthesizeResponse(BaseModel):
    query: str
    analysis: str
    synthesis: str
    implementation_plan: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return service health and version information."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/ingest/pdf", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a PDF research paper.

    The uploaded file is parsed, chunked, embedded via Ollama, and stored in
    the ChromaDB vector store.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    data = await file.read()
    try:
        docs = _loader.load_bytes(data, source_name=file.filename)
        chunks = _chunker.chunk(docs)
        get_default_store().add_documents(chunks)
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        message="PDF ingested successfully.",
        chunks_added=len(chunks),
        source=file.filename,
    )


@app.post("/ingest/arxiv", response_model=IngestResponse, tags=["Ingestion"])
def ingest_arxiv(request: IngestArxivRequest) -> IngestResponse:
    """Fetch and ingest a research paper from arXiv by its identifier."""
    try:
        docs = _loader.load_arxiv(request.arxiv_id)
        chunks = _chunker.chunk(docs)
        get_default_store().add_documents(chunks)
    except Exception as exc:
        logger.exception("arXiv ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        message="arXiv paper ingested successfully.",
        chunks_added=len(chunks),
        source=f"arxiv:{request.arxiv_id}",
    )


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
def ingest_url(request: IngestUrlRequest) -> IngestResponse:
    """Download and ingest a research paper from a direct PDF URL."""
    try:
        docs = _loader.load_url(request.url)
        chunks = _chunker.chunk(docs)
        get_default_store().add_documents(chunks)
    except Exception as exc:
        logger.exception("URL ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        message="URL paper ingested successfully.",
        chunks_added=len(chunks),
        source=request.url,
    )


@app.post("/synthesize", response_model=SynthesizeResponse, tags=["Synthesis"])
def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """Run the multi-agent LangGraph synthesis workflow for a research query.

    The workflow executes an 11-node pipeline with a conditional feedback loop:

    1. **retrieve_context** — fetches relevant chunks from ChromaDB.
    2. **normalize** — enforces consistent dated metadata on each chunk.
    3. **score_chunks** — ranks chunks by novelty, practicality, adoption, and relevance.
    4. **cluster_chunks** — groups related chunks into thematic clusters.
    5. **analyze_papers** — extracts key concepts via the LLM.
    6. **synthesize_findings** — produces a cross-paper narrative.
    7. **generate_implementation** — outputs an engineering plan.
    8. **generate_prompts** — creates code-generation prompts from the plan.
    9. **track_artifacts** — records paper-to-solution lineage.
    10. **create_digest** — produces a concise weekly research digest.
    11. **apply_feedback** — evaluates digest quality and adjusts scoring weights;
        loops back to *score_chunks* when quality < 0.8 and iteration < 2.
    """
    try:
        graph = get_compiled_graph()
        result: dict[str, Any] = graph.invoke(
            {"query": request.query, "max_results": request.max_results}
        )
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SynthesizeResponse(
        query=request.query,
        analysis=result.get("analysis", ""),
        synthesis=result.get("synthesis", ""),
        implementation_plan=result.get("implementation_plan", ""),
    )
