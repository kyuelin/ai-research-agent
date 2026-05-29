# AI Research Agent

Status: actively maintained.
A production-ready multi-agent system that converts AI research papers into engineered systems via LangGraph, FastAPI, ChromaDB, and Ollama.

## Repository contents

Python agent project for ingestion, memory, processing, and synthesis workflows.

- `api/` — API layer and interfaces
- `ingestion/` — document intake pipeline
- `llm/` — LLM integration helpers
- `memory/` — state and retrieval memory
- `processing/` — core processing pipeline
- `synthesis/` — result synthesis logic
- `tests/` — test suite
- `config.py` — runtime configuration
- `requirements.txt` — Python dependencies
- `pytest.ini` — pytest settings
## Architecture

```
ingestion/          PDF / arXiv / URL paper loading (SSRF-protected)
processing/         Text chunking + Ollama embeddings
memory/             ChromaDB-backed vector store (singleton)
llm/                Centralised OllamaClient singleton (chat + embeddings)
synthesis/          LangGraph 11-node agent pipeline with feedback loop
api/                FastAPI HTTP endpoints
```

### Synthesis Pipeline

```
START → retrieve_context → normalize → score_chunks ←──────────────────────┐
  → cluster_chunks → analyze_papers → synthesize_findings                   │
  → generate_implementation → generate_prompts → track_artifacts            │
  → create_digest → apply_feedback ──(quality ≥ 0.8 or iter ≥ 2)──→ END   │
                              └──(quality < 0.8 and iter < 2)───────────────┘
```

| Node | Input | Output |
|---|---|---|
| `retrieve_context` | `query`, `max_results` | `retrieved_chunks` |
| `normalize` | `retrieved_chunks` | `normalized_chunks` |
| `score_chunks` | `query`, `normalized_chunks`, `feedback` | `scores` |
| `cluster_chunks` | `scores` | `clusters` |
| `analyze_papers` | `query`, `clusters` \| `scores` \| `retrieved_chunks` | `analysis` |
| `synthesize_findings` | `query`, `analysis` | `synthesis` |
| `generate_implementation` | `query`, `synthesis` | `implementation_plan` |
| `generate_prompts` | `implementation_plan` | `code_prompts` |
| `track_artifacts` | `scores`, `implementation_plan`, `code_prompts` | `artifacts` |
| `create_digest` | `query`, `synthesis`, `implementation_plan` | `digest` |
| `apply_feedback` | `digest`, `iteration` | `feedback`, `iteration` |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` and `llama3` pulled
- ChromaDB (embedded — no server required by default)

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd ai-research-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Copy and edit .env
cp .env.example .env

# 4. Pull required Ollama models
ollama pull nomic-embed-text
ollama pull llama3
```

## Running the API

```bash
uvicorn api.main:app --reload
```

Interactive docs: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/ingest/pdf` | Upload a PDF (multipart form) |
| `POST` | `/ingest/arxiv` | Ingest paper by arXiv ID |
| `POST` | `/ingest/url` | Ingest PDF from a direct URL |
| `POST` | `/synthesize` | Run multi-agent synthesis |

### Example: Ingest an arXiv paper

```bash
curl -X POST http://localhost:8000/ingest/arxiv \
     -H "Content-Type: application/json" \
     -d '{"arxiv_id": "2310.06825"}'
```

### Example: Run synthesis

```bash
curl -X POST http://localhost:8000/synthesize \
     -H "Content-Type: application/json" \
     -d '{"query": "How do transformer attention mechanisms work?"}'
```

## Configuration

All settings are read from environment variables or a `.env` file (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_LLM_MODEL` | `llama3` | LLM for synthesis |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `CHROMA_COLLECTION` | `research_papers` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_RETRIEVAL_RESULTS` | `5` | Top-k retrieval results |

## Testing

```bash
pytest tests/ -v
```

All external services (Ollama, ChromaDB) are mocked in the test suite.

## Maturity review

**Maturity:** Functional research pipeline with useful building blocks, but still missing product packaging.

**What remains to make this a functional application:**
- Add a user-facing interface or orchestration layer.
- Define deployment, secrets, and provider configuration clearly.
- Strengthen evals, observability, and end-to-end integration tests.
