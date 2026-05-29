# ai-research-agent

A modular AI-powered research agent that can crawl, search, analyze, and synthesize information from papers and web sources.

## Components
- `main.py` - CLI entry point
- `api/agent.py` - Orchestrates the research workflow
- `processing/` - Crawl, search, analyze, optimize, and reporting helpers
- `ingestion/` - Paper ingestion and parsing
- `memory/` - Memory store abstraction
- `llm/` - LLM client integration
- `synthesis/` - Final synthesis step
- `tests/` - Unit tests for the pipeline

## Repository contents

Research agent package with a CLI, paper pipeline modules, and tests.

- `main.py` — command-line entry point for running research jobs
- `config.py` — research configuration dataclass and JSON loader
- `api/` — pipeline orchestration
- `ingestion/` — paper parsing and loading helpers
- `processing/` — crawl, load, search, analyze, optimize, and report steps
- `memory/` — file-backed memory store
- `llm/` — LLM client stub/integration layer
- `synthesis/` — final synthesis step
- `tests/` — unit tests
- `requirements.txt` — Python dependencies
- `pytest.ini` — pytest settings

## Maturity review

**Maturity:** Functional CLI-based research application with paper loading, search, analysis, reporting, memory persistence, and JSON output.

**What remains to make this a fuller product:**
- Replace the stub LLM client with a real provider-backed implementation.
- Add real crawlers/connectors for remote document sources.
- Add a user interface if non-CLI users should operate the agent.
- Add packaging and integration tests for the full flow.

## Usage

Run the CLI from the repository root:

```bash
python main.py --query "AI research" --source example.com --output output/result.json --memory-file state/memory.json
```

You can also ingest paper lines directly:

```bash
python main.py --query "AI research" --paper "Paper A|Abstract A|http://example.com|ml|ai"
```

Or supply a JSON config file:

```json
{
  "query": "AI research",
  "sources": ["example.com"],
  "memory_path": "state/memory.json",
  "output_path": "output/result.json"
}
```

```bash
python main.py --config config.json
```

## Development

Run the test suite with `pytest`.
