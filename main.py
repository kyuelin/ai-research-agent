from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from api.agent import ResearchAgent
from config import ResearchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI research agent")
    parser.add_argument("--config", help="Path to a JSON config file")
    parser.add_argument("--query", help="Research query")
    parser.add_argument("--source", action="append", dest="sources", default=[], help="Source to crawl (repeatable)")
    parser.add_argument("--sources-file", help="File with one source per line")
    parser.add_argument("--paper", action="append", dest="papers", default=[], help="Paper line to ingest (repeatable)")
    parser.add_argument("--papers-file", help="File with one paper per line in title|abstract|url|keyword format")
    parser.add_argument("--memory-file", help="Path to persist memory between runs")
    parser.add_argument("--output", help="Write the full result JSON to this path")
    parser.add_argument("--api-key", help="API key for the LLM client")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> ResearchConfig:
    payload: Dict[str, Any] = {}
    if args.config:
        payload.update(json.loads(Path(args.config).read_text()))
    if args.query:
        payload["query"] = args.query
    if args.sources:
        payload["sources"] = args.sources
    if args.sources_file:
        sources = [line.strip() for line in Path(args.sources_file).read_text().splitlines() if line.strip()]
        payload["sources"] = payload.get("sources", []) + sources
    if args.papers:
        payload["papers"] = args.papers
    if args.papers_file:
        papers = [line.strip() for line in Path(args.papers_file).read_text().splitlines() if line.strip()]
        payload["papers"] = payload.get("papers", []) + papers
    if args.memory_file:
        payload["memory_path"] = args.memory_file
    if args.output:
        payload["output_path"] = args.output
    if args.api_key:
        payload["api_key"] = args.api_key
    return ResearchConfig.from_mapping(payload)


def main() -> int:
    args = parse_args()
    config = load_config(args)
    if not config.query:
        raise SystemExit("A research query is required. Use --query or --config.")
    result = ResearchAgent(config).run()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
