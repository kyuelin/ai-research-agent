from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from config import ResearchConfig
from ingestion.paper_loader import PaperLoader
from llm.openai_client import OpenAIClient
from memory.memory_store import MemoryStore
from processing.analyze import ResearchAnalyzer
from processing.crawler import PaperCrawler
from processing.optimize import ResultOptimizer
from processing.plan import ResearchPlanner
from processing.report import ReportGenerator
from processing.search import PaperSearcher
from synthesis.synthesize import SynthesisEngine


def _normalize_config(config: ResearchConfig | Dict[str, Any]) -> ResearchConfig:
    if isinstance(config, ResearchConfig):
        return config
    return ResearchConfig.from_mapping(config)


@dataclass
class ResearchAgent:
    config: ResearchConfig | Dict[str, Any]
    client: Any = field(default_factory=OpenAIClient)
    memory: MemoryStore | None = None

    def __post_init__(self) -> None:
        self.config = _normalize_config(self.config)
        if self.client is None:
            self.client = OpenAIClient(api_key=self.config.api_key)
        if self.memory is None:
            self.memory = MemoryStore(path=self.config.memory_path)

    def run(self) -> Dict[str, Any]:
        sources: List[str] = self.config.sources
        query: str = self.config.query

        planner = ResearchPlanner()
        crawler = PaperCrawler()
        loader = PaperLoader()
        searcher = PaperSearcher()
        analyzer = ResearchAnalyzer()
        optimizer = ResultOptimizer()
        reporter = ReportGenerator()
        synthesizer = SynthesisEngine()

        plan = planner.build_plan(query, sources)
        crawled_sources = crawler.crawl(sources)
        papers = []
        for source in crawled_sources:
            papers.extend(crawler.fetch_papers(source))
        loaded = papers or loader.load_from_lines(self.config.papers)
        searched = searcher.search(query, loaded)
        analyzed = analyzer.analyze(searched, self.client)
        optimized = optimizer.optimize(analyzed)
        report = reporter.build(query, optimized)
        synthesis = synthesizer.synthesize(report, self.memory)
        result = {
            "config": self.config.as_dict(),
            "plan": plan,
            "crawled": [source.__dict__ for source in crawled_sources],
            "loaded": [paper.__dict__ for paper in loaded],
            "searched": [paper.__dict__ for paper in searched],
            "analyzed": analyzed,
            "optimized": optimized,
            "report": report,
            "synthesis": synthesis,
        }
        if self.config.output_path:
            output = Path(self.config.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, default=str))
        return result
