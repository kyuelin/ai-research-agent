from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping


@dataclass
class ResearchConfig:
    query: str
    sources: List[str] = field(default_factory=list)
    papers: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    memory_path: str | None = None
    output_path: str | None = None
    api_key: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchConfig":
        sources = data.get("sources", [])
        papers = data.get("papers", [])
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(papers, str):
            papers = [papers]
        return cls(
            query=str(data.get("query", "")).strip(),
            sources=[str(source).strip() for source in sources if str(source).strip()],
            papers=[str(paper).strip() for paper in papers if str(paper).strip()],
            options=dict(data.get("options", {})),
            memory_path=data.get("memory_path"),
            output_path=data.get("output_path"),
            api_key=data.get("api_key"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ResearchConfig":
        import json

        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("Config JSON must contain an object at the top level")
        return cls.from_mapping(payload)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
