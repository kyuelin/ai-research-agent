from pathlib import Path

from config import ResearchConfig


def test_research_config_from_json(tmp_path: Path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"query": "AI", "sources": ["example.com"], "papers": ["Paper A|Abstract"], "memory_path": "memory.json"}')
    config = ResearchConfig.from_json(config_file)
    assert config.query == "AI"
    assert config.sources == ["example.com"]
    assert config.papers == ["Paper A|Abstract"]
    assert config.memory_path == "memory.json"
