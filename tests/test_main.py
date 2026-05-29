from pathlib import Path
from types import SimpleNamespace

from main import load_config


def test_load_config_merges_cli_inputs(tmp_path: Path):
    sources_file = tmp_path / "sources.txt"
    sources_file.write_text("example.org
")
    papers_file = tmp_path / "papers.txt"
    papers_file.write_text("Paper B|Abstract B|http://example.org|ml
")
    args = SimpleNamespace(
        config=None,
        query="AI research",
        sources=["example.com"],
        sources_file=str(sources_file),
        papers=["Paper A|Abstract A"],
        papers_file=str(papers_file),
        memory_file=str(tmp_path / "memory.json"),
        output=str(tmp_path / "result.json"),
        api_key="secret",
    )
    config = load_config(args)
    assert config.query == "AI research"
    assert config.sources == ["example.com", "example.org"]
    assert config.papers == ["Paper A|Abstract A", "Paper B|Abstract B|http://example.org|ml"]
    assert config.api_key == "secret"
