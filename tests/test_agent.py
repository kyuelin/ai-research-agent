from pathlib import Path

from api.agent import ResearchAgent


def test_agent_run_returns_pipeline_outputs(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    output_file = tmp_path / "result.json"
    agent = ResearchAgent({"query": "AI research", "sources": ["example.com"], "memory_path": str(memory_file), "output_path": str(output_file)})
    result = agent.run()
    assert "plan" in result
    assert "report" in result
    assert result["synthesis"]["memory_saved"] is True
    assert memory_file.exists()
    assert output_file.exists()
