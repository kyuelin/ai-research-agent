from __future__ import annotations


def synthesize_findings(report, memory):
    memory.save(report)
    count = report.get("result_count", len(report.get("report", [])))
    return {
        "summary": f"Synthesis complete with {count} result(s).",
        "memory_saved": True,
        "highlights": report.get("highlights", []),
    }
