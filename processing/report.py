from __future__ import annotations


def generate_report(results):
    highlights = []
    for result in results[:3]:
        if isinstance(result, dict) and result.get("summary"):
            highlights.append(result["summary"])
        else:
            highlights.append(str(result))
    return {
        "report": results,
        "status": "generated",
        "result_count": len(results),
        "highlights": highlights,
    }
