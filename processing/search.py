from __future__ import annotations


def _flatten(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten(item)
    elif value is not None:
        yield str(value)


def search_documents(query, documents):
    needle = query.strip().lower()
    if not needle:
        return list(documents)
    results = []
    for doc in documents:
        haystack = " ".join(_flatten(doc)).lower()
        if needle in haystack:
            results.append(doc)
    return results
