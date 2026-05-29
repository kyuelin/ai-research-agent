from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


class MemoryStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._items: List[Any] = []
        if self.path and self.path.exists():
            self._items = json.loads(self.path.read_text())

    def save(self, item):
        self._items.append(item)
        self._persist()

    def all(self):
        return list(self._items)

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, indent=2, default=str))
