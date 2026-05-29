from pathlib import Path

from memory.memory_store import MemoryStore


def test_memory_store_collects_items(tmp_path: Path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path=path)
    store.save({"a": 1})
    assert store.all()[0]["a"] == 1
    reloaded = MemoryStore(path=path)
    assert reloaded.all()[0]["a"] == 1
