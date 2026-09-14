"""Garbage collection never deletes a writer's temporary file (2026-09-14 L2)."""

from __future__ import annotations

from pathlib import Path

from genesis.persistence import ObjectStore


def test_collect_garbage_spares_in_flight_temporaries(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    ref = store.put(b"kept")
    shard = ref.path.parent if ref.path else tmp_path / "objects" / ref.digest[:2]
    temporary = shard / ".object-abc123"
    temporary.write_bytes(b"being written")
    orphan = store.put(b"unreferenced")
    assert store.collect_garbage({ref.digest}) == 1
    assert temporary.is_file()
    assert not (tmp_path / "objects" / orphan.digest[:2] / orphan.digest[2:]).exists()
