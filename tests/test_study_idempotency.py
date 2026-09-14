"""An idempotency key names one request, not whatever was sent first (2026-09-14 M7/L1).

POST /studies returned the cached result for any replay of a key, so reusing it
with a different body silently returned the first study and never created the
second; and every key was kept forever.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.persistence import PersistenceCoordinator


def test_a_key_reused_with_a_different_body_is_a_conflict(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    headers = {"Idempotency-Key": "k1"}
    first = client.post("/studies", json={"id": "study-a", "title": "A"}, headers=headers)
    again = client.post("/studies", json={"id": "study-a", "title": "A"}, headers=headers)
    assert first.status_code == 201 and again.json() == first.json()
    other = client.post("/studies", json={"id": "study-b", "title": "B"}, headers=headers)
    assert other.status_code == 409
    assert other.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_the_idempotency_table_is_bounded(tmp_path: Path) -> None:
    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.IDEMPOTENCY_LIMIT = 3  # type: ignore[misc]
        for index in range(5):
            store.record_idempotency(f"k{index}", {"i": index}, "h")
        count = store.connection.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0]
        assert count == 3
        assert store.get_idempotency("k4", "h") == {"i": 4}
        assert store.get_idempotency("k0", "h") is None
    finally:
        store.close()
