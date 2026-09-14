"""Per-run reads of events and artifacts use an index (2026-09-14 M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.persistence import PersistenceCoordinator


@pytest.mark.parametrize("table", ["events", "artifacts"])
def test_a_run_filter_searches_an_index(tmp_path: Path, table: str) -> None:
    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        plan = store.connection.execute(
            f"EXPLAIN QUERY PLAN SELECT * FROM {table} WHERE run_id = ?", ("r",)
        ).fetchall()
        detail = " ".join(str(row[-1]) for row in plan)
        assert "USING INDEX" in detail and f"SCAN {table}" not in detail, detail
    finally:
        store.close()
