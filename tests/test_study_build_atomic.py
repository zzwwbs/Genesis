"""A refused build record leaves no implied study behind (2026-09-14 M6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.persistence import PersistenceCoordinator


def test_a_failed_build_record_rolls_back_the_implied_study(tmp_path: Path) -> None:
    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        record = {
            "build_hash": "b1",
            "study_id": "ghost-study",
            "package_version": 7,  # no such package version: the FK refuses it
            "compiler_version": "x",
            "created_at": "2026-09-14T00:00:00+00:00",
        }
        with pytest.raises(ValueError, match="FK_VIOLATION"):
            store.record_study_build(record)
        studies = store.connection.execute(
            "SELECT COUNT(*) FROM studies WHERE study_id = 'ghost-study'"
        ).fetchone()[0]
        assert studies == 0
        assert not store.connection.in_transaction
    finally:
        store.close()
