"""The four remaining low-severity findings of the 2026-09-26 review.

1. The versioned open path validated a `user_version` read taken before any lock.
2. An imported dictionary entry with no table was served as an empty table.
3. `analysis_build` on an imported run skipped the imported branch, answering
   with an empty dataset set.
4. The export strip tested only the value's type, so a record's own mapping
   `value` was dropped from an exploded dataset's table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genesis.service import GenesisService


def test_a_versioned_workspace_still_opens_and_validates(tmp_path: Path) -> None:
    """The validation now runs under BEGIN IMMEDIATE; the ordinary path must hold."""
    workspace = tmp_path / "ws"
    GenesisService(workspace).close()
    service = GenesisService(workspace)
    try:
        version = service.persistence.connection.execute("PRAGMA user_version").fetchone()[0]
        assert version > 0
    finally:
        service.close()


def test_an_imported_dictionary_entry_without_a_table_is_refused(tmp_path: Path) -> None:
    """A listed dataset with no file read as a dataset that had no rows."""
    service = GenesisService(tmp_path / "ws")
    try:
        bundle = tmp_path / "bundle"
        (bundle / "datasets").mkdir(parents=True)
        (bundle / "datasets" / "dictionary.json").write_text(json.dumps({"d": {"columns": ["x"]}}))
        with pytest.raises(ValueError, match="carries no table"):
            service._import_dataset_tables(bundle)
    finally:
        service.close()


def test_an_analysis_build_does_not_apply_to_an_imported_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The imported branch was skipped, so the answer was an empty dataset set."""
    service = GenesisService(tmp_path / "ws")
    try:
        monkeypatch.setattr(
            service,
            "get_run",
            lambda run_id: {"imported_datasets": {"tables": {}, "dictionary": "digest"}},
        )
        with pytest.raises(ValueError, match="analysis build does not apply"):
            service.run_datasets("r", analysis_build="builds/b")
    finally:
        service.close()


def test_the_payload_copy_of_value_is_stripped_and_a_records_own_is_kept(tmp_path: Path) -> None:
    """The strip tested only the value's type, so a record's own mapping `value`
    was dropped for every exploded dataset."""
    service = GenesisService(tmp_path / "ws")
    try:
        build = tmp_path / "build"
        build.mkdir()
        (build / "outcome_plan.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "datasets": [
                        {"id": "d", "source": {"kind": "artifacts", "artifact_type": "t"}}
                    ],
                    "outcomes": [],
                }
            )
        )
        # The artifact's payload copy: its fields are spread into the row, so the
        # nested copy repeats them as one JSON cell.
        copy_row = {"artifact_id": "a", "score": 1, "value": {"score": 1}}
        # A record's own value: not a copy of the row's fields.
        own_row = {"artifact_id": "a", "label": "x", "value": {"note": "kept"}}
        tables, _dictionary = GenesisService._dataset_tables({"d": [copy_row, own_row]}, build)
        rows = tables["d"]
        # Columns are aligned across the dataset, so a stripped copy shows as an
        # empty cell rather than a missing key, and a kept mapping is serialized
        # to a single JSON cell.
        assert rows[0]["value"] is None
        assert json.loads(rows[1]["value"]) == {"note": "kept"}
    finally:
        service.close()
