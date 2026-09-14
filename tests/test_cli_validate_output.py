"""`genesis validate --output DIR` must never delete a directory it did not create.

The cleanup meant for the command's own scratch build ran on whatever --output
named, so an existing directory was wiped -- even when compile refused to write
into it, and when validation failed first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from genesis.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "specification"


def test_an_existing_output_directory_survives_a_refused_compile(tmp_path: Path) -> None:
    existing = tmp_path / "notes"
    existing.mkdir()
    (existing / "notes.txt").write_text("keep me")
    with pytest.raises(Exception, match="BUILD_EXISTS"):
        main(["validate", str(FIXTURE), "--output", str(existing)])
    assert (existing / "notes.txt").read_text() == "keep me"


def test_an_existing_output_directory_survives_a_failed_validation(tmp_path: Path) -> None:
    broken = tmp_path / "pkg"
    shutil.copytree(FIXTURE, broken)
    (broken / "study.yaml").write_text("not: [valid")
    existing = tmp_path / "notes"
    existing.mkdir()
    (existing / "notes.txt").write_text("keep me")
    with pytest.raises(ValueError):
        main(["validate", str(broken), "--output", str(existing)])
    assert (existing / "notes.txt").read_text() == "keep me"


def test_validation_leaves_no_build_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = tmp_path / "check"
    main(["validate", str(FIXTURE), "--output", str(fresh)])
    assert '"valid": true' in capsys.readouterr().out
    assert not fresh.exists()
    source = tmp_path / "pkg"
    shutil.copytree(FIXTURE, source)
    main(["validate", str(source)])
    assert not (source / ".genesis-build-check").exists()
