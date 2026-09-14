"""Importing a package never follows a link out of it (2026-09-14 M2b).

Every other import path refused links; import_package copied through them, so a
member linked to a file outside the package brought that file's content into
the specification, every build of it, and every export.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from genesis.service import GenesisService

FIXTURE = Path(__file__).parent / "fixtures" / "specification"


def _package(workspace: Path) -> Path:
    source = workspace / "imports" / "pkg"
    shutil.copytree(FIXTURE, source)
    return source


def test_a_linked_prompt_is_refused(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        source = _package(service.workspace)
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP-SECRET-OUTSIDE")
        (source / "prompts").mkdir(exist_ok=True)
        (source / "prompts" / "leak.txt").symlink_to(secret)
        with pytest.raises(ValueError, match="IMPORT_LINK: package member 'prompts/leak.txt'"):
            service.import_package(source, specification_id="linked")
        assert not service._specification_dir("linked").exists()
    finally:
        service.close()


def test_an_integrity_entry_outside_the_package_is_refused(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        source = _package(service.workspace)
        (tmp_path / "outside.txt").write_text("x")
        (source / "integrity.json").write_text(json.dumps({"../../../outside.txt": "0" * 64}))
        with pytest.raises(ValueError, match="not a file inside the package"):
            service.import_package(source, specification_id="escaped")
    finally:
        service.close()
