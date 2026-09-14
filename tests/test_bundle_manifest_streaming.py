"""Bundle members are hashed in chunks, never read whole (2026-09-14 M18)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from genesis import evidence


def test_member_digests_are_streamed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    member = tmp_path / "states.parquet"
    member.write_bytes(b"x" * (3 << 20))

    def refuse(_self: Path) -> bytes:
        raise AssertionError("a bundle member was read whole")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    manifest = evidence.write_bundle_manifest(
        tmp_path,
        export_mode="full",
        run_id="r",
        source_run_id="r",
        local_import_id=None,
        package_digest="p",
        build_digest="b",
        scientific_config_digest="s",
        capabilities=[],
        omissions=[],
        retention_policy="retain_raw_responses",
    )
    monkeypatch.undo()
    assert hashlib.sha256(b"x" * (3 << 20)).hexdigest() in manifest.read_text()
