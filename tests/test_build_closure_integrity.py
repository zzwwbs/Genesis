"""A build's copied package bytes are authenticated, not merely present.

The integrity manifest covered the build's own files only, so a tampered
`closure/study.yaml` verified clean and would be exported as the study that ran
(2026-09-14 M8).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from genesis.compiler import StudyCompiler

FIXTURE = Path(__file__).parent / "fixtures" / "specification"


def _build(tmp_path: Path) -> Path:
    build = StudyCompiler(FIXTURE).compile(tmp_path / "build")
    assert StudyCompiler.verify_build(build.path)
    return build.path


def test_a_tampered_closure_member_fails_verification(tmp_path: Path) -> None:
    root = _build(tmp_path)
    member = root / "closure" / "study.yaml"
    original = member.read_bytes()
    os.chmod(member, stat.S_IRUSR | stat.S_IWUSR)
    member.write_bytes(original + b"\n# tampered\n")
    with pytest.raises(ValueError, match="BUILD_INTEGRITY: closure/study.yaml"):
        StudyCompiler.verify_build(root)
    member.write_bytes(original)
    assert StudyCompiler.verify_build(root)


def test_an_unlisted_or_missing_closure_member_fails_verification(tmp_path: Path) -> None:
    root = _build(tmp_path)
    closure = root / "closure"
    os.chmod(closure, stat.S_IRWXU)
    (closure / "extra.yaml").write_text("x: 1\n")
    with pytest.raises(ValueError, match="unexpected closure member extra.yaml"):
        StudyCompiler.verify_build(root)
    (closure / "extra.yaml").unlink()
    (closure / "study.yaml").unlink()
    with pytest.raises(ValueError, match="closure/study.yaml missing"):
        StudyCompiler.verify_build(root)


def test_an_excluded_prompt_file_does_not_change_build_identity(tmp_path: Path) -> None:
    """source_hash read every prompt file, including ones the build excludes (M9)."""
    import shutil

    hashes = []
    for index, secret in enumerate(("sk-one", "sk-two")):
        package = tmp_path / f"pkg{index}"
        shutil.copytree(FIXTURE, package)
        (package / "prompts").mkdir(exist_ok=True)
        (package / "prompts" / "api_key.txt").write_text(secret)
        hashes.append(StudyCompiler(package).compile(tmp_path / f"b{index}").build_hash)
    assert hashes[0] == hashes[1]


def test_a_failed_build_leaves_no_temporary_directory_and_keeps_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup raised on nested closure directories, masking the cause (2026-09-14 L3)."""
    import genesis.compiler as compiler_module

    def refuse(_src: object, _dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(compiler_module.os, "replace", refuse)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(OSError, match="disk full"):
        StudyCompiler(FIXTURE).compile(out / "build")
    assert list(out.iterdir()) == []
