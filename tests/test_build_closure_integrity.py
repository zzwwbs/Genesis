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


def test_a_symlinked_asset_is_refused_not_followed(tmp_path: Path) -> None:
    """A link carried a credential's bytes under a dataset's name (2026-09-26 R1').

    The exclusion judged the path as the package wrote it, so ``data/extra.csv``
    was an ordinary data asset while ``keys.env`` was a credential -- and the
    closure read the link, pinning the secret as the study's data as both
    ``data/extra.csv`` and ``closure/data/extra.csv``. Every sibling import path
    refuses links; the closure now does too, whatever the link points at.
    """
    import shutil

    secret = b"API_KEY=sk-live-never-carry-me"
    package = tmp_path / "pkg"
    shutil.copytree(FIXTURE, package)
    (package / "keys.env").write_bytes(secret)
    (package / "data").mkdir(exist_ok=True)
    os.symlink("../keys.env", package / "data" / "extra.csv")

    with pytest.raises(ValueError, match="symbolic link"):
        StudyCompiler(package).compile(tmp_path / "build")


def test_a_symlink_to_an_ordinary_data_file_is_refused_too(tmp_path: Path) -> None:
    """The rule is "regular files only", not "no links to excluded files".

    A target-name check would still carry a key hidden in an innocuously named
    file, and a symlinked directory would reach assets the walk never collects.
    """
    import shutil

    package = tmp_path / "pkg"
    shutil.copytree(FIXTURE, package)
    (package / "data").mkdir(exist_ok=True)
    (package / "data" / "population.csv").write_text("id,n\na,1\n")
    os.symlink("population.csv", package / "data" / "alias.csv")

    with pytest.raises(ValueError, match="symbolic link"):
        StudyCompiler(package).compile(tmp_path / "build")


def test_a_symlinked_directory_in_the_asset_tree_is_refused(tmp_path: Path) -> None:
    """It contributed nothing and compiled into a build no run could use (I4)."""
    import shutil

    package = tmp_path / "pkg"
    shutil.copytree(FIXTURE, package)
    (package / "elsewhere").mkdir()
    (package / "elsewhere" / "population.csv").write_text("id,n\na,1\n")
    os.symlink("elsewhere", package / "data")

    with pytest.raises(ValueError, match="symbolic link"):
        StudyCompiler(package).compile(tmp_path / "build")


def test_a_broken_symlink_in_the_asset_tree_is_refused(tmp_path: Path) -> None:
    """A dangling link is not a regular file either, and silently contributes nothing."""
    import shutil

    package = tmp_path / "pkg"
    shutil.copytree(FIXTURE, package)
    os.symlink("nowhere", package / "data")

    with pytest.raises(ValueError, match="symbolic link"):
        StudyCompiler(package).compile(tmp_path / "build")
