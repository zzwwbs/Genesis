"""A patch that adds a declared field keeps it, through preview and approval.

The preview kept only keys the current document already had. A fresh study has
no `extensions`, and the default workflow writes its two required foundation
decisions under `/study/extensions/genesis.elicitation/...`, so both were dropped
while the stage recorded them as covered. The form rebuilt at approval also
omitted extensions, so they were lost a second time.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from genesis.compiler import StudyCompiler
from genesis.elicitation import apply_operations, sanitised_candidate
from genesis.service import GenesisService

FIXTURE = Path(__file__).parent / "fixtures" / "specification"
PATH = "/study/extensions/genesis.elicitation/simulation-boundary"


def test_a_patch_adding_extensions_to_a_fresh_study_keeps_them() -> None:
    current = {"study": {"schema_version": "1.0", "study_id": "s", "title": "t"}}
    patch = [SimpleNamespace(op="add", path=PATH, value="users on one platform")]
    candidate = sanitised_candidate(current, apply_operations(current, patch))
    assert candidate["study"]["extensions"] == {
        "genesis.elicitation": {"simulation-boundary": "users on one platform"}
    }


def test_fields_no_canonical_model_declares_are_still_dropped() -> None:
    current = {"study": {"schema_version": "1.0", "study_id": "s", "title": "t"}}
    patch = [SimpleNamespace(op="add", path="/study/not_a_field", value=1)]
    assert (
        "not_a_field" not in sanitised_candidate(current, apply_operations(current, patch))["study"]
    )


def test_the_form_rebuilt_from_a_package_keeps_its_extensions(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    shutil.copytree(FIXTURE, package)
    study = yaml.safe_load((package / "study.yaml").read_text())
    study["extensions"] = {"genesis.elicitation": {"comparison-objective": "with vs without"}}
    (package / "study.yaml").write_text(yaml.safe_dump(study))
    form = GenesisService._form_from_package(StudyCompiler(package)._load(), package, "s")
    assert form["extensions"] == study["extensions"]
    canonical = GenesisService._canonical_specification(form)
    assert canonical["study"]["extensions"] == study["extensions"]
