"""EXM — immutable package closure and effective execution identity (G4/G1 foundation).

Covers content-addressed package closure with original bytes, canonical
hashing that is stable across relocation, path/symlink/credential rejection,
and an effective execution manifest with a run-id-independent scientific
configuration digest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from genesis.execution_manifest import (
    build_package_closure,
    resolve_execution_manifest,
    scientific_config_digest,
)
from genesis.service import GenesisService

SCHEMA = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}


def _write_package(root: Path, study_id: str = "closure-study") -> dict[str, str]:
    """Write a study package with nested prompts, schemas and data assets.

    Returns the normalized package-relative path to original byte content.
    """
    spec = root / ".genesis" / "specifications" / study_id
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "study.yaml").write_text(
        f"schema_version: '1.0'\nstudy_id: {study_id}\ntitle: closure study\n"
    )
    (spec / "openness.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: '" + study_id + "'\nprocesses: []\n"
    )
    (spec / "theory.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: '" + study_id + "'\ntheory_family: exploratory\n"
    )
    (spec / "domain.yaml").write_text("schema_version: '1.0'\nstudy_id: '" + study_id + "'\n")
    (spec / "protocol.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: '"
        + study_id
        + "'\ntime_model: {type: rounds, end: 1}\nconditions: [{id: base}]\n"
    )
    (spec / "outcomes.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: '" + study_id + "'\noutcomes: []\n"
    )
    (spec / "models.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: '" + study_id + "'\nmodels: []\n"
    )
    (spec / "processes.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: '" + study_id + "'\nprocesses:\n  - id: tick\n"
        "    executor: {mode: deterministic}\n    context_policy: public\n"
    )
    (spec / "metadata.json").write_text(json.dumps({"version": 1, "study_id": study_id}))
    prompts = spec / "prompts"
    prompts.mkdir()
    (prompts / "compose.txt").write_text("Compose from {context}")
    schemas = spec / "schemas"
    schemas.mkdir()
    (schemas / "output.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    data = spec / "data"
    data.mkdir()
    (data / "population.csv").write_text("id,value\n1,42\n")
    nested = data / "nested"
    nested.mkdir()
    (nested / "matrix.json").write_text("[1,2,3]")
    return {
        str(p.relative_to(spec).as_posix()): p.read_text() for p in spec.rglob("*") if p.is_file()
    }


def test_package_closure_pins_nested_assets() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_package(root)
        spec_dir = root / ".genesis" / "specifications" / "closure-study"
        closure = build_package_closure(spec_dir)

        paths = {asset["path"] for asset in closure.manifest["assets"]}
        for expected in (
            "study.yaml",
            "openness.yaml",
            "theory.yaml",
            "domain.yaml",
            "protocol.yaml",
            "outcomes.yaml",
            "models.yaml",
            "processes.yaml",
            "metadata.json",
            "prompts/compose.txt",
            "schemas/output.yaml",
            "data/population.csv",
            "data/nested/matrix.json",
        ):
            assert expected in paths, f"missing {expected}"
        by_path = {asset["path"]: asset for asset in closure.manifest["assets"]}
        assert by_path["prompts/compose.txt"]["size"] == len("Compose from {context}")
        assert by_path["data/population.csv"]["media_type"] == "text/csv"
        assert by_path["schemas/output.yaml"]["media_type"] == "text/yaml"
        # Digests are sha256 of the original bytes, recorded per member.
        import hashlib

        assert (
            by_path["data/population.csv"]["digest"]
            == hashlib.sha256(b"id,value\n1,42\n").hexdigest()
        )


def test_closure_digest_is_stable_across_relocation(tmp_path: Path) -> None:
    original = tmp_path / "a" / ".genesis" / "specifications" / "closure-study"
    moved = tmp_path / "b" / "moved-workspace" / ".genesis" / "specifications" / "closure-study"
    _write_package(tmp_path / "a")
    moved.parents[2].mkdir(parents=True, exist_ok=True)
    shutil.copytree(original, moved)

    first = build_package_closure(original)
    second = build_package_closure(moved)
    assert first.digest == second.digest
    assert first.manifest == second.manifest


def test_closure_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    spec_dir = tmp_path / ".genesis" / "specifications" / "escape-study"
    spec_dir.mkdir(parents=True)
    (spec_dir / "study.yaml").write_text("schema_version: '1.0'\nstudy_id: escape-study\n")
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("do not inline")
    # A symlink inside the package pointing outside the package.
    try:
        (spec_dir / "leak.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(ValueError, match="escape|outside|symlink"):
        build_package_closure(spec_dir)


def test_closure_excludes_credential_files(tmp_path: Path) -> None:
    spec_dir = tmp_path / ".genesis" / "specifications" / "secrets-study"
    spec_dir.mkdir(parents=True)
    (spec_dir / "study.yaml").write_text("schema_version: '1.0'\nstudy_id: secrets-study\n")
    (spec_dir / ".credentials.yaml").write_text("api_key: super-secret\n")
    (spec_dir / "creds.json").write_text('{"key": "secret"}')

    closure = build_package_closure(spec_dir)
    paths = {asset["path"] for asset in closure.manifest["assets"]}
    assert ".credentials.yaml" not in paths
    assert "creds.json" not in paths
    assert "study.yaml" in paths


def test_run_identity_pins_nested_package_assets(tmp_path: Path) -> None:
    """G4 foundation: the executed run pins the closure of the approved package."""
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "closure-approve",
                "title": "closure approve",
                "processes": [
                    {
                        "id": "tick",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {},
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [],
                "models": [],
            }
        )
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "closure-approve" / "schemas"
        )
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "nested-schema.yaml").write_text(
            "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
        )
        revised = service.update_specification(
            "closure-approve", {"description": "with nested schema"}, draft["version"]
        )
        service.approve_specification("closure-approve", revised["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/closure-approve", specification_id="closure-approve"
        )
        service.create_run(
            {"id": "closure-run", "study_id": "closure-approve", "build": compiled["path"]}
        )
        service.execute_run("closure-run")
        manifest = service.get_run("closure-run")["manifest"]
        assert manifest.get("package_closure_digest")
        # The live package edited after execution must not change the pinned digest.
        current = service.get_specification("closure-approve")
        service.update_specification(
            "closure-approve", {"description": "edited later"}, current["version"]
        )
        assert (
            service.get_run("closure-run")["manifest"]["package_closure_digest"]
            == manifest["package_closure_digest"]
        )
    finally:
        service.close()


def test_execution_manifest_carries_effective_configuration(tmp_path: Path) -> None:
    spec_dir = tmp_path / ".genesis" / "specifications" / "manifest-study"
    spec_dir.mkdir(parents=True)
    _write_package(tmp_path, study_id="manifest-study")
    closure = build_package_closure(spec_dir)
    build_manifest = {
        "build_hash": "b-1",
        "study_id": "manifest-study",
        "compiler_version": "1.0",
    }
    protocol = {
        "conditions": [{"id": "base"}, {"id": "strict", "factors": {"policy": "strict"}}],
        "replications": 3,
        "random_streams": [{"id": "conventional", "seed": 7}],
    }
    manifest = resolve_execution_manifest(
        condition_id="strict",
        factors={"policy": "strict"},
        replication=3,
        build_manifest=build_manifest,
        protocol=protocol,
        package_closure_digest=closure.digest,
        protocol_digest="p-1",
        model_configuration_digest="m-1",
        outcome_plan_digest="o-1",
        origin_experiment_id="experiment-1",
    )
    assert manifest["manifest_version"] == 1
    assert manifest["package_digest"] == closure.digest
    assert manifest["build_digest"] == "b-1"
    assert manifest["protocol_digest"] == "p-1"
    assert manifest["condition"] == {"id": "strict", "factors": {"policy": "strict"}}
    assert manifest["replication"] == 3
    assert manifest["origin_experiment_id"] == "experiment-1"
    assert manifest["randomness"]["algorithm_version"] == "genesis-rng-v1"
    assert manifest["randomness"]["stream_scheme_version"] == 1
    assert manifest["runtime_contract_version"] == 1


def test_scientific_config_digest_is_run_id_independent(tmp_path: Path) -> None:
    spec_dir = tmp_path / ".genesis" / "specifications" / "digest-study"
    _write_package(tmp_path, study_id="digest-study")
    closure = build_package_closure(spec_dir)
    base = dict(
        condition_id="base",
        factors={},
        replication=1,
        build_manifest={"build_hash": "b", "study_id": "digest-study"},
        protocol={"conditions": [{"id": "base"}], "replications": 1},
        package_closure_digest=closure.digest,
        protocol_digest="p",
        model_configuration_digest="m",
        outcome_plan_digest="o",
        origin_experiment_id=None,
    )
    first = resolve_execution_manifest(**base)
    second = resolve_execution_manifest(**{**base, "origin_experiment_id": "other-exp"})
    assert scientific_config_digest(first) == scientific_config_digest(second)
    changed = resolve_execution_manifest(
        **{
            **base,
            "factors": {"policy": "strict"},
            "condition_id": "strict",
            "protocol_digest": "p2",
        }
    )
    assert scientific_config_digest(changed) != scientific_config_digest(first)
    # The digest is deterministic.
    assert scientific_config_digest(first) == scientific_config_digest(
        resolve_execution_manifest(**base)
    )
