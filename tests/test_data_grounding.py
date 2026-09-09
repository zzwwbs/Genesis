"""AW-06: data-grounded domain initialization and origin propagation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genesis.compiler import StudyCompiler, ValidationIssue
from genesis.service import GenesisService

DATA = {
    "data/population.csv": "creator_id,audience_size,orientation\nc1,1200,music\nc2,3400,games\n",
}


def _write_package(
    root: Path,
    *,
    mode: str = "empirical",
    data_source: str = "data/population.csv",
    write_data: bool = True,
) -> Path:
    source = root / "package"
    source.mkdir(parents=True, exist_ok=True)
    if mode == "empirical":
        initialization = (
            f"initialization:\n  mode: empirical\n  data_source: {data_source}\n"
            "  state_field: population"
        )
    else:
        initialization = "initialization:\n  mode: researcher"
    artifacts = {
        "study.yaml": ('schema_version: "1.0"\nstudy_id: grounded-study\ntitle: grounded\n'),
        "openness.yaml": (
            'schema_version: "1.0"\nstudy_id: grounded-study\nprocesses:\n'
            "  - id: tick\n    executor: {}\n    context_policy: public\n"
            "    state_effects: [{field: counter, op: set}]\n"
        ),
        "theory.yaml": (
            'schema_version: "1.0"\nstudy_id: grounded-study\ntheory_family: exploratory\n'
        ),
        "domain.yaml": (
            'schema_version: "1.0"\nstudy_id: grounded-study\n'
            "states:\n  - id: counter\n    value_type: integer\n    initial: 0\n"
            "  - id: population\n    value_type: object\n"
            f"{initialization}\n"
        ),
        "protocol.yaml": (
            'schema_version: "1.0"\nstudy_id: grounded-study\ntime_model: {type: rounds, end: 2}\n'
        ),
        "outcomes.yaml": 'schema_version: "1.0"\nstudy_id: grounded-study\noutcomes: []\n',
        "models.yaml": 'schema_version: "1.0"\nstudy_id: grounded-study\n',
    }
    for name, content in artifacts.items():
        (source / name).write_text(content)
    if write_data:
        for relative, content in DATA.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    return source


def test_empirical_study_compiles_with_data_manifest(tmp_path: Path) -> None:
    source = _write_package(tmp_path)
    build = StudyCompiler(source).compile(tmp_path / "grounded-build")
    manifest = json.loads((build.path / "data_manifest.json").read_text())
    assert set(manifest) == {"population.csv"}
    assert len(manifest["population.csv"]) == 64
    assert StudyCompiler.verify_build(build.path)
    assert (build.path / "data" / "population.csv").is_file()  # asset frozen into build
    initialization = json.loads((build.path / "initialization.json").read_text())
    assert initialization["mode"] == "empirical"
    assert initialization["data_source"] == "data/population.csv"


def test_missing_data_asset_fails_compilation(tmp_path: Path) -> None:
    source = _write_package(tmp_path, mode="empirical", write_data=False)
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "bad-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "DATA_SOURCE_MISSING" in codes


def test_unsupported_data_format_fails_compilation(tmp_path: Path) -> None:
    source = _write_package(
        tmp_path, mode="empirical", data_source="data/population.txt", write_data=False
    )
    (source / "data").mkdir(exist_ok=True)
    (source / "data" / "population.txt").write_text("nope")
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "bad-format-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "DATA_SOURCE_INVALID" in codes


def _service_with_grounded_run(tmp_path: Path) -> GenesisService:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    source = _write_package(workspace / "imports")
    imported = service.import_package(source, specification_id="grounded-study")
    assert imported["status"] == "draft"
    approved = service.approve_specification("grounded-study", 1, "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(
        None, "builds/grounded-study", specification_id="grounded-study"
    )
    service.create_run(
        {"id": "grounded-run", "study_id": "grounded-study", "build": compiled["path"]}
    )
    return service


def test_runtime_initializes_population_from_empirical_data(tmp_path: Path) -> None:
    service = _service_with_grounded_run(tmp_path)
    try:
        result = service.execute_run(
            "grounded-run", executor_overrides={"tick": lambda _inv: {"counter": 1}}
        )
        assert result["status"] == "completed"
        _version, state = service.persistence.latest_json_state("grounded-run")
        assert state["counter"] == 1
        population = state["population"]
        assert population["origin"] == "imported"
        assert population["data_source"] == "data/population.csv"
        assert len(population["rows"]) == 2
        assert population["rows"][0]["creator_id"] == "c1"
    finally:
        service.close()


def test_export_includes_data_manifest(tmp_path: Path) -> None:
    service = _service_with_grounded_run(tmp_path)
    try:
        service.execute_run("grounded-run")
        service.export_run("grounded-run", "exports/grounded")
        bundle = service.workspace / "exports/grounded"
        assert (bundle / "data_manifest.json").is_file()
        manifest = json.loads((bundle / "data_manifest.json").read_text())
        assert manifest["population.csv"]
    finally:
        service.close()


def test_reproducibility_bundle_preserves_empirical_population(tmp_path: Path) -> None:
    """F2: a reproducibility (full) run bundle must carry the empirical
    initialization data so an imported replay of an empirically grounded run
    still initializes the population rows (regression: initialization.json /
    schemas.json / data assets were omitted from the bundle)."""
    import shutil

    from genesis.evidence import ExportMode
    from genesis.replay import ReplayMode

    service = _service_with_grounded_run(tmp_path)
    try:
        service.execute_run("grounded-run")
        _version, state = service.persistence.latest_json_state("grounded-run")
        assert len(state["population"]["rows"]) == 2

        service.export_run("grounded-run", "exports/repro", mode=ExportMode.REPRODUCIBILITY)
        bundle = service.workspace / "exports" / "repro"
        assert (bundle / "initialization.json").is_file()
        assert (bundle / "schemas.json").is_file()

        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "repro-bundle")
        importer = GenesisService(other)
        try:
            imported = importer.import_run(other / "imports" / "repro-bundle", run_id="imp-ground")
            assert imported["status"] == "imported"
            build_path = importer.resolve_path(importer.get_run("imp-ground").get("build"))
            assert (build_path / "initialization.json").is_file()
            assert (build_path / "data" / "population.csv").is_file()
            preview = importer.replay_preview(
                "imp-ground", mode=ReplayMode.PARTIAL, boundary="phase:0"
            )
            replayed = importer.replay_run(
                "imp-ground",
                mode=ReplayMode.PARTIAL,
                boundary="phase:0",
                preview_token=preview["preview_token"],
            )
            _version, replayed_state = importer.persistence.latest_json_state(replayed["run_id"])
            assert len(replayed_state.get("population", {}).get("rows", [])) == 2
        finally:
            importer.close()
    finally:
        service.close()
