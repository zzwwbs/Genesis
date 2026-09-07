"""AW-07: stochastic and computational executor modes wired through the service."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from genesis.compiler import ValidationIssue
from genesis.service import GenesisService

sys.path.insert(0, str(Path(__file__).parent))


STUDY = {
    "id": "modes-study",
    "title": "executor modes",
    "theory": {"theory_family": "exploratory"},
    "domain": {
        "states": [{"id": "counter", "value_type": "integer", "initial": 2}],
        "visibility": [{"id": "modes-context", "allow": ["counter"]}],
    },
    "protocol": {"time_model": {"type": "rounds", "end": 2}},
    "outcomes": [],
    "models": [],
}


def _spec(process: dict) -> GenesisService:
    service = None

    def inner(tmp_path: Path) -> GenesisService:
        nonlocal service
        service = GenesisService(tmp_path / "workspace")
        payload = {**STUDY, "processes": [process], "id": "modes-study"}
        payload.pop("id")
        payload = {"id": "modes-study", **payload}
        draft = service.create_specification(payload)
        approved = service.approve_specification("modes-study", draft["version"], "researcher")
        assert approved["status"] == "approved"
        compiled = service.compile_study(None, "builds/modes-study", specification_id="modes-study")
        service.create_run(
            {"id": "modes-run", "study_id": "modes-study", "build": compiled["path"]}
        )
        return service

    return inner


def test_stochastic_mode_requires_declared_function(tmp_path: Path) -> None:

    from genesis.compiler import StudyCompiler

    source = tmp_path / "pkg"
    source.mkdir()
    (source / "study.yaml").write_text('schema_version: "1.0"\nstudy_id: x\ntitle: x\n')
    (source / "openness.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: x\nprocesses:\n'
        "  - id: s\n    executor:\n      mode: stochastic\n    context_policy: private\n"
    )
    (source / "theory.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: x\ntheory_family: exploratory\n'
    )
    (source / "domain.yaml").write_text('schema_version: "1.0"\nstudy_id: x\n')
    (source / "protocol.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: x\ntime_model: {type: rounds}\n'
    )
    (source / "outcomes.yaml").write_text('schema_version: "1.0"\nstudy_id: x\noutcomes: []\n')
    (source / "models.yaml").write_text('schema_version: "1.0"\nstudy_id: x\n')
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "bad")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "EXECUTOR_UNAVAILABLE" in codes


def test_stochastic_executor_runs_seeded_function(tmp_path: Path) -> None:
    process = {
        "id": "s",
        "executor": {
            "mode": "stochastic",
            "parameters": {"function": "executor_functions:stochastic_tick"},
        },
        "context_policy": "modes-context",
        "state_effects": [{"field": "counter", "op": "set"}],
    }
    service = _spec(process)(tmp_path)
    try:
        result = service.execute_run("modes-run")
        assert result["status"] == "completed"
        _version, state = service.persistence.latest_json_state("modes-run")
        assert 0 <= state["counter"] <= 5
        # A second run derives a distinct seed -> distinct realisation is valid.
        service.create_run(
            {
                "id": "modes-run-2",
                "study_id": "modes-study",
                "build": service.get_run("modes-run")["build"],
            }
        )
        service.execute_run("modes-run-2")
        _v2, state2 = service.persistence.latest_json_state("modes-run-2")
        assert 0 <= state2["counter"] <= 5
    finally:
        service.close()


def test_computational_executor_doubles_context_value(tmp_path: Path) -> None:
    process = {
        "id": "c",
        "executor": {
            "mode": "computational",
            "parameters": {"entry_point": "executor_functions:computational_double"},
        },
        "context_policy": "modes-context",
        "state_effects": [{"field": "counter", "op": "set"}],
    }
    service = _spec(process)(tmp_path)
    try:
        result = service.execute_run("modes-run")
        assert result["status"] == "completed"
        _version, state = service.persistence.latest_json_state("modes-run")
        assert state["counter"] == 4  # initial 2 doubled via the authorised context
        events = service.trace_run("modes-run")
        completed = [e for e in events if e["kind"] == "process_completed"]
        assert completed[0]["metadata"]["mode"] == "computational"
    finally:
        service.close()


def test_extension_mode_resolves_registered_factory(tmp_path: Path) -> None:
    """AW-07: extension processes run their registered factory."""

    import executor_functions

    from genesis.service import GenesisService

    service = None

    def run(tmp_path: Path) -> GenesisService:
        nonlocal service
        service = GenesisService(tmp_path / "workspace")
        payload = {
            "id": "extension-study",
            "title": "extensions",
            "processes": [
                {
                    "id": "probe",
                    "executor": {"mode": "extension", "extension_ref": "fixture-extension"},
                    "context_policy": "public",
                }
            ],
            "theory": {"theory_family": "exploratory"},
            "domain": {},
            "protocol": {"time_model": {"type": "rounds", "end": 2}},
            "outcomes": [],
            "models": [],
        }
        service.register_extension("fixture-extension", executor_functions.extension_probe)
        draft = service.create_specification(payload)
        service.approve_specification("extension-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/extension-study", specification_id="extension-study"
        )
        service.create_run(
            {"id": "ext-run", "study_id": "extension-study", "build": compiled["path"]}
        )
        return service

    service = run(tmp_path)
    try:
        result = service.execute_run("ext-run")
        assert result["status"] == "completed"
        events = service.trace_run("ext-run")
        completed = [e for e in events if e["kind"] == "process_completed"]
        assert completed[0]["metadata"]["mode"] == "extension"
        artifacts = service.persistence.list_artifacts("ext-run")
        payload = artifacts[0]["payload"]
        assert b"extension" in payload and b"7" in payload
        # The registered extension is listed.
        listed = service.list_extensions()
        assert any(item["id"] == "fixture-extension" for item in listed)
    finally:
        service.close()


def test_extension_mode_fails_when_unregistered(tmp_path: Path) -> None:
    """AW-07: unregistered extension references fail clearly at execution."""
    import pytest

    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        service.register_extension("other-extension", lambda _inv: {"x": 1})
        draft = service.create_specification(
            {
                "id": "missing-ext-study",
                "title": "missing ext",
                "processes": [
                    {
                        "id": "probe",
                        "executor": {
                            "mode": "extension",
                            "extension_ref": "ghost-extension",
                        },
                        "context_policy": "public",
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {},
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [],
                "models": [],
            }
        )
        service.approve_specification("missing-ext-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/missing-ext-study", specification_id="missing-ext-study"
        )
        service.create_run(
            {
                "id": "ghost-run",
                "study_id": "missing-ext-study",
                "build": compiled["path"],
            }
        )
        with pytest.raises(ValueError, match="EXECUTOR_UNREGISTERED"):
            service.execute_run("ghost-run")
    finally:
        service.close()


def test_extension_registry_survives_service_restart(tmp_path: Path) -> None:
    """Finding 6: resolvable extensions reload after a service restart."""
    import executor_functions

    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    first = GenesisService(workspace)
    first.register_extension(
        "persistent-extension",
        executor_functions.extension_probe,
        entry_point="executor_functions:extension_probe",
    )
    first.close()
    reopened = GenesisService(workspace)
    try:
        listed = reopened.list_extensions()
        assert any(item["id"] == "persistent-extension" for item in listed)
        # And the reloaded factory is usable.

        from genesis.service import _SelectiveExecutor  # noqa: F401

        executor = opened_registry_factory(reopened)
        assert executor is not None
    finally:
        reopened.close()


def opened_registry_factory(service) -> object:

    registry = service._extensions
    if registry is None:
        return None
    from genesis.runtime import ProcessInvocation

    factory = registry.get("persistent-extension")
    result = factory(ProcessInvocation("i", "r", "p", context=None))
    assert result == {"extension": True, "value": 7}
    return factory


def test_extension_integrity_binds_to_module_code(tmp_path) -> None:
    """Finding 6: integrity hashes module bytes; code edits block reload."""
    import hashlib
    import sys

    import executor_functions  # noqa: F401  (module dir on path)

    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    module_dir = tmp_path / "extmod"
    module_dir.mkdir()
    module_file = module_dir / "extmod.py"
    module_file.write_text("def factory(invocation):\n    return {'value': 1}\n")
    # Make the tmp module importable under a unique name.
    sys.path.insert(0, str(module_dir))
    entry_point = "extmod:factory"
    first = GenesisService(workspace)
    registered = first.register_extension(
        "code-bound-extension", __import__("extmod").factory, entry_point=entry_point
    )
    material = entry_point.encode() + b"\0" + module_file.read_bytes()
    assert registered["integrity_hash"] == hashlib.sha256(material).hexdigest()
    assert registered["hash_method"] == "module-sha256"
    first.close()

    reopened = GenesisService(workspace)
    try:
        assert any(item["id"] == "code-bound-extension" for item in reopened.list_extensions())
    finally:
        reopened.close()

    # Changing the implementation invalidates the frozen hash.
    module_file.write_text("def factory(invocation):\n    return {'value': 2}\n")
    again = GenesisService(workspace)
    try:
        ids = {item["id"] for item in again.list_extensions()}
        assert "code-bound-extension" not in ids
    finally:
        again.close()
    sys.path.remove(str(module_dir))
