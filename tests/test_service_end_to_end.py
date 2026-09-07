import shutil
from pathlib import Path

import pytest

from genesis.service import GenesisService

FIXTURE = Path(__file__).parent / "fixtures" / "specification"


def make_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    source = workspace / "studies" / "demo"
    source.parent.mkdir(parents=True)
    shutil.copytree(FIXTURE, source)
    return workspace, source


def test_compile_execute_and_restart_are_durable(tmp_path: Path) -> None:
    workspace, _ = make_workspace(tmp_path)
    service = GenesisService(workspace)
    service.initialize()
    service.create_study({"id": "platform-governance", "title": "Platform governance"})
    build = service.compile_study("studies/demo", "builds/demo")
    assert build["study_id"] == "platform-governance"
    service.create_run({"id": "run-1", "study_id": "platform-governance", "build": "builds/demo"})
    completed = service.execute_run("run-1")
    assert completed["status"] == "completed"
    service.close()

    reopened = GenesisService(workspace)
    assert reopened.get_study("platform-governance")["title"] == "Platform governance"
    assert reopened.get_run("run-1")["status"] == "completed"
    assert reopened.trace_run("run-1") == []
    reopened.close()


def test_workspace_rejects_paths_outside_its_root(tmp_path: Path) -> None:
    workspace, _ = make_workspace(tmp_path)
    service = GenesisService(workspace)
    with pytest.raises(ValueError, match="WORKSPACE_CONTAINMENT"):
        service.compile_study("../outside", "builds/demo")
    with pytest.raises(ValueError, match="WORKSPACE_CONTAINMENT"):
        service.compile_study("studies/demo", "../outside")
    service.close()


def test_run_controls_use_durable_expected_versions(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "workspace")
    service.create_run({"id": "run-1"})
    paused = service.transition_run("run-1", "paused", 0)
    assert paused["status"] == "paused"
    with pytest.raises(ValueError, match="EXPECTED_VERSION"):
        service.transition_run("run-1", "running", 0)
    service.close()
