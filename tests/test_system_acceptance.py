import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.analysis import AnalysisExporter
from genesis.app import create_app
from genesis.service import GenesisService


def test_restartable_local_workflow_replay_checkpoint_and_export(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source = workspace / "studies" / "demo"
    source.parent.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent / "fixtures" / "specification", source)

    service = GenesisService(workspace)
    service.create_study({"id": "platform-governance", "title": "Platform governance"})
    service.compile_study("studies/demo", "builds/demo")
    service.create_run({"id": "run-1", "study_id": "platform-governance", "build": "builds/demo"})
    assert service.execute_run("run-1")["status"] == "completed"
    checkpoint = service.checkpoint_run("run-1")
    assert service.persistence.restore_checkpoint(checkpoint)["run"]["status"] == "completed"
    replay = service.replay_run("run-1")
    assert replay["source_run_id"] == "run-1"
    service.export_run("run-1", "exports/run-1")
    assert AnalysisExporter.verify_bundle(workspace / "exports" / "run-1")
    service.close()

    client = TestClient(create_app(workspace))
    assert client.get("/runs/run-1").json()["status"] == "completed"
    assert client.get("/runs/run-1/events").json() == []
