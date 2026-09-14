"""Fixes from the full fresh review at eef14d4 (H1-H3, M4-M18)."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from genesis.providers import ProviderRequest, ProviderResponse
from genesis.service import GenesisService

STUDY: dict[str, Any] = {
    "id": "review-study",
    "title": "review study",
    "models": [{"id": "mp", "provider": "openai-compatible", "model": "m1", "parameters": {}}],
    "processes": [
        {
            "id": "compose",
            "openness_rationale": "content form is the phenomenon",
            "closure_rationale": "a fixed corpus would remove the variation",
            "executor": {"mode": "generative", "model_profile": "mp"},
            "context_policy": "public",
            "prompt_ref": "compose",
            "outputs": [{"artifact_type": "text", "schema_ref": "compose-out"}],
        }
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {"artifacts": [{"id": "compose-out", "artifact_type": "text"}]},
    "protocol": {"time_model": {"type": "rounds", "end": 1}},
    "outcomes": [],
    "prompts": {"compose": "Compose from {context}"},
}


class _HookProvider:
    provider = "openai-compatible"
    hook: Callable[[ProviderRequest], None] | None = None

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if type(self).hook is not None:
            type(self).hook(request)
        text = json.dumps({"text": "generated"})
        return ProviderResponse(text, self.provider, request.model, "r", parsed=json.loads(text))


def _approve_and_compile(service: GenesisService, workspace: Path, target: str) -> dict[str, Any]:
    schemas = workspace / ".genesis" / "specifications" / STUDY["id"] / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    current = service.get_specification(STUDY["id"])
    revised = service.update_specification(
        STUDY["id"], {"description": f"compiled to {target}"}, current["version"]
    )
    service.approve_specification(STUDY["id"], revised["version"], "researcher")
    return service.compile_study(None, target, specification_id=STUDY["id"])


@pytest.fixture
def compiled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _HookProvider)
    _HookProvider.hook = None
    workspace = tmp_path / "ws"
    service = GenesisService(workspace)
    service.create_model_profile(
        {
            "id": "mp",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m1",
            "api_key_env": "GENESIS_REVIEW_KEY",
        }
    )
    service.create_specification(STUDY)
    build = _approve_and_compile(service, workspace, "builds/main")
    service.create_run({"id": "run-a", "study_id": STUDY["id"], "build": build["path"]})
    try:
        yield service, workspace
    finally:
        _HookProvider.hook = None
        service.close()


# --- H3: a failure before execution leaves the run as it was ---------------------


def test_a_missing_build_leaves_the_run_created(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        service.create_run({"id": "r", "build": "builds/nope"})
        for _attempt in range(2):
            with pytest.raises(FileNotFoundError):
                service.execute_run("r")
            run = service.get_run("r")
            assert run["status"] == "created" and run.get("manifest") is None
        assert service.transition_run("r", "cancelled", run["version"])["status"] == "cancelled"
    finally:
        service.close()


# --- M8: a cancel observed during execution is not transitioned twice ------------


def test_cancelling_during_execution_returns_the_cancelled_run(compiled) -> None:
    service, _workspace = compiled

    def cancel(_request: ProviderRequest) -> None:
        run = service.get_run("run-a")
        service.transition_run("run-a", "cancelled", run["version"])

    _HookProvider.hook = cancel
    assert service.execute_run("run-a")["status"] == "cancelled"


# --- M18: one caller executes a run at a time; a running run can be resumed ------


def test_a_second_concurrent_execution_is_refused(compiled) -> None:
    service, _workspace = compiled
    started, release = threading.Event(), threading.Event()

    def block(_request: ProviderRequest) -> None:
        started.set()
        release.wait(10)

    _HookProvider.hook = block
    results: list[dict[str, Any]] = []
    worker = threading.Thread(target=lambda: results.append(service.execute_run("run-a")))
    worker.start()
    try:
        assert started.wait(10)
        with pytest.raises(ValueError, match="RUN_ALREADY_EXECUTING"):
            service.execute_run("run-a")
    finally:
        release.set()
        worker.join(20)
    assert results and results[0]["status"] == "completed"
    assert service.execute_run("run-a")["status"] == "completed"


def test_a_run_left_running_is_resumed_and_recorded_as_such(compiled) -> None:
    service, _workspace = compiled
    run = service.get_run("run-a")
    service.persistence.transition_run("run-a", "running", run["version"])  # a crashed process
    assert service.execute_run("run-a")["status"] == "completed"
    assert service.get_run("run-a")["executions"][-1]["resumed_from_status"] == "running"


# --- M9: a resumed run refuses a build that no longer matches its manifest -------


def test_resuming_against_a_replaced_build_is_refused(compiled) -> None:
    service, workspace = compiled

    def pause(_request: ProviderRequest) -> None:
        run = service.get_run("run-a")
        service.transition_run("run-a", "paused", run["version"])

    _HookProvider.hook = pause
    assert service.execute_run("run-a")["status"] == "paused"
    _HookProvider.hook = None
    service.update_specification(
        STUDY["id"],
        {"title": "a different study"},
        service.get_specification(STUDY["id"])["version"],
    )
    other = _approve_and_compile(service, workspace, "builds/other")
    build_path = service.resolve_path(service.get_run("run-a")["build"])
    shutil.rmtree(build_path)
    shutil.copytree(service.resolve_path(other["path"]), build_path)
    with pytest.raises(ValueError, match="RUN_BUILD_CHANGED"):
        service.execute_run("run-a")
    assert service.get_run("run-a")["status"] == "paused"


# --- H1: only plain-text prompts are accepted, and they are compiled -------------

GOLDEN = Path(__file__).resolve().parent / "golden_studies" / "clickbait_mini"


def test_a_prompt_that_is_not_txt_is_refused_at_compile(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    package = tmp_path / "pkg"
    shutil.copytree(GOLDEN, package)
    prompt = package / "prompts" / "detect-clickbait.txt"
    prompt.rename(prompt.with_suffix(".yaml"))
    with pytest.raises(ValueError, match=r"must be prompts/detect-clickbait\.txt"):
        StudyCompiler(package).compile(tmp_path / "build")


def test_every_referenced_prompt_reaches_the_compiled_templates(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    build = StudyCompiler(GOLDEN).compile(tmp_path / "build")
    templates = json.loads((Path(build.path) / "prompt_templates.json").read_text())
    processes = json.loads((Path(build.path) / "processes.json").read_text())
    referenced = {p["prompt_ref"] for p in processes if p.get("prompt_ref")}
    assert referenced and referenced <= set(templates)


# --- H2: secrets and unsupported files stay out of closures and builds ----------


def test_secrets_and_unsupported_files_stay_out_of_the_closure_and_build(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler
    from genesis.execution_manifest import build_package_closure

    package = tmp_path / "pkg"
    shutil.copytree(GOLDEN, package)
    (package / "data").mkdir(exist_ok=True)
    planted = {
        "data/.env": "hidden",
        "prompts/.credentials/tokens.txt": "hidden",
        "schemas/keys.yaml": "credential",
        "data/api-token/rows.csv": "credential",
        "data/notes.xlsx": "unsupported file type",
        "prompts/draft.md": "unsupported file type",
    }
    for relative in planted:
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret\n")
    (package / "data" / "kept.csv").write_text("id,value\n1,2\n")

    closure = build_package_closure(package)
    paths = set(closure.assets)
    assert not paths & set(planted)
    assert "data/kept.csv" in paths
    assert dict(closure.excluded) == planted

    build = StudyCompiler(package).compile(tmp_path / "build")
    root = Path(build.path)
    assert not any((root / "closure" / relative).exists() for relative in planted)
    assert not (root / "data" / ".env").exists() and not (root / "data" / "notes.xlsx").exists()
    assert (root / "data" / "kept.csv").is_file()
    reported = {
        entry["path"]: entry["reason"] for entry in build.manifest["package_files_excluded"]
    }
    assert reported == planted


def test_a_clean_package_closure_digest_is_unchanged_by_the_exclusion_report(
    tmp_path: Path,
) -> None:
    from genesis.execution_manifest import build_package_closure

    closure = build_package_closure(GOLDEN)
    assert closure.excluded == ()


# --- M10/M11: retention redacts raw responses, scoped to the declaring process ---


def test_retention_redacts_only_raw_responses_of_declaring_processes(tmp_path: Path) -> None:
    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "purge.db", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        payloads = {
            "raw": {"process_id": "compose", "outputs": {"response": "sk"}, "raw_response": "sk"},
            "word": {"producer_process": "compose", "value": {"text": "a response word"}},
            "other": {"process_id": "keep", "raw_response": "kept raw"},
        }
        for version, (artifact_id, payload) in enumerate(payloads.items(), start=1):
            store.commit_process_result(
                {"event_id": f"e{version}", "run_id": "r", "kind": "process_completed"},
                {"run_id": "r", "state_version": version, "payload": b"{}"},
                [
                    {
                        "artifact_id": artifact_id,
                        "run_id": "r",
                        "payload": json.dumps(payload).encode(),
                    }
                ],
            )
        assert store.retention_purge("r", {"compose"}) == 1
        rows = {row["artifact_id"]: json.loads(row["payload"]) for row in store.iter_artifacts("r")}
        assert set(rows) == {"raw", "word", "other"}
        assert rows["raw"]["raw_response"] == "<purged-by-retention>"
        # Declared outputs remain, whatever their field is called: redacting any
        # key named "response" overwrote a study's own output (2026-09-14 H2).
        assert rows["raw"]["outputs"]["response"] == "sk"
        assert rows["word"]["value"] == {"text": "a response word"}
        assert rows["other"]["raw_response"] == "kept raw"
        assert store.retention_purge("r", {"compose"}) == 0  # idempotent
    finally:
        store.close()


def test_exported_outcomes_redact_raw_responses_under_purge_retention(
    compiled, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _workspace = compiled
    service.execute_run("run-a")
    monkeypatch.setattr(
        GenesisService,
        "_purging_processes",
        staticmethod(lambda processes: {"compose"}),
    )
    monkeypatch.setattr(
        service,
        "evaluate_outcomes",
        lambda run_id: [{"outcome_id": "texts", "raw_response": "top-secret"}],
    )
    service.export_run("run-a", "exports/review")
    exported = "".join(
        path.read_text(errors="ignore")
        for path in (service.workspace / "exports" / "review").rglob("*")
        if path.is_file() and path.suffix in {".json", ".csv"}
    )
    assert "top-secret" not in exported


# --- M6: unreferenced objects are reported and collected safely ------------------


def test_garbage_collection_removes_only_unreferenced_objects(compiled) -> None:
    import os
    import time

    service, _workspace = compiled
    service.execute_run("run-a")
    orphan = service.persistence.object_store.put(b"left behind by a failed commit")
    old = time.time() - 7200
    for path in service.persistence.object_store.root.glob("??/*"):
        os.utime(path, (old, old))
    report = service.collect_garbage()
    assert report["unreferenced_objects"] >= 1 and report["removed_objects"] == 0
    assert (service.persistence.object_store.root / orphan.digest[:2] / orphan.digest[2:]).exists()
    applied = service.collect_garbage(apply=True)
    assert applied["removed_objects"] == report["unreferenced_objects"]
    assert not (
        service.persistence.object_store.root / orphan.digest[:2] / orphan.digest[2:]
    ).exists()
    assert service.trace_run("run-a")
    version = service.get_specification(STUDY["id"])["version"]
    assert service.get_package_snapshot(STUDY["id"], version)
    assert service.doctor()["checks"]["unreferenced_objects"] == 0


def test_garbage_collection_refuses_to_remove_while_a_run_is_running(compiled) -> None:
    service, _workspace = compiled
    run = service.get_run("run-a")
    service.persistence.transition_run("run-a", "running", run["version"])
    with pytest.raises(ValueError, match="GC_RUNS_ACTIVE"):
        service.collect_garbage(apply=True)
    assert service.collect_garbage()["applied"] is False


# --- M5: a backup restores with its evidence ------------------------------------


def test_a_backup_restores_with_the_objects_its_records_reference(compiled, tmp_path: Path) -> None:
    service, _workspace = compiled
    service.execute_run("run-a")
    events = service.trace_run("run-a")
    result = service.backup_run("run-a", "backups/genesis.db")
    restored = tmp_path / "restored"
    (restored / ".genesis").mkdir(parents=True)
    shutil.copy(result["path"], restored / ".genesis" / "genesis.db")
    shutil.copytree(result["objects_path"], restored / ".genesis" / "objects")
    again = GenesisService(restored)
    try:
        assert [e["event_id"] for e in again.trace_run("run-a")] == [e["event_id"] for e in events]
    finally:
        again.close()


# --- M7: the parquet writer refuses a source that changed between passes --------


def test_parquet_writer_refuses_a_source_that_changes_between_passes(tmp_path: Path) -> None:
    from genesis.analysis import AnalysisExporter

    calls = {"n": 0}

    def drifting():
        calls["n"] += 1
        if calls["n"] == 1:
            return iter([{"a": 1, "b": "x"}])
        return iter([{"a": 2.5, "b": "x", "c": "added"}])

    target = tmp_path / "rows.parquet"
    with pytest.raises(ValueError, match="PARQUET_SOURCE_CHANGED"):
        AnalysisExporter.stream_rows_to_parquet(drifting, target)
    assert not target.exists()
    grown = {"n": 0}

    def growing():
        grown["n"] += 1
        return iter([{"a": 1}] * grown["n"])

    with pytest.raises(ValueError, match="PARQUET_SOURCE_CHANGED"):
        AnalysisExporter.stream_rows_to_parquet(growing, target)
    AnalysisExporter.stream_rows_to_parquet(lambda: iter([{"a": 1.5}, {"a": 2}]), target)
    assert target.is_file()


# --- M4, M12-M15: profiles and diagnostics ---------------------------------------

LIVE = {
    "provider": "openai-compatible",
    "base_url": "https://example.test/v1",
    "model": "m",
    "api_key_env": "GENESIS_REVIEW_KEY",
}


def _create_together(service: GenesisService, profile_ids: list[str]) -> list[str]:
    errors: list[str] = []
    barrier = threading.Barrier(len(profile_ids))

    def create(profile_id: str) -> None:
        barrier.wait()
        try:
            service.create_model_profile({**LIVE, "id": profile_id})
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=create, args=(pid,)) for pid in profile_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_concurrent_profile_writes_keep_every_profile(tmp_path: Path) -> None:
    for trial in range(8):
        service = GenesisService(tmp_path / f"ws{trial}")
        try:
            assert _create_together(service, ["p0", "p1", "p2", "p3"]) == []
            assert sorted(p["id"] for p in service.list_model_profiles()) == [
                "p0",
                "p1",
                "p2",
                "p3",
            ]
        finally:
            service.close()


def test_doctor_checks_database_integrity_and_reports_stored_secrets(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        healthy = service.doctor()
        assert healthy["checks"]["database_integrity"] == "ok"
        assert healthy["checks"]["credential_secrets_stored"] is False
        service.create_model_profile({**LIVE, "id": "pasted", "api_key": "sk-pasted"})
        checks = service.doctor()["checks"]
        assert checks["credential_secrets_stored"] is True
        assert checks["credential_presence"]["pasted"] is True

        class Corrupt:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def execute(self, sql: str, *args: Any) -> Any:
                if sql.startswith("PRAGMA quick_check"):
                    return self._inner.execute("SELECT '*** in database main *** page 3 corrupt'")
                return self._inner.execute(sql, *args)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        real = service.persistence.connection
        service.persistence.connection = Corrupt(real)  # type: ignore[assignment]
        try:
            report = service.doctor()
        finally:
            service.persistence.connection = real
        assert report["status"] == "degraded"
        assert "corrupt" in report["checks"]["database_integrity"]
    finally:
        service.close()


def test_answer_pool_profiles_report_status_and_are_not_used_as_assistants(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        service.create_model_profile(
            {"id": "pooled", "provider": "answer-pool", "model": "m", "pool": "p.json"}
        )
        assert service.model_profile_status("pooled") == {
            "id": "pooled",
            "credential_present": True,
            "credential_required": False,
        }
        with pytest.raises(ValueError, match="no model profile configured"):
            service.draft_from_model(instruction="draft")
        with pytest.raises(ValueError, match="cannot act as the assistant"):
            service.draft_from_model(instruction="draft", profile_id="pooled")
        with pytest.raises(ValueError, match="cannot act as the assistant"):
            service._elicitation_provider("pooled")
        assert service.doctor()["checks"]["credential_presence"]["pooled"] is True
    finally:
        service.close()


def test_unknown_model_profile_fields_are_refused(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        with pytest.raises(ValueError, match="INVALID_FIELD: .*time_out"):
            service.create_model_profile({**LIVE, "id": "typo", "time_out": 5})
        created = service.create_model_profile({**LIVE, "id": "live"})
        with pytest.raises(ValueError, match="INVALID_FIELD: .*modle"):
            service.update_model_profile("live", {"modle": "m2"}, created["version"])
        assert service.get_model_profile("live")["model"] == "m"
    finally:
        service.close()


# --- M16/M17: elicitation sessions ----------------------------------------------


@pytest.fixture
def assistant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.test_elicitation_conversation import ScriptedProvider

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ScriptedProvider)
    service = GenesisService(tmp_path / "workspace")
    service.create_model_profile({**LIVE, "id": "assistant", "model": "scripted-model"})
    for study in ("study", "other-study"):
        service.create_specification({"id": study, "title": study, "description": "draft"})
    try:
        yield service
    finally:
        service.close()


def _start(service: GenesisService, specification_id: str, session_id: str | None = None) -> Any:
    payload = {
        "specification_id": specification_id,
        "workflow_id": "three-layer-study",
        "model_profile_id": "assistant",
        "researcher_id": "researcher",
    }
    if session_id:
        payload["session_id"] = session_id
    return service.start_elicitation(payload)


def test_a_session_id_cannot_be_reused_for_another_specification(assistant) -> None:
    session = _start(assistant, "study", session_id="fixed-session")
    assert (
        _start(assistant, "study", session_id="fixed-session")["session_id"]
        == session["session_id"]
    )
    with pytest.raises(ValueError, match="ELICITATION_SESSION_CONFLICT"):
        _start(assistant, "other-study", session_id="fixed-session")


def test_a_failed_answer_is_rolled_back_so_a_retry_records_it_once(
    assistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _start(assistant, "study")
    session_id = session["session_id"]
    store = assistant._elicitation_store
    original = store.put_turn_evaluation
    failures = {"left": 1}

    def failing(*args: Any, **kwargs: Any) -> Any:
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "put_turn_evaluation", failing)
    with pytest.raises(OSError, match="disk full"):
        assistant.submit_elicitation_message(session_id, "Agents choose what to say.")
    after_failure = assistant.get_elicitation(session_id)
    assert after_failure["turns"] == session["turns"]
    assert after_failure["status"] == session["status"]
    retried = assistant.submit_elicitation_message(session_id, "Agents choose what to say.")
    answers = [turn["answer"] for turn in retried["turns"]]
    assert answers.count("Agents choose what to say.") == 1
