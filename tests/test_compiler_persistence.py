import json
from pathlib import Path

import pytest

from genesis.compiler import StudyCompiler, ValidationIssue
from genesis.persistence import ObjectStore, PersistenceCoordinator

FIXTURES = Path(__file__).parent / "fixtures" / "specification"


def test_compiler_loads_all_canonical_files_and_emits_deterministic_build(tmp_path):
    first = StudyCompiler(FIXTURES).compile(tmp_path / "one")
    second = StudyCompiler(FIXTURES).compile(tmp_path / "two")

    assert first.study_id == "platform-governance"
    assert first.build_hash == second.build_hash
    assert (tmp_path / "one" / "build_manifest.json").read_text() == (
        tmp_path / "two" / "build_manifest.json"
    ).read_text()
    assert {p.name for p in (tmp_path / "one").iterdir()} >= {
        "build_manifest.json",
        "processes.json",
        "process_graph.json",
        "context_policies.json",
        "state_model.json",
        "artifact_catalog.json",
        "outcome_plan.json",
        "validation_report.json",
    }


def test_compiler_reports_unresolved_context_reference(tmp_path):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    openness = source / "openness.yaml"
    openness.write_text(
        openness.read_text().replace(
            "processes: []",
            """processes:
  - id: decide
    executor: {}
    context_policy: missing-policy
""",
        )
    )
    with pytest.raises(ValueError, match="REF_CONTEXT_POLICY"):
        StudyCompiler(source).compile(tmp_path / "build")


def test_compiler_rejects_immediate_dependency_cycle(tmp_path):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    (source / "openness.yaml").write_text("""schema_version: '1.0'
study_id: platform-governance
processes:
  - id: first
    executor: {}
    context_policy: private
    dependencies: {after: [second]}
  - id: second
    executor: {}
    context_policy: private
    dependencies: {after: [first]}
""")
    with pytest.raises(ValueError, match="GRAPH_IMMEDIATE_CYCLE"):
        StudyCompiler(source).compile(tmp_path / "build")


def test_compiler_validation_issue_exposes_structured_fields(tmp_path):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    (source / "openness.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: platform-governance\nprocesses:\n"
        "  - id: decide\n    executor: {}\n    context_policy: missing-policy\n"
    )
    with pytest.raises(ValidationIssue) as caught:
        StudyCompiler(source).compile(tmp_path / "build")
    issue = caught.value.issues[0]
    assert issue.code == "REF_CONTEXT_POLICY"
    assert issue.severity == "error"
    assert issue.source_file == "openness.yaml"
    assert issue.json_pointer.endswith("context_policy")
    assert issue.remediation


def test_delayed_dependency_is_emitted_with_metadata(tmp_path):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    (source / "openness.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: platform-governance\nprocesses:\n"
        "  - id: first\n    executor: {}\n    context_policy: private\n"
        "    dependencies: {after: [second], delay: {rounds: 1}}\n"
        "  - id: second\n    executor: {}\n    context_policy: private\n"
        "    dependencies: {after: [first], delay: {rounds: 2}}\n"
    )
    build = StudyCompiler(source).compile(tmp_path / "build")
    graph = __import__("json").loads((build.path / "process_graph.json").read_text())
    assert graph["first"][0]["delayed"] is True
    assert graph["first"][0]["delay"]["rounds"] == 1


@pytest.mark.parametrize("delay", [0, -1, "one", None])
def test_compiler_rejects_non_positive_or_non_numeric_edge_delay(tmp_path, delay):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    (source / "openness.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: platform-governance\nprocesses:\n"
        f"  - id: first\n    executor: {{}}\n    context_policy: private\n"
        f"    dependencies: {{after: [second], delay: {{rounds: {delay!r}}}}}\n"
        "  - id: second\n    executor: {}\n    context_policy: private\n"
    )
    with pytest.raises(ValueError, match="DELAY_INVALID"):
        StudyCompiler(source).compile(tmp_path / "build")


def test_compiler_rejects_non_finite_delay_and_malformed_dependencies(tmp_path):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    (source / "openness.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: platform-governance\nprocesses:\n"
        "  - id: first\n    executor: {}\n    context_policy: private\n"
        "    dependencies: {after: nope, delay: {rounds: .nan}}\n"
    )
    with pytest.raises(ValueError, match="DEPENDENCIES_INVALID|DELAY_INVALID"):
        StudyCompiler(source).compile(tmp_path / "build")


def test_compiler_rejects_duplicate_ids_across_catalogs(tmp_path):
    source = tmp_path / "spec"
    source.mkdir()
    for path in FIXTURES.glob("*.yaml"):
        (source / path.name).write_text(path.read_text())
    (source / "models.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: platform-governance\nmodels:\n"
        "  - {id: same, provider: p, model: m}\n  - {id: same, provider: p, model: m}\n"
    )
    with pytest.raises(ValueError, match="DUPLICATE_ID"):
        StudyCompiler(source).compile(tmp_path / "build")


def test_build_integrity_manifest_verifies_and_rejects_tampering(tmp_path):
    import os
    import stat

    build = StudyCompiler(FIXTURES).compile(tmp_path / "build")
    assert StudyCompiler.verify_build(build.path)
    os.chmod(build.path / "processes.json", stat.S_IRUSR | stat.S_IWUSR)
    (build.path / "processes.json").write_text("tampered")
    with pytest.raises(ValueError, match="BUILD_INTEGRITY"):
        StudyCompiler.verify_build(build.path)


def test_build_integrity_rejects_missing_expected_file(tmp_path):
    build = StudyCompiler(FIXTURES).compile(tmp_path / "build")
    (build.path / "processes.json").unlink()
    with pytest.raises(ValueError, match="BUILD_INTEGRITY"):
        StudyCompiler.verify_build(build.path)


def test_build_integrity_rejects_manifest_tampering_and_unexpected_files(tmp_path):
    import os
    import stat

    build = StudyCompiler(FIXTURES).compile(tmp_path / "build")
    manifest = build.path / "integrity_manifest.json"
    payload = json.loads(manifest.read_text())
    payload["unexpected.json"] = "x"
    os.chmod(manifest, stat.S_IRUSR | stat.S_IWUSR)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="BUILD_INTEGRITY"):
        StudyCompiler.verify_build(build.path)


def test_object_store_is_content_addressed_and_integrity_checked(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    ref = store.put(b"hello", media_type="text/plain")
    assert ref.digest == store.put(b"hello", media_type="text/plain").digest
    assert store.get(ref) == b"hello"
    assert ref.path.exists()
    with pytest.raises(ValueError, match="integrity"):
        store.verify(ref.__class__(digest="0" * 64, media_type="text/plain", size=5))


def test_object_store_validates_size_and_cleans_orphans(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(ValueError, match="size"):
        store.put(b"x", size=2)
    orphan = tmp_path / "objects" / "orphan.tmp"
    orphan.write_bytes(b"orphan")
    assert store.cleanup_orphans() == 1
    assert not orphan.exists()


def test_object_store_rejects_conflicting_metadata_and_collects_unreferenced(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    ref = store.put(b"x", media_type="text/plain")
    with pytest.raises(ValueError, match="metadata"):
        store.put(b"x", media_type="application/json")
    assert store.collect_garbage(set()) == 1
    assert not ref.path.exists()


def test_object_store_verifies_declared_size_and_media_type_across_coordinators(tmp_path):
    first = PersistenceCoordinator(tmp_path / "one.db", tmp_path / "objects")
    first.commit_process_result(
        {"event_id": "e", "run_id": "r", "kind": "x"},
        {"run_id": "r", "state_version": 1, "payload": b"x"},
        [],
    )
    ref = first.object_store.put(b"x")
    with pytest.raises(ValueError, match="integrity|metadata"):
        first.object_store.verify(ref.__class__(ref.digest, "text/plain", 99, ref.path))
    second = PersistenceCoordinator(tmp_path / "one.db", tmp_path / "objects")
    with pytest.raises(ValueError, match="metadata"):
        second.object_store.put(b"x", "application/json")


def test_persistence_commit_is_atomic_and_rolls_back_on_failure(tmp_path):
    db = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    db.commit_process_result(
        event={"event_id": "e1", "run_id": "r1", "kind": "state_changed"},
        state={"run_id": "r1", "state_version": 1, "payload": b"state"},
        artifacts=[{"artifact_id": "a1", "run_id": "r1", "payload": b"artifact"}],
    )
    assert db.count("events") == db.count("states") == db.count("artifacts") == 1
    with pytest.raises(RuntimeError):
        db.commit_process_result(
            event={"event_id": "e2", "run_id": "r1", "kind": "broken"},
            state={"run_id": "r1", "state_version": 2, "payload": b"new"},
            artifacts=[{"artifact_id": "a2", "run_id": "r1", "payload": b"new"}],
            fail_after="state",
        )
    assert db.count("events") == db.count("states") == db.count("artifacts") == 1


def test_persistence_expected_version_and_idempotency(tmp_path):
    db = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    event = {"event_id": "e1", "run_id": "r1", "kind": "state_changed"}
    state = {"run_id": "r1", "state_version": 1, "payload": b"state"}
    db.commit_process_result(event, state, [], expected_state_version=0)
    db.commit_process_result(event, state, [], expected_state_version=0)
    assert db.count("events") == 1
    with pytest.raises(ValueError, match="STATE_VERSION"):
        db.commit_process_result(
            {"event_id": "e2", "run_id": "r1", "kind": "state_changed"},
            {"run_id": "r1", "state_version": 3, "payload": b"new"},
            [],
            expected_state_version=0,
        )


def test_persistence_rejects_mismatched_run_and_nonsequential_state(tmp_path):
    db = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    with pytest.raises(ValueError, match="RUN_ID"):
        db.commit_process_result(
            {"event_id": "e1", "run_id": "r1", "kind": "x"},
            {"run_id": "r2", "state_version": 1, "payload": b"x"},
            [],
        )
    db.commit_process_result(
        {"event_id": "e1", "run_id": "r1", "kind": "x"},
        {"run_id": "r1", "state_version": 1, "payload": b"x"},
        [],
    )
    with pytest.raises(ValueError, match="STATE_VERSION"):
        db.commit_process_result(
            {"event_id": "e2", "run_id": "r1", "kind": "x"},
            {"run_id": "r1", "state_version": 3, "payload": b"y"},
            [],
            expected_state_version=1,
        )
