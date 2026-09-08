"""EVD — capability-labelled evidence exchange (G4).

Covers run-pinned package closure export (editing/deleting the live package
must not change an executed run's export), exploration vs. reproducibility
modes with machine-readable capabilities, atomic publication, safe import
preflight and snapshot-vs-recomputed labelling.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from genesis.evidence import (
    ExportMode,
    evaluate_capabilities,
)
from genesis.service import GenesisService

SCHEMA = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}


def _prepared_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        import genesis.service as service_module

        class FakeProvider:
            provider = "openai-compatible"

            def __init__(self, **_kw):
                pass

            def generate(self, request):
                import json as _json

                from genesis.providers import ProviderResponse

                text = _json.dumps({"text": "evidence-content"})
                return ProviderResponse(
                    text, self.provider, request.model, "req-1", parsed=_json.loads(text)
                )

        original = service_module.OpenAICompatibleProvider
        service_module.OpenAICompatibleProvider = FakeProvider  # type: ignore[misc]
        try:
            _compile_evidence_study(service, workspace)
        finally:
            service_module.OpenAICompatibleProvider = original
    finally:
        service.close()
    return workspace


def _compile_evidence_study(service: GenesisService, workspace: Path) -> None:
    service.create_model_profile(
        {
            "id": "mp",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m1",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    draft = service.create_specification(
        {
            "id": "evidence-study",
            "title": "evidence study",
            "models": [
                {"id": "mp", "provider": "openai-compatible", "model": "m1", "parameters": {}}
            ],
            "processes": [
                {
                    "id": "compose",
                    "openness_rationale": "content form is the phenomenon",
                    "closure_rationale": "bounded options would constrain the process",
                    "executor": {"mode": "generative", "model_profile": "mp"},
                    "context_policy": "public",
                    "prompt_ref": "compose",
                    "outputs": [{"artifact_type": "text", "schema_ref": "compose-out"}],
                },
                {"id": "finalize", "executor": {}, "context_policy": "public"},
            ],
            "theory": {"theory_family": "exploratory"},
            "domain": {
                "artifacts": [{"id": "compose-out", "artifact_type": "text"}],
                "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
            },
            "protocol": {"time_model": {"type": "rounds", "end": 1}},
            "outcomes": [],
            "prompts": {"compose": "Compose from {context}"},
        }
    )
    schema_dir = workspace / ".genesis" / "specifications" / "evidence-study" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    data_dir = workspace / ".genesis" / "specifications" / "evidence-study" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "seed.txt").write_text("evidence-seed\n")
    revised = service.update_specification(
        "evidence-study", {"description": "with assets"}, draft["version"]
    )
    service.approve_specification("evidence-study", revised["version"], "researcher")
    compiled = service.compile_study(
        None, "builds/evidence-study", specification_id="evidence-study"
    )
    service.create_run(
        {"id": "evidence-run", "study_id": "evidence-study", "build": compiled["path"]}
    )
    service.execute_run("evidence-run")


def test_export_pins_run_package_after_live_edit(tmp_path: Path) -> None:
    """EVD-T01: exporting an executed run uses its pinned closure, not the live package."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        spec_dir = workspace / ".genesis" / "specifications" / "evidence-study"
        (spec_dir / "study.yaml").write_text("schema_version: '1.0'\nstudy_id: DIFFERENT\n")
        (spec_dir / "schemas" / "compose-out.yaml").write_text("type: string\n")
        service.export_run("evidence-run", "exports/pinned", mode=ExportMode.REPRODUCIBILITY)
        bundle = workspace / "exports" / "pinned"
        closure = json.loads((bundle / "package_closure.json").read_text())
        by_path = {asset["path"]: asset for asset in closure["assets"]}
        assert by_path["schemas/compose-out.yaml"]["media_type"] == "text/yaml"
        assert (
            (bundle / "package" / "schemas" / "compose-out.yaml")
            .read_text()
            .startswith("type: object")
        ), "export must use the executed package bytes, not the edited live schema"
        # bundle manifest pins digest + capabilities
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        assert manifest["export_mode"] == "reproducibility"
        assert manifest["package_digest"]
        caps = {cap["capability"]: cap for cap in manifest["capabilities"]}
        assert caps["inspect"]["available"] is True
        assert caps["reexecute"]["available"] is True
    finally:
        service.close()


def test_reproducibility_export_requires_pinned_closure(tmp_path: Path) -> None:
    """EVD-T04: a requested full/reproducibility export fails with a completeness report."""
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "no-build-study",
                "title": "no build",
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
        service.approve_specification("no-build-study", draft["version"], "researcher")
        service.create_run({"id": "no-build-run", "study_id": "no-build-study", "build": ""})
        with pytest.raises(ValueError, match="REPRODUCIBILITY"):
            service.export_run("no-build-run", "exports/full", mode=ExportMode.REPRODUCIBILITY)
        # Exploration works without a reproducibility claim.
        service.export_run("no-build-run", "exports/explore", mode=ExportMode.EXPLORATION)
        manifest = json.loads(
            (tmp_path / "workspace" / "exports" / "explore" / "bundle_manifest.json").read_text()
        )
        caps = {cap["capability"]: cap for cap in manifest["capabilities"]}
        assert caps["reexecute"]["available"] is False
        assert caps["reexecute"]["missing"]
    finally:
        service.close()


def test_export_refuses_overwrite_by_default(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("evidence-run", "exports/once", mode=ExportMode.EXPLORATION)
        with pytest.raises(ValueError, match="EXPORT_DESTINATION"):
            service.export_run("evidence-run", "exports/once", mode=ExportMode.EXPLORATION)
    finally:
        service.close()


def test_import_preflight_rejects_tampered_members(tmp_path: Path) -> None:
    """EVD-T05: tampered content or unsafe paths expose no valid partial run."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("evidence-run", "exports/tamper", mode=ExportMode.EXPLORATION)
        bundle = workspace / "exports" / "tamper"
        (bundle / "events.json").write_text(json.dumps([{"event_id": "tampered"}]))

        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "bundle")
        importer = GenesisService(other)
        try:
            with pytest.raises(ValueError, match="IMPORT_BUNDLE|IMPORT_INTEGRITY"):
                importer.import_run(other / "imports" / "bundle", run_id="tamper-run")
            with pytest.raises(KeyError):
                importer.get_run("tamper-run")
        finally:
            importer.close()
    finally:
        service.close()


def test_exploration_import_preserves_snapshot_and_origin(tmp_path: Path) -> None:
    """EVD-T06: exploration import/re-export keeps outcome snapshots and origin."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("evidence-run", "exports/origin", mode=ExportMode.EXPLORATION)
        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(workspace / "exports" / "origin", other / "imports" / "bundle")
        importer = GenesisService(other)
        try:
            result = importer.import_run(other / "imports" / "bundle", run_id="imported-evidence")
            assert result["status"] == "imported"
            # Re-export labels the snapshot and keeps the origin identity.
            importer.export_run("imported-evidence", "exports/again", mode=ExportMode.EXPLORATION)
            again = other / "exports" / "again"
            manifest = json.loads((again / "bundle_manifest.json").read_text())
            assert manifest["source_run_id"] == "evidence-run"
            assert (
                manifest["local_import_id"] in {"", None}
                or manifest.get("run_id") == "imported-evidence"
            )
        finally:
            importer.close()
    finally:
        service.close()


def test_capability_evaluation_reports_missing_prerequisites() -> None:
    caps = evaluate_capabilities(
        has_build=False,
        has_closure=False,
        has_recorded_outputs=False,
        has_checkpoint_evidence=False,
        has_outcomes=True,
    )
    by_id = {cap["capability"]: cap for cap in caps}
    assert by_id["inspect"]["available"] is True
    assert by_id["read_outcome_snapshot"]["available"] is True
    assert by_id["recompute_outcomes"]["available"] is False
    assert by_id["reexecute"]["available"] is False
    assert by_id["replay_recorded"]["available"] is False
    assert by_id["branch_at_checkpoint"]["available"] is False


def test_capability_evaluation_reexecute_requires_build_and_closure() -> None:
    caps = evaluate_capabilities(
        has_build=True,
        has_closure=True,
        has_recorded_outputs=True,
        has_checkpoint_evidence=False,
        has_outcomes=True,
    )
    by_id = {cap["capability"]: cap for cap in caps}
    assert by_id["reexecute"]["available"] is True
    assert by_id["replay_recorded"]["available"] is True
    assert by_id["branch_at_checkpoint"]["available"] is False


# ---------------------------------------------------------------------------
# F1: evidence import integrity and size enforcement
# ---------------------------------------------------------------------------


def test_import_rejects_incomplete_member_coverage(tmp_path: Path) -> None:
    """F1: a manifest omitting a bundle member must be rejected."""
    from genesis.evidence import verify_bundle_manifest

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run_manifest.json").write_text('{"run_id": "r"}')
    (bundle / "events.json").write_text("[]")
    # members lists only one of the two real files -> incomplete coverage.
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "export_mode": "exploration",
                "members": [
                    {
                        "path": "run_manifest.json",
                        "digest": hashlib.sha256(b'{"run_id": "r"}').hexdigest(),
                        "size": 15,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete member coverage"):
        verify_bundle_manifest(bundle)


def test_import_verifies_actual_member_size(tmp_path: Path) -> None:
    """F1: declared size must match the actual file size, not be trusted."""
    from genesis.evidence import verify_bundle_manifest

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "events.json").write_text("[1,2,3]")
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "export_mode": "exploration",
                "members": [
                    {
                        "path": "events.json",
                        "digest": hashlib.sha256(b"[1,2,3]").hexdigest(),
                        "size": 999,  # declared size does not match the real file
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="size differs"):
        verify_bundle_manifest(bundle)


def test_import_rejects_absolute_member_path(tmp_path: Path) -> None:
    """F1: absolute member paths must be rejected."""
    from genesis.evidence import verify_bundle_manifest

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("data")
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": 1,
                "export_mode": "exploration",
                "members": [
                    {
                        "path": str(outside),
                        "digest": hashlib.sha256(b"data").hexdigest(),
                        "size": 4,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unsafe member path"):
        verify_bundle_manifest(bundle)


def test_import_size_limit_applies_to_real_bundle(tmp_path: Path) -> None:
    """F1: a full-export bundle under a tiny size limit must be rejected."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("evidence-run", "exports/sized", mode=ExportMode.EXPLORATION)
        bundle = workspace / "exports" / "sized"
        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "bundle")
        importer = GenesisService(other)
        try:
            with pytest.raises(ValueError, match="IMPORT_SIZE"):
                importer.import_run(
                    other / "imports" / "bundle", run_id="sized-run", size_limit_bytes=1
                )
        finally:
            importer.close()
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F6: reproducibility capabilities reflect actual bundle contents
# ---------------------------------------------------------------------------


def test_reproducibility_bundle_includes_executable_build_files(tmp_path: Path) -> None:
    """F6: a reproducibility export must carry the executable build files."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("evidence-run", "exports/full", mode=ExportMode.REPRODUCIBILITY)
        bundle = workspace / "exports" / "full"
        for name in (
            "processes.json",
            "context_policies.json",
            "state_model.json",
            "artifact_catalog.json",
            "outcome_plan.json",
        ):
            assert (bundle / name).is_file(), f"missing executable build file {name}"
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        caps = {cap["capability"]: cap for cap in manifest["capabilities"]}
        assert caps["reexecute"]["available"] is True
        assert caps["replay_recorded"]["available"] is True
    finally:
        service.close()


def test_exploration_bundle_without_build_does_not_claim_reexecute(tmp_path: Path) -> None:
    """F6: a bundle without executable build files must not advertise them."""
    from genesis.evidence import evaluate_capabilities

    caps = evaluate_capabilities(
        has_build=False,
        has_closure=False,
        has_recorded_outputs=False,
        has_checkpoint_evidence=False,
        has_outcomes=True,
    )
    by_id = {cap["capability"]: cap for cap in caps}
    assert by_id["reexecute"]["available"] is False
    assert "build" in by_id["reexecute"]["missing"]


# ---------------------------------------------------------------------------
# F14 (effect): exported bundle paths point at real files
# ---------------------------------------------------------------------------


def test_export_returns_existing_destination_paths(tmp_path: Path) -> None:
    """F14: export_run must return paths under the final destination, not the
    renamed staging directory."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        paths = service.export_run("evidence-run", "exports/final", mode=ExportMode.EXPLORATION)
        assert paths, "no paths returned"
        for path in paths:
            assert path.exists(), f"returned path does not exist: {path}"
            assert path.is_relative_to((workspace / "exports" / "final").resolve())
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F3 (effect): reproducibility imports restore the executable build and replay
# ---------------------------------------------------------------------------


def test_reproducibility_import_restores_build_and_replays(tmp_path: Path) -> None:
    """F3: importing a reproducibility bundle restores + registers the build,
    so replay no longer fails with REPLAY_SOURCE_MISSING."""
    from genesis.replay import ReplayMode

    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("evidence-run", "exports/full", mode=ExportMode.REPRODUCIBILITY)
        bundle = workspace / "exports" / "full"
        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "bundle")
        importer = GenesisService(other)
        try:
            importer.import_run(other / "imports" / "bundle", run_id="imported-full")
            record = importer.get_run("imported-full")
            assert record.get("build"), "imported run must carry a restored build"
            assert record.get("manifest", {}).get("build_restored") is True
            # Replay works against the verified build.
            replay = importer.replay_run("imported-full", mode=ReplayMode.FULL)
            assert replay["run_id"].startswith("imported-full-replay-")
        finally:
            importer.close()
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F6 (effect): legacy import path applies coverage/containment/size checks
# ---------------------------------------------------------------------------


def test_legacy_import_rejects_empty_integrity_manifest(tmp_path: Path) -> None:
    """F6: an empty integrity.json must not bypass coverage checks."""
    service = GenesisService(tmp_path / "workspace-import")
    bundle = service.workspace / "legacy-bundle"
    bundle.mkdir()
    (bundle / "run_manifest.json").write_text(
        json.dumps({"run_id": "legacy-run", "status": "completed"})
    )
    (bundle / "events.json").write_text(json.dumps([{"event_id": "e1"}]))
    (bundle / "artifacts.json").write_text(json.dumps([]))
    (bundle / "integrity.json").write_text("{}")
    try:
        with pytest.raises(ValueError, match="IMPORT_INTEGRITY"):
            service.import_run(bundle, run_id="legacy-run", size_limit_bytes=1)
    finally:
        service.close()


def test_legacy_import_enforces_containment_and_size(tmp_path: Path) -> None:
    """F6: legacy members must stay contained and count toward the size limit."""
    service = GenesisService(tmp_path / "workspace-import2")
    bundle = service.workspace / "legacy-bundle2"
    bundle.mkdir()
    (bundle / "run_manifest.json").write_text(
        json.dumps({"run_id": "legacy-run2", "status": "completed"})
    )
    (bundle / "events.json").write_text(json.dumps([{"event_id": "e1"}]))
    (bundle / "artifacts.json").write_text(json.dumps([]))
    import hashlib as _hashlib

    (bundle / "integrity.json").write_text(
        json.dumps(
            {
                "run_manifest.json": _hashlib.sha256(
                    (bundle / "run_manifest.json").read_bytes()
                ).hexdigest(),
                "events.json": _hashlib.sha256((bundle / "events.json").read_bytes()).hexdigest(),
                "artifacts.json": _hashlib.sha256(
                    (bundle / "artifacts.json").read_bytes()
                ).hexdigest(),
            }
        )
    )
    try:
        with pytest.raises(ValueError, match="IMPORT_SIZE"):
            service.import_run(bundle, run_id="legacy-run2", size_limit_bytes=1)
    finally:
        service.close()
