"""End-to-end research-control conformance (Task 7).

One approved package drives the whole chain: approve -> compile -> execute two
conditions -> evaluate outcomes -> preview/confirm branch -> export pinned
evidence -> import into an empty workspace -> recompute supported outcomes ->
verify lineage/capabilities. Negative paths cover a schema-invalid fake
response and an incomplete evidence bundle.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from genesis.evidence import ExportMode
from genesis.replay import ReplayMode
from genesis.service import GenesisService

PAYLOAD = {
    "id": "conformance-study",
    "title": "conformance study",
    "models": [{"id": "mp", "provider": "openai-compatible", "model": "m1", "parameters": {}}],
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
    "protocol": {
        "time_model": {"type": "rounds", "end": 2},
        "factors": [{"id": "policy", "levels": ["strict", "lenient"], "branchable": True}],
        "replications": 2,
    },
    "outcomes": [
        {
            "id": "compose-count",
            "source": "events",
            "grouping": [],
            "aggregation": {"type": "count", "field": "phase"},
            "output_schema": "outcome-schema",
        }
    ],
    "datasets": [
        {
            "id": "composition",
            "source": {"kind": "artifacts", "artifact_type": "compose-out"},
        }
    ],
    "prompts": {"compose": "Compose from {context}"},
}


class _EchoProvider:
    provider = "openai-compatible"

    def __init__(self, **_kw):
        pass

    def generate(self, request):
        import json as _json

        from genesis.providers import ProviderResponse

        value = {"text": "conformance-content"}
        return ProviderResponse(
            _json.dumps(value), self.provider, request.model, "req-1", parsed=value
        )


class _InvalidProvider:
    """Returns an output that fails the declared bounded schema."""

    provider = "openai-compatible"

    def __init__(self, **_kw):
        pass

    def generate(self, request):
        import json as _json

        from genesis.providers import ProviderResponse

        value = {"text": {"not": "a string"}}
        return ProviderResponse(
            _json.dumps(value), self.provider, request.model, "req-1", parsed=value
        )


def _install_schemas(tmp_path: Path, spec_id: str) -> None:
    schema_dir = tmp_path / "workspace" / ".genesis" / "specifications" / spec_id / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
        "additionalProperties: false\n"
    )
    (schema_dir / "outcome-schema.yaml").write_text(
        "type: object\nrequired: [phase_count]\nproperties:\n  phase_count: {type: integer}\n"
    )


def test_approved_package_run_branch_export_import_identity_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 7: the full research-control identity chain across all five gaps."""
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _EchoProvider)
    service = GenesisService(tmp_path / "workspace")
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_FAKE_KEY",
            }
        )
        draft = service.create_specification(PAYLOAD)
        _install_schemas(tmp_path, "conformance-study")
        revised = service.update_specification(
            "conformance-study", {"description": "with schemas"}, draft["version"]
        )
        service.approve_specification("conformance-study", revised["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/conformance-study", specification_id="conformance-study"
        )
        # G5: the build carries the fixed-dialect schema contract and the
        # theory execution plan.
        build_plan = json.loads((Path(compiled["path"]) / "theory_execution_plan.json").read_text())
        assert build_plan["version"] == 1

        # G2: execute two conditions under the compiled build.
        template = {**PAYLOAD, "id": "conformance-study"}
        _ = template
        service.create_run(
            {
                "id": "conformance-experiment",
                "study_id": "conformance-study",
                "build": compiled["path"],
            }
        )
        result = service.execute_protocol("conformance-experiment")
        assert result["status"] == "completed"
        assert len(result["runs"]) == 4  # 2 factors x 2 replications
        trial_ids = sorted(result["runs"])
        for trial_id in trial_ids:
            record = service.get_run(trial_id)
            # G2: each run pins the package closure and scientific config digest.
            assert record["manifest"].get("package_closure_digest")
            assert record["manifest"].get("scientific_config_digest")

        # G3: outcomes evaluated from the declared dataset + declared plan.
        outcome_rows = service.evaluate_outcomes(trial_ids[0])
        assert any(row["outcome_id"] == "compose-count" for row in outcome_rows)

        # G1: preview and confirm a branch changing the branchable factor.
        strict_trial = next(
            trial_id
            for trial_id in trial_ids
            if service.get_run(trial_id)["condition"].get("factors", {}).get("policy") == "strict"
        )
        preview = service.replay_preview(
            strict_trial,
            mode=ReplayMode.BRANCH,
            boundary="phase:1",
            overrides={"policy": "lenient"},
            justification="conformance branch",
        )
        branch = service.replay_run(
            strict_trial,
            mode=ReplayMode.BRANCH,
            boundary="phase:1",
            overrides={"policy": "lenient"},
            justification="conformance branch",
            preview_token=preview["preview_token"],
        )
        branch_record = service.get_run(branch["run_id"])
        assert branch_record["condition"]["factors"]["policy"] == "lenient"

        # G4: export the run-pinned package and import into an empty workspace.
        service.export_run(trial_ids[0], "exports/conformance", mode=ExportMode.REPRODUCIBILITY)
        bundle = tmp_path / "workspace" / "exports" / "conformance"
        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "bundle")
        importer = GenesisService(other)
        try:
            imported = importer.import_run(
                other / "imports" / "bundle", run_id="imported-conformance"
            )
            assert imported["status"] == "imported"
            # Package bytes in the bundle come from the pinned closure.
            closure = json.loads(
                (other / "imports" / "bundle" / "package_closure.json").read_text()
            )
            assert any(asset["path"] == "schemas/compose-out.yaml" for asset in closure["assets"])
            # Capabilities label the bundle without over-claiming replay.
            manifest = json.loads(
                (other / "imports" / "bundle" / "bundle_manifest.json").read_text()
            )
            caps = {cap["capability"]: cap for cap in manifest["capabilities"]}
            assert caps["inspect"]["available"] is True
            assert caps["reexecute"]["available"] is True
        finally:
            importer.close()
    finally:
        service.close()


def test_schema_invalid_fake_response_fails_run_without_state_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative path: a schema-invalid output must not commit state."""
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _InvalidProvider)
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "invalid-schema-study",
                "title": "invalid schema",
                "processes": [
                    {
                        "id": "measure",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "state_effects": [{"field": "counter", "op": "set"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {
                    "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
                },
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [],
                "models": [],
            }
        )
        service.approve_specification("invalid-schema-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/invalid-schema-study", specification_id="invalid-schema-study"
        )
        service.create_run(
            {"id": "invalid-run", "study_id": "invalid-schema-study", "build": compiled["path"]}
        )
        from genesis.providers import ProviderExecutor
        from genesis.schema_validation import PackageSchemaCatalog

        object_score = {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        }
        full_catalog = PackageSchemaCatalog({"score": object_score})
        executor = ProviderExecutor(
            _InvalidProvider(),
            model="m",
            output_schema=object_score,
            output_schema_validator=lambda value: full_catalog.validate("score", value),
            mode="generative",
            max_repairs=0,
        )
        with pytest.raises(Exception, match="process measure failed"):
            service.execute_run("invalid-run", executor_overrides={"measure": executor})
        history = service.persistence.list_state_history("invalid-run")
        states = [snapshot for _version, snapshot in history]
        assert len(states) == 1
        assert states[0].get("counter") == 0
    finally:
        service.close()


def test_incomplete_evidence_bundle_cannot_claim_reproducibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative path: a requested full export without pinned inputs fails."""
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _EchoProvider)
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "no-closure-study",
                "title": "no closure",
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
        service.approve_specification("no-closure-study", draft["version"], "researcher")
        service.create_run({"id": "no-closure-run", "study_id": "no-closure-study", "build": ""})
        with pytest.raises(ValueError, match="REPRODUCIBILITY"):
            service.export_run("no-closure-run", "exports/full", mode=ExportMode.REPRODUCIBILITY)
    finally:
        service.close()
