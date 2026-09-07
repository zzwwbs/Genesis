"""AW-14: complete canonical package preservation and round-trip fidelity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from genesis.service import GenesisService
from genesis.specification import (
    DomainSpec,
    ModelsSpec,
    OpennessSpec,
    OutcomesSpec,
    ProtocolSpec,
    StudySpec,
    TheorySpec,
)

FULL_PAYLOAD = {
    "id": "roundtrip-study",
    "title": "Round-trip study",
    "description": "Exercises every canonical field.",
    "owners": ["researcher-a"],
    "source_citations": ["doi:10.1000/example"],
    "artifact_refs": ["artifact-ref-1"],
    "package_compatibility": {"min_genesis_version": "0.1.0"},
    "origin": {"origin": "researcher", "evidence_refs": ["decision-018"]},
    "approval": {
        "status": "confirmed",
        "confirmed_by": "researcher",
        "confirmed_at": "2026-08-30T09:00:00Z",
    },
    "extensions": {"example.namespace": {"note": "extension value"}},
    "processes": [
        {
            "id": "formulate",
            "name": "Formulate strategy",
            "actors": ["creator"],
            "trigger": {"type": "phase", "phase": 0},
            "dependencies": {"after": [], "delay": {"rounds": 1}},
            "openness_rationale": "strategy form is focal",
            "closure_rationale": "a menu would remove the variation",
            "executor": {"mode": "deterministic", "parameters": {"step": 1}},
            "context_policy": "public",
            "prompt_ref": None,
            "inputs": ["creator-strategy"],
            "outputs": [{"artifact_type": "strategy", "schema_ref": "creator-strategy"}],
            "state_effects": [{"field": "retained-strategy", "op": "set"}],
            "trace_policy": {"record_context": True, "record_raw_response": False},
            "retry_policy": {"max_attempts": 2},
            "origin": {"origin": "researcher", "evidence_refs": ["decision-001"]},
        },
        {"id": "evaluate", "executor": {"mode": "deterministic"}, "context_policy": "public"},
        {"id": "settle", "executor": {"mode": "deterministic"}, "context_policy": "public"},
        {"id": "loop-feedback", "executor": {"mode": "deterministic"}, "context_policy": "public"},
    ],
    "theory": {
        "theory_family": "variation-selection-retention",
        "constructs": [{"id": "creator-strategy", "theory_role": "variation_unit"}],
        "process_mappings": [
            {"process": "formulate", "theory_function": "variation"},
            {"process": "evaluate", "theory_function": "selection"},
            {"process": "settle", "theory_function": "retention"},
            {"process": "loop-feedback", "theory_function": "feedback"},
        ],
        "relations": [{"from": "formulate", "to": "evaluate", "relation": "produces_variant"}],
        "feedback": [{"from": "performance-history", "to": "formulate", "relation": "conditions"}],
        "delays": [{"process": "formulate", "rounds": 2}],
        "observables": [{"id": "strategy-diversity", "definition": "distinct strategies"}],
    },
    "domain": {
        "actors": [{"id": "creator", "theory_role": "variation_producer"}],
        "attributes": [{"id": "audience-size", "value_type": "integer"}],
        "states": [
            {"id": "retained-strategy", "persistence": "across_rounds"},
            {"id": "performance-history", "persistence": "across_rounds"},
        ],
        "artifacts": [{"id": "creator-strategy", "artifact_type": "strategy"}],
        "mechanisms": [{"id": "platform-recommendation", "implements": "variation"}],
        "institutions": [{"id": "moderation-policy", "type": "rule_set"}],
        "initialization": {"mode": "researcher"},
        "visibility": [{"id": "public", "allow": ["retained-strategy"]}],
        "availability": [{"path": "retained-strategy", "available_when": {"after_round": 1}}],
        "updates": [{"state": "retained-strategy", "op": "set"}],
        "persistence": [{"object": "creator-strategy", "policy": "retain_with_lineage"}],
    },
    "protocol": {
        "time_model": {"type": "rounds", "start": 0, "end": 10, "step": 1},
        "termination": [{"condition": "max_rounds"}],
        "conditions": [{"id": "base"}, {"id": "alternative"}],
        "replications": 3,
        "matching": {"mode": "none"},
        "random_streams": [{"id": "stream-a", "seed": 42}],
        "model_freezing": True,
        "budgets": {"max_calls": 100},
        "checkpoints": {"interval": 5},
        "replay_retention": {"mode": "full"},
    },
    "outcomes": [
        {
            "id": "strategy-count",
            "source": "events",
            "filters": [{"process_id": "formulate"}],
            "grouping": [],
            "aggregation": {"type": "count", "field": "outputs"},
            "missingness": {"policy": "exclude"},
            "output_schema": "outcome-schema",
        }
    ],
    "models": [
        {
            "id": "creator-model",
            "provider": "mock",
            "model": "mock-1",
            "capabilities": ["structured_output"],
        }
    ],
}

MODELS = {
    "study": StudySpec,
    "openness": OpennessSpec,
    "theory": TheorySpec,
    "domain": DomainSpec,
    "protocol": ProtocolSpec,
    "outcomes": OutcomesSpec,
    "models": ModelsSpec,
}


def test_full_payload_round_trips_without_field_loss(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        created = service.create_specification(FULL_PAYLOAD)
        assert created["status"] == "draft"
        directory = workspace / ".genesis/specifications/roundtrip-study"
        expected = service._canonical_specification(FULL_PAYLOAD)
        for name, model in MODELS.items():
            parsed = yaml.safe_load((directory / f"{name}.yaml").read_text())
            assert parsed == expected[name], f"{name}.yaml lost fields"
            model.model_validate(parsed)  # canonical schema still accepts the round trip
    finally:
        service.close()


def test_unknown_form_field_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        with pytest.raises(ValueError, match="INVALID_FIELD.*bogus_field"):
            service.create_specification({"id": "bad", "title": "t", "bogus_field": {"x": 1}})
    finally:
        service.close()


def test_origin_approval_and_package_compatibility_survive_lifecycle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(FULL_PAYLOAD)
        updated = service.update_specification(
            "roundtrip-study", {"title": "Renamed"}, draft["version"]
        )
        assert updated["form"]["origin"] == FULL_PAYLOAD["origin"]
        assert updated["form"]["approval"] == FULL_PAYLOAD["approval"]
        assert updated["form"]["package_compatibility"] == FULL_PAYLOAD["package_compatibility"]
        approved = service.approve_specification(
            "roundtrip-study", updated["version"], "researcher"
        )
        assert approved["status"] == "approved"
        compiled = service.compile_study(
            None, "builds/roundtrip-study", specification_id="roundtrip-study"
        )
        build_root = workspace / compiled["path"]
        processes = json.loads((build_root / "processes.json").read_text())
        assert processes[0]["origin"]["origin"] == "researcher"
        assert processes[0]["origin"]["evidence_refs"] == ["decision-001"]
        assert processes[0]["closure_rationale"] == "a menu would remove the variation"
        assert processes[0]["trace_policy"]["record_raw_response"] is False
    finally:
        service.close()


def test_direct_yaml_and_guided_authoring_converge(tmp_path: Path) -> None:
    """AST-005 / ACC-002: the same package authored both ways yields equal canonical models."""
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        service.create_specification(FULL_PAYLOAD)
        guided_dir = workspace / ".genesis/specifications/roundtrip-study"
        guided: dict[str, Any] = {
            name: yaml.safe_load((guided_dir / f"{name}.yaml").read_text()) for name in MODELS
        }
        # Re-interpret the guided files as a direct-YAML package directory.
        direct_dir = workspace / "direct"
        direct_dir.mkdir(exist_ok=True)
        from genesis.compiler import StudyCompiler

        for name, value in guided.items():
            (direct_dir / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
        direct_build = StudyCompiler(direct_dir).compile(workspace / "direct-build")
        guided_build = StudyCompiler(guided_dir).compile(workspace / "guided-build")
        assert direct_build.build_hash == guided_build.build_hash
    finally:
        service.close()


def test_process_round_trip_extensions_namespace(tmp_path: Path) -> None:
    """Extension namespaces are preserved into every canonical artifact."""
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        service.create_specification(FULL_PAYLOAD)
        directory = workspace / ".genesis/specifications/roundtrip-study"
        study = yaml.safe_load((directory / "study.yaml").read_text())
        assert study["extensions"] == {"example.namespace": {"note": "extension value"}}
        openness = yaml.safe_load((directory / "openness.yaml").read_text())
        assert openness["extensions"] == {"example.namespace": {"note": "extension value"}}
    finally:
        service.close()
