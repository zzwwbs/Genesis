from __future__ import annotations

import json
from pathlib import Path

from genesis.compiler import StudyCompiler
from genesis.runtime import expand_protocol_conditions

PACKAGE = Path(__file__).parent / "golden_studies" / "clickbait_mini"


def test_clickbait_mini_is_a_canonical_executable_package(tmp_path) -> None:
    build = StudyCompiler(PACKAGE).compile(tmp_path / "build")

    processes = json.loads((build.path / "processes.json").read_text())
    protocol = json.loads((build.path / "protocol.json").read_text())
    modes = {process["executor"]["mode"] for process in processes}

    assert {
        "generative",
        "semantic-evaluator",
        "rule",
        "computational",
        "stochastic",
        "state-transition",
    }.issubset(modes)
    assert len(expand_protocol_conditions(protocol)) == 6
    assert protocol["time_model"] == {"type": "rounds", "start": 1, "end": 6, "step": 1}
    assert protocol["phases"][1] == {"id": "treatment", "start": 4, "end": 6}


def test_clickbait_mini_declares_information_boundaries_and_longitudinal_state(
    tmp_path,
) -> None:
    build = StudyCompiler(PACKAGE).compile(tmp_path / "build")
    policies = {
        policy["id"]: policy
        for policy in json.loads((build.path / "context_policies.json").read_text())
    }
    state = {item["id"]: item for item in json.loads((build.path / "state_model.json").read_text())}

    assert "state.latent-strategies" not in policies["detector-context"]["allow"]
    assert policies["full-article-context"]["available_when"]["inputs"] == {
        "event": "article-selected",
        "recipient_match": True,
        "source_match_event": True,
    }
    assert policies["title-context"]["cardinality"]["inputs"] == 3
    assert {"follows", "creator-memory", "revenue"}.issubset(state)
