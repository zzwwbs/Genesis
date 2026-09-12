"""Engine gaps found by mapping the clickbait design onto GENESIS (G1-G8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from genesis.execution_manifest import resolve_execution_manifest, scientific_config_digest
from genesis.replay import ReplayMode
from genesis.service import GenesisService

# --- helpers ----------------------------------------------------------------------


def _write_package(root: Path, files: dict[str, dict[str, Any]], study_id: str) -> Path:
    source = root / "pkg"
    source.mkdir(parents=True, exist_ok=True)
    base = {"schema_version": "1.0", "study_id": study_id}
    defaults: dict[str, dict[str, Any]] = {
        "study": {"title": study_id},
        "theory": {"theory_family": "exploratory"},
        "domain": {},
        "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {"outcomes": []},
        "models": {"models": []},
    }
    for name, value in {**defaults, **files}.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump({**base, **value}, sort_keys=False))
    return source


@pytest.fixture
def study_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A study-code module on sys.path whose source a test can change."""
    directory = tmp_path / "code"
    directory.mkdir()
    name = f"g1_study_code_{abs(hash(tmp_path)) % 10**8}"
    path = directory / f"{name}.py"
    path.write_text(
        "def tick(invocation):\n"
        "    current = invocation.context.data.get('counter', 0) if invocation.context else 0\n"
        "    return {'counter': current + 1}\n"
    )
    monkeypatch.syspath_prepend(str(directory))
    yield name, path
    sys.modules.pop(name, None)


def _code_package(workspace: Path, module: str) -> Path:
    return _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    {
                        "id": "tick",
                        "executor": {
                            "mode": "computational",
                            "parameters": {"entry_point": f"{module}:tick"},
                        },
                        "context_policy": "s",
                        "trigger": {"type": "phase", "phase": 0, "repeat": True},
                        "state_effects": [{"field": "counter", "op": "set"}],
                    }
                ]
            },
            "domain": {
                "visibility": [{"id": "s", "allow": ["counter"]}],
                "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
            },
        },
        "code-study",
    )


# --- G1: study code is pinned by the run record ------------------------------------


def test_the_run_manifest_records_the_study_code_it_runs(tmp_path: Path, study_module) -> None:
    module, path = study_module
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        build = service.compile_study(_code_package(workspace, module), "builds/code")["path"]
        service.create_run({"id": "r", "study_id": "code-study", "build": build})
        service.execute_run("r")
        manifest = service.get_run("r")["manifest"]
        import hashlib

        record = manifest["executor_code"]["tick"]
        assert record["reference"] == f"{module}:tick"
        assert record["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["execution"]["executor_code_digest"]
    finally:
        service.close()


def test_the_code_digest_changes_the_scientific_digest_only_when_present() -> None:
    base: dict[str, Any] = dict(
        condition_id="base",
        factors={},
        replication=1,
        build_manifest={"build_hash": "b"},
        protocol={},
        package_closure_digest="c",
        protocol_digest="p",
        model_configuration_digest="m",
        outcome_plan_digest="o",
    )
    without = resolve_execution_manifest(**base)
    assert "executor_code_digest" not in without
    with_code = resolve_execution_manifest(**base, executor_code_digest="x")
    other_code = resolve_execution_manifest(**base, executor_code_digest="y")
    assert scientific_config_digest(with_code) != scientific_config_digest(without)
    assert scientific_config_digest(with_code) != scientific_config_digest(other_code)


def test_resuming_after_the_study_code_changed_is_refused(tmp_path: Path, study_module) -> None:
    module, path = study_module
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        build = service.compile_study(_code_package(workspace, module), "builds/code")["path"]
        run = service.create_run({"id": "r", "study_id": "code-study", "build": build})
        run = service._attach_run_manifest(run)
        service.transition_run("r", "paused", run["version"])
        path.write_text(path.read_text() + "\n# the settlement rule changed\n")
        with pytest.raises(ValueError, match="RUN_CODE_CHANGED"):
            service.execute_run("r")
        assert service.get_run("r")["status"] == "paused"
    finally:
        service.close()


def test_replaying_under_changed_study_code_is_refused(tmp_path: Path, study_module) -> None:
    module, path = study_module
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        build = service.compile_study(_code_package(workspace, module), "builds/code")["path"]
        service.create_run({"id": "r", "study_id": "code-study", "build": build})
        assert service.execute_run("r")["status"] == "completed"
        path.write_text(path.read_text() + "\n# changed after the run\n")
        with pytest.raises(ValueError, match="REPLAY_CODE_CHANGED"):
            service.replay_run("r", mode=ReplayMode.FULL)
    finally:
        service.close()


# --- G5: model-call decisions write shared state under simultaneous timing --------


def test_state_store_applies_keyed_and_relation_operations() -> None:
    from genesis.runtime import StateStore

    store = StateStore({"picks": dict, "log": dict, "follows": list}, {"picks": {}, "log": {}})
    declared = {"picks", "log", "follows"}
    store.apply(
        [
            {"field": "picks", "op": "put", "key": "u1", "value": ["a"]},
            {"field": "picks", "op": "put", "key": "u2", "value": ["b"]},
            {"field": "log", "op": "append", "key": "u1", "value": "first"},
            {"field": "log", "op": "append", "key": "u1", "value": "second"},
            {"field": "follows", "op": "add-relation", "value": {"user": "u1", "creator": "w1"}},
            {"field": "follows", "op": "add-relation", "value": {"user": "u1", "creator": "w1"}},
            {"field": "follows", "op": "add-relation", "value": {"user": "u2", "creator": "w1"}},
            {"field": "follows", "op": "remove-relation", "value": {"user": "u2", "creator": "w1"}},
        ],
        declared,
    )
    state = store.snapshot()
    assert state["picks"] == {"u1": ["a"], "u2": ["b"]}
    assert state["log"] == {"u1": ["first", "second"]}
    assert state["follows"] == [{"user": "u1", "creator": "w1"}]


def test_model_call_effects_read_declared_outputs_under_the_actor_key() -> None:
    from genesis.runtime import model_call_effects

    process = {
        "id": "choose",
        "state_effects": [
            {"field": "picks", "op": "put", "key": "actor", "from": "choice.ids"},
            {"field": "notes", "op": "append", "from": "choice.note"},
            {"field": "missing", "op": "append", "from": "choice.absent"},
        ],
    }
    effects = model_call_effects(process, {"choice": {"ids": ["a"], "note": "n"}}, ("u1",))
    assert effects == [
        {"field": "picks", "op": "put", "value": ["a"], "key": "u1"},
        {"field": "notes", "op": "append", "value": "n"},
    ]
    with pytest.raises(ValueError, match="STATE_EFFECT_KEY"):
        model_call_effects(process, {"choice": {"ids": []}}, ("u1", "u2"))


class _ChoiceProvider:
    provider = "openai-compatible"

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def generate(self, request: Any) -> Any:
        import json

        from genesis.providers import ProviderResponse

        actor = request.prompt.split("ACTOR=")[1].split()[0]
        value = {"ids": [f"article-{actor}"], "note": f"seen by {actor}"}
        text = json.dumps(value)
        return ProviderResponse(text, self.provider, request.model, "r", parsed=value)


def _choice_package(workspace: Path, effects: list[dict[str, Any]]) -> Path:
    source = _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    {
                        "id": "choose",
                        "actors": ["u1", "u2", "u3"],
                        "information_timing": {"mode": "simultaneous", "order": "shuffled"},
                        "openness_rationale": "which articles a user opens is the phenomenon",
                        "closure_rationale": "the choice is closed by the declared schema",
                        "executor": {"mode": "generative", "model_profile": "mp"},
                        "context_policy": "sees-picks",
                        "prompt_ref": "choose",
                        "trigger": {"type": "phase", "phase": 0, "repeat": True},
                        "outputs": [{"artifact_type": "choice", "schema_ref": "choice"}],
                        "state_effects": effects,
                    }
                ]
            },
            "domain": {
                "visibility": [{"id": "sees-picks", "allow": ["picks", "notes"]}],
                "states": [
                    {"id": "picks", "value_type": "object", "initial": {}},
                    {"id": "notes", "value_type": "array", "initial": []},
                ],
                "artifacts": [{"id": "choice", "artifact_type": "choice", "schema_ref": "choice"}],
            },
            "models": {"models": [{"id": "mp", "provider": "openai-compatible", "model": "m1"}]},
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 0}},
        },
        "choice-study",
    )
    (source / "prompts").mkdir(exist_ok=True)
    (source / "prompts" / "choose.txt").write_text("ACTOR={actor_ids} context {context}")
    (source / "schemas").mkdir(exist_ok=True)
    (source / "schemas" / "choice.yaml").write_text(
        "type: object\nproperties:\n  ids: {type: array, items: {type: string}}\n"
        "  note: {type: string}\nrequired: [ids, note]\n"
    )
    return source


def test_simultaneous_model_calls_compose_their_declared_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _ChoiceProvider)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_G5_KEY",
            }
        )
        source = _choice_package(
            workspace,
            [
                {"field": "picks", "op": "put", "key": "actor", "from": "choice.ids"},
                {"field": "notes", "op": "append", "from": "choice.note"},
            ],
        )
        build = service.compile_study(source, "builds/choice")["path"]
        service.create_run({"id": "r", "study_id": "choice-study", "build": build})
        assert service.execute_run("r")["status"] == "completed"
        state = list(service.persistence.list_state_history("r"))[-1][1]
        assert state["picks"] == {
            "u1": ["article-u1"],
            "u2": ["article-u2"],
            "u3": ["article-u3"],
        }
        assert sorted(state["notes"]) == ["seen by u1", "seen by u2", "seen by u3"]
    finally:
        service.close()


def test_a_simultaneous_model_call_that_sets_a_field_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        source = _choice_package(workspace, [{"field": "picks", "op": "set", "from": "choice"}])
        with pytest.raises(Exception, match="SIMULTANEOUS_WRITE_CONFLICT"):
            service.compile_study(source, "builds/choice")
    finally:
        service.close()


@pytest.mark.parametrize(
    ("effect", "message"),
    [
        ({"field": "picks", "op": "put", "from": "choice.ids"}, "requires key: actor"),
        ({"field": "picks", "op": "put", "key": "actor", "from": "other"}, "reads output"),
        ({"field": "notes", "op": "merge", "from": "choice"}, "uses op 'merge'"),
        ({"field": "nowhere", "op": "append", "from": "choice"}, "does not declare"),
        ({"field": "notes", "op": "put", "key": "actor", "from": "choice"}, "must be an object"),
    ],
)
def test_invalid_model_call_effects_are_refused_at_compile(
    tmp_path: Path, effect: dict[str, Any], message: str
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        source = _choice_package(workspace, [effect])
        with pytest.raises(Exception, match="STATE_EFFECT_INVALID") as raised:
            service.compile_study(source, "builds/choice")
        assert message in str(raised.value)
    finally:
        service.close()


# --- G2: conditionally scheduled producers do not block their consumers ------------


def _ready_ids(scheduler: Any, phase: int, state: dict[str, Any]) -> list[str]:
    return [item.process_id for item in scheduler.ready(phase, state=state)]


def _scheduler_state(phase: int, **condition: Any) -> dict[str, Any]:
    return {"condition": dict(condition), "protocol": {"phase": phase}}


def test_a_consumer_runs_in_phases_its_conditional_producer_skips() -> None:
    from genesis.runtime import Scheduler

    scheduler = Scheduler(
        [
            {
                "id": "reflect",
                "trigger": {
                    "type": "condition",
                    "repeat": True,
                    "predicate": {"path": "protocol.phase", "op": "in", "value": [1, 3]},
                },
            },
            {
                "id": "publish",
                "trigger": {"type": "phase", "phase": 0, "repeat": True},
                "dependencies": {"after": ["reflect"]},
            },
        ]
    )
    # Phase 0: reflection does not run, so publishing is not held back.
    assert _ready_ids(scheduler, 0, _scheduler_state(0)) == ["publish"]
    scheduler.complete("publish", 0)
    # Phase 1: reflection runs first; publishing waits for it.
    assert _ready_ids(scheduler, 1, _scheduler_state(1)) == ["reflect"]
    scheduler.complete("reflect", 1)
    assert _ready_ids(scheduler, 1, _scheduler_state(1)) == ["publish"]
    scheduler.complete("publish", 1)
    # Phase 2: skipped again, and the earlier completion does not stand in.
    assert _ready_ids(scheduler, 2, _scheduler_state(2)) == ["publish"]


def test_condition_triggers_accept_compound_predicates() -> None:
    from genesis.runtime import Scheduler

    scheduler = Scheduler(
        [
            {
                "id": "sanction",
                "trigger": {
                    "type": "condition",
                    "repeat": True,
                    "predicate": {
                        "all": [
                            {"path": "condition.governance", "op": "eq", "value": "hidden"},
                            {"path": "protocol.phase", "op": "gte", "value": 20},
                        ]
                    },
                },
            }
        ]
    )
    assert _ready_ids(scheduler, 19, _scheduler_state(19, governance="hidden")) == []
    assert _ready_ids(scheduler, 20, _scheduler_state(20, governance="none")) == []
    assert _ready_ids(scheduler, 20, _scheduler_state(20, governance="hidden")) == ["sanction"]
    with pytest.raises(ValueError, match="non-empty list"):
        Scheduler([{"id": "bad", "trigger": {"type": "condition", "predicate": {"any": []}}}])


# --- G6: context joins and projections ---------------------------------------------


def _context(policy: dict[str, Any], state: dict[str, Any], actor: str, **inputs: Any) -> Any:
    from genesis.runtime import ContextEngine, ProcessInvocation

    engine = ContextEngine({"p": policy})
    invocation = ProcessInvocation("inv", "run", "proc", actor_ids=(actor,), inputs=inputs)
    return engine.build("p", invocation, state).data


FEED_STATE: dict[str, Any] = {
    "feeds": {
        "u1": [{"article": "a1", "author": "w1"}, {"article": "a2", "author": "w2"}],
        "u2": [{"article": "a3", "author": "w3"}],
    },
    "impressions": [
        {"user": "u1", "creator": "w1", "text": "warm"},
        {"user": "u1", "creator": "w3", "text": "old"},
        {"user": "u2", "creator": "w3", "text": "dull"},
    ],
    "selected": {"u1": ["a2"]},
    "board": [
        {"author": "w1", "title": "t1", "clicks": 3, "strategy_summary": "secret"},
        {"author": "w2", "title": "t2", "clicks": 9, "strategy_summary": "secret"},
    ],
}


def test_a_scope_can_join_through_a_wildcard_selector() -> None:
    # A user's impressions of the authors in their own feed, and no others.
    data = _context(
        {
            "allow": ["impressions"],
            "scope": {"impressions": {"field": "creator", "in": "state.feeds.${actor}.*.author"}},
        },
        FEED_STATE,
        "u1",
    )
    assert [row["text"] for row in data["impressions"]] == ["warm"]


def test_a_scope_can_filter_on_a_nested_field() -> None:
    # Only the bodies of articles this user selected, from all article inputs.
    articles = {
        "art-1": {"artifact_type": "article", "value": {"article_id": "a1", "body": "one"}},
        "art-2": {"artifact_type": "article", "value": {"article_id": "a2", "body": "two"}},
    }
    data = _context(
        {
            "allow": ["inputs"],
            "scope": {"inputs": {"field": "value.article_id", "in": "state.selected.${actor}"}},
        },
        FEED_STATE,
        "u1",
        **articles,
    )
    assert list(data["inputs"]) == ["art-2"]


def test_a_projection_hands_over_part_of_each_record() -> None:
    dropped = _context(
        {"allow": ["board"], "project": {"board": {"drop": ["strategy_summary"]}}},
        FEED_STATE,
        "w1",
    )
    assert all("strategy_summary" not in row for row in dropped["board"])
    capped = _context(
        {
            "allow": ["board"],
            "cardinality": {"board": {"limit": 1, "keep": "last", "by": "clicks"}},
            "project": {"board": {"keep": ["title"]}},
        },
        FEED_STATE,
        "w1",
    )
    assert [dict(row) for row in capped["board"]] == [{"title": "t2"}]


def test_malformed_projections_are_refused_at_compile(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _write_package(
        workspace,
        {
            "openness": {"processes": []},
            "domain": {
                "states": [{"id": "board", "value_type": "array", "initial": []}],
                "visibility": [
                    {
                        "id": "p",
                        "allow": ["board"],
                        "project": {"board": {"keep": [], "drop": ["x"]}, "other": {"keep": ["x"]}},
                    }
                ],
            },
        },
        "project-study",
    )
    service = GenesisService(workspace)
    try:
        with pytest.raises(Exception, match="CONTEXT_PROJECT_INVALID") as raised:
            service.compile_study(source, "builds/project")
        assert "does not allow" in str(raised.value)
        assert "exactly one of keep or drop" in str(raised.value)
    finally:
        service.close()


def test_a_policy_without_projections_serializes_unchanged() -> None:
    from genesis.specification.models import VisibilitySpec

    assert "project" not in VisibilitySpec(id="p", allow=["x"]).model_dump(mode="json")


# --- G4: a measurement steers behaviour only where the design says so --------------

DETECT: dict[str, Any] = {
    "id": "detect",
    "measurement": True,
    "executor": {"mode": "semantic-evaluator", "model_profile": "mp"},
    "context_policy": "detector",
    "outputs": [{"artifact_type": "detection", "schema_ref": "detection"}],
    "state_effects": [{"field": "scores", "op": "append", "from": "detection"}],
}


def _settle(**extra: Any) -> dict[str, Any]:
    return {
        "id": "settle",
        "executor": {"mode": "computational", "parameters": {"entry_point": "m:f"}},
        "context_policy": "settle",
        "inputs": ["detection"],
        **extra,
    }


def _diagnose(processes: list[dict[str, Any]], policies: dict[str, Any] | None = None) -> Any:
    from genesis.measurement import measurement_diagnostics

    return measurement_diagnostics(processes, policies or {"detector": {}, "settle": {}})


def test_reading_a_measurement_without_declaring_it_is_refused() -> None:
    errors, _ = _diagnose([DETECT, _settle()])
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]
    assert "input 'detection'" in errors[0]["message"]


def test_a_declared_use_admits_the_measurement() -> None:
    use = {"source": "detect", "rationale": "the detector's score sets the penalty"}
    errors, warnings = _diagnose([DETECT, _settle(measurement_use=[use])])
    assert errors == [] and warnings == []


def test_reading_a_measurement_state_field_is_refused_too() -> None:
    reader = {
        "id": "publish",
        "executor": {"mode": "generative", "model_profile": "mp"},
        "context_policy": "creator",
    }
    errors, _ = _diagnose([DETECT, reader], {"detector": {}, "creator": {"allow": ["scores"]}})
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]
    assert "state 'scores'" in errors[0]["message"]


@pytest.mark.parametrize(
    ("use", "message"),
    [
        ({"source": "nobody", "rationale": "r"}, "is not a measurement process"),
        (
            {"source": "detect", "rationale": "r", "when": {"any": []}},
            "non-empty list",
        ),
    ],
)
def test_invalid_measurement_uses_are_refused(use: dict[str, Any], message: str) -> None:
    errors, _ = _diagnose([DETECT, _settle(measurement_use=[use])])
    assert any(error["code"] == "MEASUREMENT_USE_INVALID" for error in errors)
    assert any(message in error["message"] for error in errors)


def test_a_conditional_use_may_not_gate_a_state_read() -> None:
    use = {
        "source": "detect",
        "rationale": "r",
        "when": {"path": "condition.governance", "op": "eq", "value": "hidden"},
    }
    errors, _ = _diagnose(
        [DETECT, _settle(measurement_use=[use])],
        {"detector": {}, "settle": {"allow": ["scores"]}},
    )
    assert [error["code"] for error in errors] == ["MEASUREMENT_USE_INVALID"]
    assert "cannot be withheld per condition" in errors[0]["message"]


def test_a_declared_use_that_reads_nothing_is_reported() -> None:
    unrelated = {
        "id": "settle",
        "executor": {"mode": "computational", "parameters": {"entry_point": "m:f"}},
        "context_policy": "settle",
        "measurement_use": [{"source": "detect", "rationale": "r"}],
    }
    errors, warnings = _diagnose([DETECT, unrelated])
    assert errors == []
    assert [warning["code"] for warning in warnings] == ["MEASUREMENT_USE_UNUSED"]


def test_a_conditional_use_withholds_the_measurement_outside_its_condition() -> None:
    from genesis.measurement import gated_input_refs

    process = _settle(
        measurement_use=[
            {
                "source": "detect",
                "rationale": "the penalty applies only under governance, from round 20",
                "when": {
                    "all": [
                        {"path": "condition.governance", "op": "eq", "value": "hidden"},
                        {"path": "protocol.phase", "op": "gte", "value": 20},
                    ]
                },
            }
        ]
    )
    processes = {"detect": DETECT, "settle": process}
    governed = {"governance": "hidden"}
    assert gated_input_refs(process, processes, phase=20, condition=governed) == ["detection"]
    assert gated_input_refs(process, processes, phase=19, condition=governed) == []
    assert gated_input_refs(process, processes, phase=20, condition={"governance": "none"}) == []


# --- G3: prompt roles and named context slots --------------------------------------


def test_a_template_without_markers_is_all_user() -> None:
    from genesis.providers import split_prompt_roles

    assert split_prompt_roles("just a prompt") == (None, "just a prompt")


def test_role_markers_split_a_template_into_messages() -> None:
    from genesis.providers import split_prompt_roles

    system, user = split_prompt_roles("[system]\nYou run a page.\n[user]\nRound 3.\n")
    assert system == "You run a page."
    assert user == "Round 3."


@pytest.mark.parametrize(
    "template",
    [
        "preamble\n[user]\nx",
        "[user]\na\n[user]\nb",
        "[system]\nonly instructions",
    ],
)
def test_malformed_role_templates_are_refused(template: str) -> None:
    from genesis.providers import split_prompt_roles

    with pytest.raises(ValueError, match="PROMPT_TEMPLATE"):
        split_prompt_roles(template)


def test_named_slots_render_parts_of_the_context() -> None:
    from genesis.providers import render_prompt

    context = {"performance": {"clicks": 4}, "reflection": "users tire of warm stories"}
    system, user = render_prompt(
        "[system]\nYou are {actor_ids}.\n[user]\nRound {phase}.\n"
        "Last round: {context.performance}\nYour note: {context.reflection}\n"
        "Missing: {context.nothing}",
        context,
        ("w3",),
        7,
    )
    assert system == "You are w3."
    assert '{"clicks": 4}' in user
    assert "Your note: users tire of warm stories" in user
    assert "Missing: null" in user
    assert user.startswith("Round 7.")


def test_crlf_role_markers_still_split_a_template() -> None:
    from genesis.providers import split_prompt_roles

    system, user = split_prompt_roles("[system]\r\nRules.\r\n[user]\r\nAsk.\r\n")
    assert system.strip() == "Rules." and user.strip() == "Ask."


def test_a_malformed_prompt_template_is_refused_at_compile(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _choice_package(
        workspace, [{"field": "picks", "op": "put", "key": "actor", "from": "choice.ids"}]
    )
    (source / "prompts" / "choose.txt").write_text("preamble\n[user]\nACTOR={actor_ids}")
    service = GenesisService(workspace)
    try:
        with pytest.raises(Exception, match="PROMPT_TEMPLATE_INVALID"):
            service.compile_study(source, "builds/choice")
    finally:
        service.close()


def test_a_system_message_reaches_the_provider_and_the_recorded_digest() -> None:
    from genesis.providers import ProviderExecutor, ProviderRequest, ProviderResponse
    from genesis.runtime import ProcessInvocation

    seen: list[ProviderRequest] = []

    class _Recorder:
        provider = "test"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            seen.append(request)
            return ProviderResponse("{}", self.provider, request.model, "r", parsed={})

    invocation = ProcessInvocation("i", "run", "choose", actor_ids=("w1",), context={"a": 1})
    executor = ProviderExecutor(
        _Recorder(), model="m", prompt_template="[system]\nStanding rules.\n[user]\n{context}"
    )
    result = executor.execute(invocation)
    assert seen[0].system == "Standing rules."
    assert seen[0].prompt == '{"a": 1}'
    assert seen[0].content_digest() != ProviderRequest("m", seen[0].prompt).content_digest()
    assert result.metadata["provider_attempts"][0]["prompt_hash"] == seen[0].content_digest()


def test_every_attempt_records_the_digest_of_what_was_sent() -> None:
    from genesis.providers import ProviderExecutor, ProviderRequest, ProviderResponse
    from genesis.runtime import ProcessInvocation

    seen: list[ProviderRequest] = []

    class _RepairingProvider:
        provider = "test"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            seen.append(request)
            value = {"text": "ok"} if len(seen) > 1 else {}
            return ProviderResponse(
                json.dumps(value), self.provider, request.model, "r", parsed=value
            )

    executor = ProviderExecutor(
        _RepairingProvider(),
        model="m",
        prompt_template="[system]\nRules.\n[user]\n{context}",
        output_schema={"type": "object", "required": ["text"]},
    )
    result = executor.execute(ProcessInvocation("i", "run", "p", actor_ids=("w1",), context={}))
    attempts = result.metadata["provider_attempts"]
    assert len(attempts) == 2 and len(seen) == 2
    # The repair carries the system message too, so it must be hashed the same way.
    assert [attempt["prompt_hash"] for attempt in attempts] == [
        request.content_digest() for request in seen
    ]
    assert all(request.system == "Rules." for request in seen)


# --- G3: an actor's own earlier exchanges as context --------------------------------


def test_a_policy_admits_an_actor_s_own_exchanges_and_can_cap_them() -> None:
    from genesis.runtime import ContextEngine, ProcessInvocation

    history = [
        {
            "phase": phase,
            "attempt": 1,
            "context": {"round": phase},
            "outputs": {"title": f"t{phase}"},
        }
        for phase in range(4)
    ]
    engine = ContextEngine(
        {
            "recall": {
                "allow": ["exchanges.publish"],
                "cardinality": {"exchanges.publish": {"limit": 2, "keep": "last", "by": "phase"}},
            }
        }
    )
    invocation = ProcessInvocation(
        "i", "run", "recall", actor_ids=("w1",), exchanges={"publish": history}
    )
    data = engine.build("recall", invocation, {}).data
    kept = [dict(entry) for entry in data["exchanges"]["publish"]]
    assert [entry["phase"] for entry in kept] == [2, 3]
    assert kept[-1]["outputs"]["title"] == "t3"


def test_exchanges_are_recorded_per_actor_and_rebuilt_after_a_resume() -> None:
    controller = _controller_with_log({"publish": {}})

    class _Persistence:
        @staticmethod
        def list_artifacts(_run_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "artifact_id": "run-publish-w1-0-attempt-1",
                    "payload": json.dumps({"outputs": {"title": "first"}}),
                },
                {
                    "artifact_id": "run-publish-w2-0-attempt-1",
                    "payload": json.dumps({"outputs": {"title": "other"}}),
                },
            ]

    controller.persistence = _Persistence()
    events = [
        {
            "event_id": "run-publish-w1-0-attempt-1",
            "kind": "process_completed",
            "process_id": "publish",
            "actors": ["w1"],
            "phase": 0,
            "attempt": 1,
            "commit_order": 1,
            "context": {"followers": 4},
        },
        {
            "event_id": "run-publish-w2-0-attempt-1",
            "kind": "process_completed",
            "process_id": "publish",
            "actors": ["w2"],
            "phase": 0,
            "attempt": 1,
            "commit_order": 2,
            "context": {"followers": 40},
        },
    ]
    controller._restore_exchange_log("run", events)
    assert controller._exchange_log[("publish", ("w1",))] == [
        {"phase": 0, "attempt": 1, "outputs": {"title": "first"}, "context": {"followers": 4}}
    ]
    # One actor's exchanges never reach another's turn.
    assert [
        entry["outputs"]
        for entry in controller._exchanges_for(_turn("publish", ("w1",)))["publish"]
    ] == [{"title": "first"}]


def test_a_later_round_is_shown_what_this_actor_answered_earlier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    class _EchoProvider:
        provider = "openai-compatible"

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def generate(self, request: Any) -> Any:
            from genesis.providers import ProviderResponse

            actor = request.prompt.split("ACTOR=")[1].split()[0]
            if "RECALL" in request.prompt:
                prompts.append(request.prompt)
            value = {"ids": [f"article-{actor}"], "note": f"seen by {actor}"}
            return ProviderResponse(
                json.dumps(value), self.provider, request.model, "r", parsed=value
            )

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _EchoProvider)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _choice_package(
        workspace, [{"field": "picks", "op": "put", "key": "actor", "from": "choice.ids"}]
    )
    # A second round, and a process that reads its own earlier exchanges.
    openness = yaml.safe_load((source / "openness.yaml").read_text())
    openness["processes"].append(
        {
            "id": "recall",
            "actors": ["u1", "u2", "u3"],
            "openness_rationale": "what an actor makes of its own history is the phenomenon",
            "closure_rationale": "the answer is closed by the declared schema",
            "executor": {"mode": "generative", "model_profile": "mp"},
            "context_policy": "own-history",
            "prompt_ref": "recall",
            "trigger": {"type": "phase", "phase": 0, "repeat": True},
            "dependencies": {"after": ["choose"]},
            "outputs": [{"artifact_type": "choice", "schema_ref": "choice"}],
        }
    )
    (source / "openness.yaml").write_text(yaml.safe_dump(openness, sort_keys=False))
    domain = yaml.safe_load((source / "domain.yaml").read_text())
    domain["visibility"].append(
        {
            "id": "own-history",
            "allow": ["exchanges.choose"],
            "cardinality": {"exchanges.choose": {"limit": 2, "keep": "last", "by": "phase"}},
        }
    )
    (source / "domain.yaml").write_text(yaml.safe_dump(domain, sort_keys=False))
    protocol = yaml.safe_load((source / "protocol.yaml").read_text())
    protocol["time_model"]["end"] = 1
    (source / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))
    (source / "prompts" / "recall.txt").write_text("RECALL ACTOR={actor_ids} {context}")

    service = GenesisService(workspace)
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_G3_KEY",
            }
        )
        build = service.compile_study(source, "builds/choice")["path"]
        service.create_run({"id": "r", "study_id": "choice-study", "build": build})
        assert service.execute_run("r")["status"] == "completed"
    finally:
        service.close()

    second_round = [prompt for prompt in prompts if "ACTOR=u1" in prompt][-1]
    # The exchange carries u1's own earlier answer. It also carries the context
    # u1 was given, which legitimately includes shared state other actors wrote;
    # that an actor is handed only its OWN exchanges is asserted on the log
    # itself in the test above, where it can be checked exactly.
    assert '"ids": ["article-u1"], "note": "seen by u1"' in second_round
    # Both rounds are carried: the earlier round's exchange and this round's own,
    # newest last, under the declared cap.
    assert '"phase": 0' in second_round and '"phase": 1' in second_round
    assert second_round.index('"phase": 0') < second_round.index('"phase": 1')


def _controller_with_log(processes: dict[str, Any] | None = None) -> Any:
    from genesis.runtime import RunController

    controller = RunController.__new__(RunController)
    controller._exchange_log = {}
    controller._exchange_processes = frozenset({"publish", "diary", "board"})
    controller.scheduler = type("S", (), {"processes": processes or {}})()
    return controller


def _turn(process_id: str, actors: tuple[str, ...]) -> Any:
    return type(
        "T",
        (),
        {
            "process_id": process_id,
            "actors": actors,
            "actor_ids": actors,
            "process": {},
            "phase": 0,
        },
    )()


def test_an_exchange_does_not_carry_the_exchanges_it_was_shown() -> None:
    from genesis.runtime import _exchange_context

    # Round n's context already holds rounds n-1..n-k; storing it whole nests
    # every round inside the next.
    context = {"followers": 4, "exchanges": {"publish": [{"phase": 0, "outputs": {}}]}}
    assert _exchange_context(context) == {"followers": 4}


def test_a_group_turn_keeps_one_exchange_and_shares_it_with_no_one_else() -> None:
    from genesis.runtime import ProcessInvocation, ProcessResult

    controller = _controller_with_log()
    joint = _turn("board", ("a1", "a2"))
    call = ProcessInvocation("i", "run", "board", actor_ids=("a1", "a2"))
    controller._record_exchange(joint, call, ProcessResult(outputs={"vote": "yes"}), 1)
    # One entry for the group, not one per member.
    assert list(controller._exchange_log) == [("board", ("a1", "a2"))]
    assert len(controller._exchanges_for(joint)["board"]) == 1
    # A member acting alone is a different turn, and sees none of it.
    assert controller._exchanges_for(_turn("board", ("a1",))) == {}


def test_a_process_nothing_reads_records_no_exchanges() -> None:
    from genesis.runtime import ProcessInvocation, ProcessResult

    controller = _controller_with_log()
    unwatched = _turn("settle", ("w1",))
    call = ProcessInvocation("i", "run", "settle", actor_ids=("w1",))
    controller._record_exchange(unwatched, call, ProcessResult(outputs={"revenue": 3}), 1)
    assert controller._exchange_log == {}


def test_an_unread_exchanges_namespace_leaves_the_context_hash_alone() -> None:
    from genesis.runtime import ContextEngine, ProcessInvocation

    engine = ContextEngine({"p": {"allow": ["notes"]}})
    state = {"notes": ["n"]}
    without = engine.build("p", ProcessInvocation("i", "run", "p", actor_ids=("w1",)), state)
    with_history = engine.build(
        "p",
        ProcessInvocation(
            "i", "run", "p", actor_ids=("w1",), exchanges={"publish": [{"phase": 0}]}
        ),
        state,
    )
    assert without.content_hash == with_history.content_hash


def test_resume_rebuilds_only_what_completed_and_redacts_as_the_policy_says() -> None:
    processes = {
        "publish": {"trace_policy": {"record_raw_response": False}},
        "diary": {"trace_policy": {}},
    }
    controller = _controller_with_log(processes)

    class _Persistence:
        @staticmethod
        def list_artifacts(_run_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "artifact_id": "run-publish-w1-0-attempt-1",
                    "payload": json.dumps({"outputs": {"response": "raw text"}}),
                }
            ]

    controller.persistence = _Persistence()
    events = [
        {
            "event_id": "run-publish-w1-0-attempt-1",
            "kind": "process_completed",
            "process_id": "publish",
            "actors": ["w1"],
            "phase": 0,
            "attempt": 1,
            "commit_order": 1,
            "context": {"followers": 4, "exchanges": {"publish": []}},
        },
        {
            "event_id": "run-publish-w1-1-attempt-1",
            "kind": "process_skipped",
            "process_id": "publish",
            "actors": ["w1"],
            "phase": 1,
            "attempt": 1,
            "commit_order": 2,
        },
        {
            "event_id": "run-diary-w1-2-attempt-1",
            "kind": "process_completed",
            "process_id": "diary",
            "actors": ["w1"],
            "phase": 2,
            "attempt": 1,
            "commit_order": 3,
        },
    ]
    controller._restore_exchange_log("run", events)
    published = controller._exchange_log[("publish", ("w1",))]
    # A skipped invocation records nothing live, so it must not appear here.
    assert [entry["phase"] for entry in published] == [0]
    # The payload row redacts only generative processes; the policy is applied again.
    assert published[0]["outputs"] == {"response": "<raw-response-not-recorded>"}
    assert published[0]["context"] == {"followers": 4}
    # An invocation whose outputs are no longer retained says so.
    diary = controller._exchange_log[("diary", ("w1",))][0]
    assert diary["outputs"] == {} and diary["outputs_recorded"] is False


def test_a_purge_during_the_rebuild_does_not_stop_the_resume() -> None:
    controller = _controller_with_log({"publish": {}})

    class _Purging:
        @staticmethod
        def iter_artifacts(_run_id: str):
            raise ValueError("ARTIFACT_PURGED_DURING_READ")

    controller.persistence = _Purging()
    controller._restore_exchange_log(
        "run",
        [
            {
                "event_id": "e1",
                "kind": "process_completed",
                "process_id": "publish",
                "actors": ["w1"],
                "phase": 0,
                "attempt": 1,
            }
        ],
    )
    entry = controller._exchange_log[("publish", ("w1",))][0]
    assert entry["outputs"] == {} and entry["outputs_recorded"] is False


def test_reading_your_own_exchanges_in_a_batch_requires_a_timing_declaration() -> None:
    from genesis.information_timing import batch_dependencies

    process = {
        "id": "reflect",
        "actors": ["w1", "w2"],
        "executor": {"mode": "generative"},
        "context_policy": "own",
    }
    own = {"allow": ["exchanges.reflect"]}
    assert batch_dependencies(process, own)
    other = {"allow": ["exchanges.publish"]}
    assert batch_dependencies(process, other) == []


@pytest.mark.parametrize(
    ("allow", "expected"),
    [
        (["exchanges.nowhere"], "is not a declared process"),
        (["exchanges"], "names one process"),
    ],
)
def test_an_exchanges_path_that_names_no_process_is_refused(
    tmp_path: Path, allow: list[str], expected: str
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _write_package(
        workspace,
        {
            "openness": {"processes": []},
            "domain": {"visibility": [{"id": "p", "allow": allow}]},
        },
        "exchange-study",
    )
    service = GenesisService(workspace)
    try:
        with pytest.raises(Exception, match="CONTEXT_EXCHANGES_INVALID") as raised:
            service.compile_study(source, "builds/exchange")
        assert expected in str(raised.value)
    finally:
        service.close()


def test_an_uncapped_exchanges_path_is_flagged(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    {
                        "id": "tick",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "p",
                    }
                ]
            },
            "domain": {"visibility": [{"id": "p", "allow": ["exchanges.tick"]}]},
        },
        "exchange-study",
    )
    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/exchange")
        report = json.loads((Path(build["path"]) / "validation_report.json").read_text())
        flagged = [issue for issue in report["warnings"] if issue["code"] == "CONTEXT_UNBOUNDED"]
        assert flagged and "exchanges.tick" in flagged[0]["path"]
    finally:
        service.close()


# --- H1: a round's schedule is decided by the state the round began with -----------


def _latch_scheduler() -> Any:
    from genesis.runtime import Scheduler

    return Scheduler(
        [
            {
                "id": "reflect",
                "trigger": {
                    "type": "condition",
                    "repeat": True,
                    "predicate": {"path": "flag", "op": "truthy"},
                },
            },
            {"id": "publish", "trigger": {"type": "phase", "phase": 0, "repeat": True}},
            {"id": "late", "trigger": {"type": "phase", "phase": 5, "repeat": True}},
            {
                "id": "consume",
                "trigger": {"type": "phase", "phase": 0, "repeat": True},
                "dependencies": {"after": ["reflect", "publish", "late"]},
            },
        ]
    )


def test_asking_what_is_ready_does_not_change_what_is_ready() -> None:
    used, fresh = _latch_scheduler(), _latch_scheduler()
    # A query -- deciding a concurrency limit, say -- must leave no trace.
    used.ready(0, state={"flag": False})
    assert _ready_ids(used, 0, {"flag": True}) == _ready_ids(fresh, 0, {"flag": True})
    assert "reflect" in _ready_ids(used, 0, {"flag": True})


def test_a_consumer_is_not_held_back_by_a_producer_that_is_not_triggered() -> None:
    scheduler = _latch_scheduler()
    assert _ready_ids(scheduler, 0, {"flag": False}) == ["publish"]
    scheduler.complete("publish", 0)
    # A condition trigger is deliberately re-read as a round proceeds, so a
    # process becomes ready once a sibling writes the state it waits on. The
    # skip therefore means "not triggered when the consumer was considered".
    assert _ready_ids(scheduler, 0, {"flag": True}) == ["reflect"]


def test_a_resumed_scheduler_decides_the_same_way_as_an_uninterrupted_one() -> None:
    uninterrupted = _latch_scheduler()
    uninterrupted.ready(0, state={"flag": False})
    uninterrupted.complete("publish", 0)
    # A resume rebuilds the scheduler and replays completions, and nothing else
    # was carried live. A remembered skip made the two disagree: the resumed
    # run scheduled a producer the uninterrupted run had passed over.
    resumed = _latch_scheduler()
    resumed.complete("publish", 0)
    for state in ({"flag": False}, {"flag": True}):
        assert _ready_ids(resumed, 0, state) == _ready_ids(uninterrupted, 0, state)


def test_a_wildcard_reads_through_sets_and_collections() -> None:
    state = {
        "rows": [{"creator": "w1"}, {"creator": "w9"}],
        "feeds": {"u1": [{"author": "w1"}]},
        "tags": {"u1": {"w1"}},
    }
    through_lists = _context(
        {
            "allow": ["rows"],
            "scope": {"rows": {"field": "creator", "in": "state.feeds.${actor}.author"}},
        },
        state,
        "u1",
    )
    assert [row["creator"] for row in through_lists["rows"]] == ["w1"]
    through_sets = _context(
        {"allow": ["rows"], "scope": {"rows": {"field": "creator", "in": "state.tags.${actor}.*"}}},
        state,
        "u1",
    )
    assert [row["creator"] for row in through_sets["rows"]] == ["w1"]


def test_an_empty_user_section_is_refused() -> None:
    from genesis.providers import split_prompt_roles

    with pytest.raises(ValueError, match="empty"):
        split_prompt_roles("[system]\nRules.\n[user]\n\n")


def test_a_trigger_path_written_as_state_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    {
                        "id": "tick",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "p",
                        "trigger": {
                            "type": "condition",
                            "predicate": {"path": "state.counter", "op": "gte", "value": 1},
                        },
                    }
                ]
            },
            "domain": {
                "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
                "visibility": [{"id": "p", "allow": ["counter"]}],
            },
        },
        "trigger-study",
    )
    service = GenesisService(workspace)
    try:
        with pytest.raises(Exception, match="TRIGGER_PATH_INVALID") as raised:
            service.compile_study(source, "builds/trigger")
        assert "name 'counter'" in str(raised.value)
    finally:
        service.close()


# --- M3-M7 and the low-severity sweep ----------------------------------------------


def test_a_compound_availability_predicate_is_evaluated() -> None:
    state = {"notes": ["n"]}
    policy = {
        "allow": ["notes"],
        "available_when": {
            "predicate": {
                "all": [
                    {"path": "phase", "op": "gte", "value": 1},
                    {"path": "phase", "op": "lt", "value": 5},
                ]
            }
        },
    }
    from genesis.runtime import ContextEngine, ProcessInvocation

    engine = ContextEngine({"p": policy})
    inside = engine.build("p", ProcessInvocation("i", "r", "p", actor_ids=("w1",), phase=2), state)
    outside = engine.build("p", ProcessInvocation("i", "r", "p", actor_ids=("w1",), phase=0), state)
    assert "notes" in inside.data and "notes" not in outside.data


def test_increment_requires_numbers() -> None:
    from genesis.runtime import StateStore

    store = StateStore({"label": str, "tally": int}, {"label": "x", "tally": 1})
    with pytest.raises(TypeError, match="numeric"):
        store.apply([{"field": "label", "op": "increment", "value": "y"}], {"label"})
    store.apply([{"field": "tally", "op": "increment", "value": 2}], {"tally"})
    assert store.snapshot() == {"label": "x", "tally": 3}


def test_an_effect_reading_an_output_the_call_never_returns_is_refused() -> None:
    from genesis.information_timing import model_effect_problems

    process = {
        "id": "write",
        "actors": ["w1"],
        "executor": {"mode": "generative", "model_profile": "mp"},
        "outputs": [
            {"artifact_type": "article", "schema_ref": "a"},
            {"artifact_type": "second", "schema_ref": "b"},
        ],
        "state_effects": [{"field": "notes", "op": "append", "from": "second.text"}],
    }
    problems = model_effect_problems(process, {"notes": "array"})
    assert any("returns only 'article'" in problem for problem in problems)


def test_a_placeholder_inside_a_value_is_not_substituted() -> None:
    from genesis.providers import render_prompt

    _system, user = render_prompt(
        "[user]\n{context.note} | {actor_ids}", {"note": "hello {actor_ids}"}, ("a1", "a2"), 3
    )
    assert user == "hello {actor_ids} | a1, a2"
    _system2, whole = render_prompt("[user]\n{context.note}", {"note": "{context}"}, ("a1",), 1)
    assert whole == "{context}"


def test_non_finite_numbers_render_as_null() -> None:
    from genesis.providers import render_prompt

    _system, user = render_prompt(
        "[user]\n{context}", {"score": float("nan"), "cap": float("inf")}, ("w1",), 0
    )
    assert json.loads(user) == {"score": None, "cap": None}


def test_a_scope_field_holding_several_identities_matches_any_of_them() -> None:
    state = {"rows": [{"authors": ["w1", "w2"], "title": "t"}, {"authors": ["w3"], "title": "u"}]}
    data = _context(
        {"allow": ["rows"], "scope": {"rows": {"field": "authors", "in": "actor.ids"}}},
        state,
        "w1",
    )
    assert [row["title"] for row in data["rows"]] == ["t"]


def test_a_condition_may_not_declare_two_combinators() -> None:
    from genesis.runtime import _validate_condition

    with pytest.raises(ValueError, match="together"):
        _validate_condition(
            {
                "all": [{"path": "x", "op": "eq", "value": 1}],
                "any": [{"path": "x", "op": "eq", "value": 2}],
            }
        )


def test_a_rule_executor_checks_its_predicates_up_front() -> None:
    from genesis.runtime import RuleExecutor

    with pytest.raises(ValueError, match="mapping"):
        RuleExecutor({"rules": [{"when": "always", "outputs": {}}]})


def test_a_relative_entry_point_reports_the_contract() -> None:
    from genesis.service import _resolve_callable

    with pytest.raises(ValueError, match="EXECUTOR_UNAVAILABLE"):
        _resolve_callable(".rules:tick")


def test_a_mechanism_may_carry_the_provenance_every_other_entry_carries() -> None:
    from genesis.specification.models import InstitutionSpec, MechanismSpec

    # The guided workflow instructs the assistant to cite turns for every
    # proposed value; these two models refused it, so a whole domain draft died.
    origin = {"origin": "assistant_proposed", "evidence_refs": ["turn-2"], "rationale": "r"}
    mechanism = MechanismSpec.model_validate(
        {"id": "sanction", "description": "halves revenue", "origin": origin}
    )
    assert mechanism.origin is not None
    institution = InstitutionSpec.model_validate({"id": "platform", "origin": origin})
    assert institution.origin is not None
    # A declaration without provenance serialises exactly as it did before.
    assert MechanismSpec(id="plain").model_dump(mode="json") == {
        "id": "plain",
        "implements": None,
        "description": None,
    }
    assert InstitutionSpec(id="plain").model_dump(mode="json") == {"id": "plain", "type": None}


def test_a_state_effect_is_not_judged_against_a_domain_that_does_not_exist_yet() -> None:
    from genesis.information_timing import model_effect_problems

    # The guided route approves the openness layer before the domain layer, so
    # at that point the package declares no states at all. Judging a field
    # against an empty domain made every openness stage unapprovable.
    process = {
        "id": "reflect",
        "actors": ["w1", "w2"],
        "executor": {"mode": "generative", "model_profile": "mp"},
        "outputs": [{"artifact_type": "reflection", "schema_ref": "s"}],
        "state_effects": [
            {"field": "reflections", "op": "put", "key": "actor", "from": "reflection"}
        ],
    }
    assert model_effect_problems(process, {}) == []
    assert model_effect_problems(process, None) == []
    # Once the domain exists, a field it does not declare is still caught.
    problems = model_effect_problems(process, {"articles": "array"})
    assert any("does not declare" in problem for problem in problems)


def test_a_keyed_write_is_allowed_on_a_json_field() -> None:
    from genesis.information_timing import model_effect_problems

    process = {
        "id": "choose",
        "actors": ["u1", "u2"],
        "executor": {"mode": "generative", "model_profile": "mp"},
        "outputs": [{"artifact_type": "choice", "schema_ref": "s"}],
        "state_effects": [{"field": "picks", "op": "put", "key": "actor", "from": "choice"}],
    }
    assert model_effect_problems(process, {"picks": "json"}) == []


def test_a_repair_leaves_the_accepted_prompt_recorded() -> None:
    from genesis.providers import ProviderExecutor, ProviderRequest, ProviderResponse
    from genesis.runtime import ProcessInvocation

    seen: list[ProviderRequest] = []

    class _Repairing:
        provider = "t"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            seen.append(request)
            value = {"text": "ok"} if len(seen) > 1 else {}
            return ProviderResponse(
                json.dumps(value), self.provider, request.model, "r", parsed=value
            )

    executor = ProviderExecutor(
        _Repairing(),
        model="m",
        prompt_template="[user]\n{context}",
        output_schema={"type": "object", "required": ["text"]},
    )
    result = executor.execute(ProcessInvocation("i", "r", "p", actor_ids=("w1",), context={}))
    # The answer that was accepted is the one the invocation's hash must name.
    assert result.metadata["prompt_hash"] == seen[-1].content_digest()


# --- H6, M10: a declaration means the same thing at compile and at commit ----------


def test_duplicate_actor_ids_are_refused_in_either_form() -> None:
    from genesis.runtime import expand_actor_instances

    with pytest.raises(ValueError, match="duplicate actor ids"):
        expand_actor_instances({"id": "x", "actors": ["u1", "u1", "u2"]}, {})
    with pytest.raises(ValueError, match="duplicate actor ids"):
        expand_actor_instances({"id": "x", "actors": {"ids": ["u1", "u1"]}}, {})
    assert expand_actor_instances({"id": "x", "actors": ["u1", "u2"]}, {}) == [("u1",), ("u2",)]


def test_a_keyed_write_the_runtime_refuses_does_not_compile() -> None:
    from genesis.information_timing import timing_diagnostics, whole_field_writes
    from genesis.runtime import _composes_with_siblings

    foreign_key = {
        "id": "tally",
        "actors": ["a", "b"],
        "executor": {"mode": "computational", "parameters": {"entry_point": "m:f"}},
        "information_timing": {"mode": "simultaneous"},
        "context_policy": "p",
        "state_effects": [{"field": "picks", "op": "put", "key": "round-3"}],
    }
    # The runtime refuses this at commit, so compilation must refuse it too.
    assert not _composes_with_siblings({"field": "picks", "op": "put", "key": "round-3"}, ("a",))
    assert whole_field_writes(foreign_key)
    errors, _ = timing_diagnostics([foreign_key], {"p": {"allow": ["picks"]}})
    assert [error["code"] for error in errors] == ["SIMULTANEOUS_WRITE_CONFLICT"]

    own_key = {**foreign_key, "state_effects": [{"field": "picks", "op": "put", "key": "actor"}]}
    assert whole_field_writes(own_key) == []


# --- H4, M8, M9: isolation covers the channels a package declares ------------------


def test_a_trigger_predicate_reading_measurement_state_is_refused() -> None:
    gated = {
        "id": "sanction",
        "executor": {"mode": "computational", "parameters": {"entry_point": "m:f"}},
        "context_policy": "s",
        "trigger": {"type": "condition", "predicate": {"path": "scores", "op": "truthy"}},
    }
    errors, _ = _diagnose([DETECT, gated], {"detector": {}, "s": {}})
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]
    assert "state 'scores'" in errors[0]["message"]


def test_actors_drawn_from_measurement_state_are_refused() -> None:
    chosen = {
        "id": "act",
        "actors": {"source": "scores", "id_field": "id"},
        "executor": {"mode": "generative", "model_profile": "mp"},
        "context_policy": "s",
    }
    errors, _ = _diagnose([DETECT, chosen], {"detector": {}, "s": {}})
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]


def test_a_policy_admitting_a_measurements_exchanges_is_refused() -> None:
    reader = {
        "id": "publish",
        "executor": {"mode": "generative", "model_profile": "mp"},
        "context_policy": "x",
    }
    errors, _ = _diagnose([DETECT, reader], {"detector": {}, "x": {"allow": ["exchanges.detect"]}})
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]
    assert "exchanges.detect" in errors[0]["message"]


def test_a_feedback_slot_carrying_measurement_state_is_refused() -> None:
    from genesis.measurement import measurement_diagnostics

    reader = {
        "id": "publish",
        "executor": {"mode": "generative", "model_profile": "mp"},
        "context_policy": "f",
    }
    errors, _ = measurement_diagnostics(
        [DETECT, reader],
        {"detector": {}, "f": {"allow": ["feedback.last"]}},
        {"publish": ["scores"]},
    )
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]
    assert "state 'scores'" in errors[0]["message"]


def test_a_measurement_without_declared_outputs_is_reported() -> None:
    # A measurement that declares no outputs can commit an artifact nothing
    # recognises as its own, so a consumer reading it is never reported.
    silent = {
        "id": "watch",
        "measurement": True,
        "executor": {"mode": "computational", "parameters": {"entry_point": "m:f"}},
        "context_policy": "d",
    }
    _errors, warnings = _diagnose([silent], {"d": {}})
    assert [warning["code"] for warning in warnings] == ["MEASUREMENT_OUTPUTS_UNDECLARED"]
    declared = {**silent, "outputs": [{"artifact_type": "watched", "schema_ref": "s"}]}
    _declared_errors, declared_warnings = _diagnose([declared], {"d": {}})
    assert declared_warnings == []


def test_a_consumer_cannot_defeat_its_own_gate_by_declaring_the_same_type() -> None:
    from genesis.measurement import gated_input_refs

    detect = {**DETECT, "outputs": [{"artifact_type": "score", "schema_ref": "s"}]}
    consumer = _settle(
        inputs=["score"],
        outputs=[{"artifact_type": "score", "schema_ref": "s"}],
        measurement_use=[
            {
                "source": "detect",
                "rationale": "r",
                "when": {"path": "condition.governance", "op": "eq", "value": "hidden"},
            }
        ],
    )
    outside = gated_input_refs(
        consumer,
        {"detect": detect, "settle": consumer},
        phase=0,
        condition={"governance": "none"},
    )
    assert outside == []


def test_a_gate_opens_for_a_factor_condition() -> None:
    from genesis.measurement import gated_input_refs
    from genesis.runtime import expand_protocol_conditions

    detect = {**DETECT, "outputs": [{"artifact_type": "score", "schema_ref": "s"}]}
    consumer = _settle(
        inputs=["score"],
        measurement_use=[
            {
                "source": "detect",
                "rationale": "r",
                "when": {"path": "condition.governance", "op": "eq", "value": "hidden"},
            }
        ],
    )
    conditions = {
        str(item["id"]): item
        for item in expand_protocol_conditions(
            {"factors": [{"id": "governance", "levels": ["none", "hidden"]}]}
        )
    }
    governed = conditions["governance-hidden"]
    ungoverned = conditions["governance-none"]
    # A factor condition nests its levels; the predicate names the factor.
    assert gated_input_refs(consumer, {"detect": detect}, phase=0, condition=governed) == ["score"]
    assert gated_input_refs(consumer, {"detect": detect}, phase=0, condition=ungoverned) == []


# --- H2, H3, M1, M2: the pin describes code without running it ---------------------


def _package_module(root: Path, name: str, init: str, rules: str) -> None:
    package = root / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(init)
    (package / "rules.py").write_text(rules)


def test_locating_study_code_never_imports_it(tmp_path: Path) -> None:
    from genesis.service import _executor_code_identity

    marker = tmp_path / "ran.txt"
    _package_module(
        tmp_path,
        "sidepkg",
        f"from pathlib import Path\n\nPath({str(marker)!r}).write_text('ran')\n",
        "def tick(invocation):\n    return {}\n",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        identity = _executor_code_identity(
            [
                {
                    "id": "tick",
                    "executor": {
                        "mode": "computational",
                        "parameters": {"entry_point": "sidepkg.rules:tick"},
                    },
                }
            ]
        )
        assert identity["tick"]["source_sha256"]
        # Importing to locate the module would have run the parent package.
        assert not marker.exists()
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("sidepkg", None)


def test_a_package_that_raises_does_not_stop_a_manifest(tmp_path: Path) -> None:
    from genesis.service import _executor_code_identity

    _package_module(
        tmp_path,
        "angrypkg",
        "raise RuntimeError('module-level failure')\n",
        "def tick(invocation):\n    return {}\n",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        identity = _executor_code_identity(
            [
                {
                    "id": "tick",
                    "executor": {
                        "mode": "computational",
                        "parameters": {"entry_point": "angrypkg.rules:tick"},
                    },
                }
            ]
        )
        assert identity["tick"]["source_sha256"]
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("angrypkg", None)


def test_the_pin_covers_the_file_that_defines_the_executor(tmp_path: Path) -> None:
    from genesis.service import _executor_code_identity

    _package_module(
        tmp_path,
        "reexported",
        "from .rules import tick\n",
        "def tick(invocation):\n    return {'v': 1}\n",
    )
    sys.path.insert(0, str(tmp_path))
    process = {
        "id": "tick",
        "executor": {"mode": "computational", "parameters": {"entry_point": "reexported:tick"}},
    }
    try:
        before = _executor_code_identity([process])["tick"]["source_sha256"]
        (tmp_path / "reexported" / "rules.py").write_text(
            "def tick(invocation):\n    return {'v': 999}\n"
        )
        after = _executor_code_identity([process])["tick"]["source_sha256"]
        # The rules the executor actually runs changed, so the pin must change.
        assert before and after and before != after
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("reexported", None)


def test_study_code_that_nothing_pins_is_refused(tmp_path: Path, study_module) -> None:
    module, path = study_module
    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        build = service.compile_study(_code_package(workspace, module), "builds/code")["path"]
        service.create_run({"id": "r", "study_id": "code-study", "build": build})
        path.unlink()
        with pytest.raises(ValueError, match="RUN_CODE_UNPINNED"):
            service.execute_run("r")
        # Supplying the executor directly is the declared way through.
        service.create_run({"id": "r2", "study_id": "code-study", "build": build})
        run = service.execute_run("r2", executor_overrides={"tick": lambda _invocation: {}})
        assert run["status"] == "completed"
    finally:
        service.close()


def test_a_bundle_does_not_promise_re_execution_of_code_it_lacks() -> None:
    from genesis.evidence import evaluate_capabilities

    def capability(**kwargs: Any) -> dict[str, Any]:
        found = evaluate_capabilities(
            has_build=True,
            has_closure=True,
            has_recorded_outputs=True,
            has_checkpoint_evidence=False,
            has_outcomes=True,
            **kwargs,
        )
        return next(item for item in found if item["capability"] == "reexecute")

    assert capability()["available"] is True
    withheld = capability(has_executor_code=False)
    assert withheld["available"] is False and "executor_code" in withheld["missing"]


# --- H5: a projection reaches the nested field it names ----------------------------


def test_a_projection_drops_and_keeps_nested_fields() -> None:
    from genesis.runtime import _project_value

    records = [{"id": "a1", "value": {"title": "t", "body": "PRIVATE-STRATEGY"}}]
    assert _project_value(records, "board", {"drop": ["value.body"]}) == [
        {"id": "a1", "value": {"title": "t"}}
    ]
    assert _project_value(records, "board", {"keep": ["value.title"]}) == [
        {"value": {"title": "t"}}
    ]
    # A name that is neither a field nor a path into one is refused.
    with pytest.raises(ValueError, match="dotted path"):
        _project_value(records, "board", {"drop": ["value."]})


def test_a_nested_projection_survives_the_context_engine(tmp_path: Path) -> None:
    state = {"board": [{"id": "a1", "value": {"title": "t", "body": "PRIVATE-STRATEGY"}}]}
    data = _context(
        {"allow": ["board"], "project": {"board": {"drop": ["value.body"]}}}, state, "w1"
    )
    assert "PRIVATE-STRATEGY" not in str(data)
    assert data["board"][0]["value"]["title"] == "t"


# --- Review fixes on the work above ------------------------------------------------


def test_an_unresolvable_module_does_not_change_the_configuration_digest() -> None:
    from genesis.execution_manifest import executor_code_digest, executor_code_reference

    resolved = {"a": {"reference": "m:f", "source_sha256": "abc"}}
    unresolved = {**resolved, "b": {"reference": "gone:f", "source_sha256": None}}
    # Whether a module happens to be importable is a property of the machine.
    assert executor_code_digest(unresolved) == executor_code_digest(resolved)
    assert executor_code_digest({"b": {"reference": "gone:f", "source_sha256": None}}) == ""
    # A computational process with no entry point at all names no code.
    assert executor_code_reference({"executor": {"mode": "computational"}}) is None


def test_building_a_manifest_does_not_import_study_code(tmp_path: Path) -> None:
    marker = tmp_path / "imported.txt"
    directory = tmp_path / "code"
    directory.mkdir()
    name = "g1_side_effect_module"
    (directory / f"{name}.py").write_text(
        f"from pathlib import Path\n\nPath({str(marker)!r}).write_text('imported')\n\n"
        "def tick(invocation):\n    return {}\n"
    )
    sys.path.insert(0, str(directory))
    try:
        from genesis.service import _executor_code_identity

        identity = _executor_code_identity(
            [
                {
                    "id": "tick",
                    "executor": {
                        "mode": "computational",
                        "parameters": {"entry_point": f"{name}:tick"},
                    },
                }
            ]
        )
        assert identity["tick"]["source_sha256"]
        assert not marker.exists()
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop(name, None)


def test_a_projection_says_what_it_projects() -> None:
    from genesis.runtime import _project_value

    picks = {"u1": ["a1"], "u2": ["a2"]}
    # Guessing from the value's shape dropped whole entries; this must not.
    with pytest.raises(ValueError, match="CONTEXT_PROJECT"):
        _project_value(picks, "picks", {"keep": ["article"]})
    record = {"title": "t", "clicks": 3, "strategy_summary": "secret"}
    assert _project_value(
        record, "board", {"drop": ["strategy_summary"], "applies_to": "record"}
    ) == {
        "title": "t",
        "clicks": 3,
    }
    with pytest.raises(ValueError, match="must be 'records' or 'record'"):
        _project_value(record, "board", {"keep": ["title"], "applies_to": "each"})


def test_gating_withholds_only_what_the_measurement_alone_produces() -> None:
    from genesis.measurement import gated_input_refs

    detect = {**DETECT, "outputs": [{"artifact_type": "score", "schema_ref": "s"}]}
    tally = {"id": "tally", "outputs": [{"artifact_type": "score", "schema_ref": "s"}]}
    settle = _settle(
        inputs=["score", "detect"],
        measurement_use=[
            {
                "source": "detect",
                "rationale": "r",
                "when": {"path": "condition.governance", "op": "eq", "value": "hidden"},
            }
        ],
    )
    processes = {"detect": detect, "tally": tally, "settle": settle}
    outside = gated_input_refs(settle, processes, phase=1, condition={"governance": "none"})
    # 'score' also comes from a process the declaration says nothing about.
    assert outside == ["score"]


def test_a_keyed_write_composes_only_under_the_acting_actor() -> None:
    from genesis.runtime import _composes_with_siblings

    assert _composes_with_siblings({"field": "p", "op": "put", "key": "u1"}, ("u1",))
    assert not _composes_with_siblings({"field": "p", "op": "put", "key": "round-3"}, ("u1",))
    assert not _composes_with_siblings({"field": "p", "op": "put"}, ("u1",))
    assert _composes_with_siblings({"field": "p", "op": "append"}, ("u1",))
    assert not _composes_with_siblings({"field": "p", "op": "set"}, ("u1",))


def test_string_effects_and_actor_fan_out_are_checked_at_compile() -> None:
    from genesis.information_timing import model_effect_problems

    call: dict[str, Any] = {
        "id": "write",
        "actors": ["w1", "w2"],
        "executor": {"mode": "generative", "model_profile": "mp"},
        "outputs": [{"artifact_type": "article", "schema_ref": "a"}],
        "state_effects": ["revenue"],
    }
    states = {"revenue": "number", "picks": "object"}
    problems = model_effect_problems(call, states)
    assert any("reads output 'revenue'" in problem for problem in problems)

    unfanned = {
        **call,
        "actors": {"source": "population.users", "fan_out": False},
        "state_effects": [{"field": "picks", "op": "put", "key": "actor", "from": "article"}],
    }
    assert any(
        "one invocation per actor" in problem for problem in model_effect_problems(unfanned, states)
    )


def test_a_selector_that_cannot_identify_records_is_refused() -> None:
    state = {"rows": [{"creator": "w1"}], "relations": {"u1": ["w1"]}}
    with pytest.raises(ValueError, match="CONTEXT_SCOPE"):
        _context(
            {
                "allow": ["rows"],
                "scope": {"rows": {"field": "creator", "in": "state.relations"}},
            },
            state,
            "u1",
        )


def test_a_measurement_read_through_a_scope_selector_is_refused() -> None:
    reader = {
        "id": "recommend",
        "executor": {"mode": "computational", "parameters": {"entry_point": "m:f"}},
        "context_policy": "feed",
    }
    policies = {
        "detector": {},
        "feed": {
            "allow": ["articles"],
            "scope": {"articles": {"field": "id", "in": "state.scores"}},
        },
    }
    errors, _ = _diagnose([DETECT, reader], policies)
    assert [error["code"] for error in errors] == ["MEASUREMENT_LEAK"]
    assert "state 'scores'" in errors[0]["message"]


def test_a_null_output_is_written_but_a_missing_one_is_not() -> None:
    from genesis.runtime import model_call_effects

    process = {
        "id": "c",
        "state_effects": [
            {"field": "note", "op": "set", "from": "out.note"},
            {"field": "other", "op": "set", "from": "out.absent"},
        ],
    }
    effects = model_call_effects(process, {"out": {"note": None}}, ("w1",))
    assert effects == [{"field": "note", "op": "set", "value": None}]


# --- workflow grammar and cross-layer attribution ---------------------------------


def _three_layer_workflow():
    from genesis.elicitation import WorkflowRegistry
    from genesis.service import _workflows_root

    return WorkflowRegistry(_workflows_root()).get("three-layer-study")


def test_the_evaluation_request_shows_template_bodies_not_paths() -> None:
    """A path is unusable context: the assistant cannot open files."""
    from genesis.elicitation import ElicitationAssistant, ElicitationSession

    workflow = _three_layer_workflow()
    stage = workflow.stage("domain")
    assert stage.template_texts, "template bodies must be resolved at registry load"
    session = ElicitationSession(
        session_id="s1",
        specification_id="spec",
        workflow_id=workflow.id,
        workflow_version=workflow.version,
        model_profile_id="mp",
        researcher_id="r1",
        base_specification_version=0,
        current_stage=stage.id,
    )
    request = ElicitationAssistant().assemble_evaluation_request(workflow, stage, session)
    assert "templates/domain.yaml" in request
    # The grammar itself, not just the filename.
    assert "items: {limit: 1, keep: last, by: phase}" in request
    assert "PREDICATE GRAMMAR" in request


def test_the_workflow_instructions_carry_the_declaration_grammar() -> None:
    workflow = _three_layer_workflow()
    text = workflow.instructions_text()
    for fragment in (
        "{path, op, value}",
        "{field, op, from, key}",
        "scope: {field, in}",
        "either",
    ):
        assert fragment in text


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"source_file": "domain.yaml", "message": "bad"}, "owned by this stage"),
        ({"source_file": "openness.yaml", "message": "bad"}, "owned by the 'openness' stage"),
        ({"source_file": "package", "message": "bad"}, "bad"),
    ],
)
def test_a_refusal_names_the_layer_that_owns_the_defect(
    error: dict[str, Any], expected: str
) -> None:
    from genesis.elicitation import attribute_specification_error

    workflow = _three_layer_workflow()
    message = attribute_specification_error(error, workflow, workflow.stage("domain"))
    assert expected in message


def test_declaring_both_factors_and_conditions_is_refused_at_compile(tmp_path: Path) -> None:
    """expand_protocol_conditions refuses this at run time; compile must too."""
    from genesis.compiler import StudyCompiler, ValidationIssue

    source = _write_package(
        tmp_path,
        {
            "openness": {"processes": []},
            "protocol": {
                "time_model": {"type": "rounds", "start": 0, "end": 1},
                "factors": [{"id": "governance", "levels": ["none", "on"]}],
                "conditions": [{"id": "none", "factors": {"governance": "none"}}],
            }
        },
        "both-study",
    )
    with pytest.raises(ValidationIssue) as raised:
        StudyCompiler(source).compile(tmp_path / "build")
    codes = [issue.code for issue in raised.value.issues]
    assert "PROTOCOL_CONDITIONS_AMBIGUOUS" in codes


def test_a_compile_error_reaches_the_gate_with_the_layer_that_owns_it() -> None:
    """The preview compiled the candidate and then discarded the result."""
    from genesis.elicitation import attribute_specification_error

    workflow = _three_layer_workflow()
    # Only the final stage gates on compile errors: an intermediate stage cannot
    # compile, because later layers are not elicited yet.
    assert workflow.next_stage(workflow.stages[-1].id) is None
    assert workflow.next_stage(workflow.stages[0].id) is not None
    entry = {
        "code": "THEORY_FEEDBACK_SOURCE_UNSUPPORTED",
        "source_file": "theory.yaml",
        "message": "source.kind 'artifact' is not supported",
    }
    message = attribute_specification_error(entry, workflow, workflow.stages[-1])
    assert "owned by the 'theory' stage" in message


def test_compile_errors_carry_the_location_the_preview_needs(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler, ValidationIssue

    source = _write_package(
        tmp_path,
        {
            "openness": {"processes": []},
            "protocol": {
                "time_model": {"type": "rounds", "start": 0, "end": 1},
                "factors": [{"id": "g", "levels": ["a", "b"]}],
                "conditions": [{"id": "a", "factors": {"g": "a"}}],
            },
        },
        "path-study",
    )
    with pytest.raises(ValidationIssue) as raised:
        StudyCompiler(source).compile(tmp_path / "build")
    issue = next(i for i in raised.value.issues if i.code == "PROTOCOL_CONDITIONS_AMBIGUOUS")
    assert issue.source_file == "protocol.yaml"


def test_a_gate_on_an_undeclared_condition_factor_is_refused(tmp_path: Path) -> None:
    """Such a gate is false in every cell, so the treatment silently never arrives."""
    from genesis.compiler import StudyCompiler, ValidationIssue

    source = _write_package(
        tmp_path,
        {
            "openness": {"processes": []},
            "protocol": {
                "time_model": {"type": "rounds", "start": 0, "end": 1},
                "factors": [{"id": "peer-visibility", "levels": ["low", "high"]}],
            },
            "domain": {
                "states": [{"id": "leaderboard", "value_type": "array", "initial": []}],
                "visibility": [
                    {
                        "id": "creator-context",
                        "allow": ["leaderboard"],
                        "available_when": {
                            # The factor is 'peer-visibility'; this names 'visibility'.
                            "leaderboard": {"path": "condition.visibility", "op": "eq",
                                            "value": "high"}
                        },
                    }
                ],
            },
        },
        "gate-study",
    )
    with pytest.raises(ValidationIssue) as raised:
        StudyCompiler(source).compile(tmp_path / "build")
    issue = next(i for i in raised.value.issues if i.code == "CONDITION_FACTOR_UNKNOWN")
    assert "peer-visibility" in issue.message
    assert issue.source_file == "domain.yaml"


def test_a_gate_on_a_declared_condition_factor_is_accepted(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    source = _write_package(
        tmp_path,
        {
            "openness": {"processes": []},
            "protocol": {
                "time_model": {"type": "rounds", "start": 0, "end": 1},
                "factors": [{"id": "peer-visibility", "levels": ["low", "high"]}],
            },
            "domain": {
                "states": [{"id": "leaderboard", "value_type": "array", "initial": []}],
                "visibility": [
                    {
                        "id": "creator-context",
                        "allow": ["leaderboard"],
                        "available_when": {
                            "leaderboard": {"path": "condition.peer-visibility", "op": "eq",
                                            "value": "high"}
                        },
                    }
                ],
            },
        },
        "gate-ok-study",
    )
    StudyCompiler(source).compile(tmp_path / "build")
