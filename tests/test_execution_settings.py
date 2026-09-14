"""How much of a study to realise is an execution decision, not a specification one.

The build fixes what the study is: its condition space, its matching policy, its
streams. How many draws, of how many worlds, and which cells to run now say how
much of it to realise -- so they belong to the run, and neither multiplies unless
a researcher asks for it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from genesis.service import GenesisService

WORLDS = {
    "data/world-a.json": {"seed": 1, "actors": ["a1"]},
    "data/world-b.json": {"seed": 2, "actors": ["b1"]},
}


def _workspace(tmp_path: Path, *, empirical: bool = False) -> Path:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    domain: dict = {"states": [{"id": "counter", "value_type": "integer", "initial": 0}]}
    if empirical:
        domain["states"].append({"id": "population", "value_type": "object", "initial": {}})
        domain["initialization"] = {
            "mode": "empirical",
            "data_source": "data/world-a.json",
            "state_field": "population",
        }
    service.create_specification(
        {
            "id": "exec-study",
            "title": "execution settings",
            "processes": [
                {
                    "id": "tick",
                    "executor": {},
                    "context_policy": "public",
                    "state_effects": [{"field": "counter", "op": "set"}],
                }
            ],
            "theory": {"theory_family": "exploratory"},
            "domain": domain,
            "protocol": {
                "time_model": {"type": "rounds", "end": 2},
                "conditions": [{"id": "base"}, {"id": "alt"}],
            },
            "outcomes": [],
            "models": [],
        }
    )
    source = workspace / ".genesis" / "specifications" / "exec-study"
    if empirical:
        # The assets must exist before approval: empirical initialization is
        # validated against the package that carries them.
        (source / "data").mkdir(parents=True, exist_ok=True)
        for name, payload in WORLDS.items():
            (source / name).write_text(json.dumps(payload))
    version = service.get_specification("exec-study")["version"]
    if empirical:
        # Adding assets changes the package hash, so record a version that
        # includes them before approving.
        form = service._current_form_payload(source, "exec-study")
        form["id"] = "exec-study"
        version = service.update_specification("exec-study", form, version)["version"]
    service.approve_specification("exec-study", version, "researcher")
    compiled = service.compile_study(None, "builds/exec-study", specification_id="exec-study")
    service.create_run({"id": "exp", "study_id": "exec-study", "build": compiled["path"]})
    service.close()
    return workspace


@pytest.fixture
def service(tmp_path: Path) -> GenesisService:
    genesis = GenesisService(_workspace(tmp_path))
    yield genesis
    genesis.close()


# --- nothing multiplies unless asked ----------------------------------------------


def test_by_default_one_draw_of_one_world_per_condition(service: GenesisService) -> None:
    result = service.execute_protocol("exp")
    assert result["runs"] == ["exp-base-1", "exp-alt-1"]


def test_the_run_sets_the_number_of_draws_not_the_package(service: GenesisService) -> None:
    result = service.execute_protocol("exp", replications=3)
    assert len(result["runs"]) == 6  # 2 conditions x 3 draws
    assert {service.get_run(r)["manifest"]["replication"] for r in result["runs"]} == {1, 2, 3}


def test_a_study_can_be_extended_with_more_draws_against_the_same_build(
    service: GenesisService,
) -> None:
    """Adding draws later must not require editing the specification."""
    first = service.execute_protocol("exp", replications=1)
    second = service.execute_protocol("exp", replications=3)
    assert set(first["runs"]) < set(second["runs"])
    builds = {service.get_run(r)["manifest"]["build_hash"] for r in second["runs"]}
    assert len(builds) == 1


def test_fewer_than_one_draw_is_refused(service: GenesisService) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        service.execute_protocol("exp", replications=0)


# --- the plan is shown before it is paid for --------------------------------------


def test_the_arithmetic_is_reported_without_dispatching_anything(
    service: GenesisService,
) -> None:
    plan = service.execute_protocol("exp", replications=3, plan_only=True)
    assert plan["runs"] == 6
    assert plan["conditions"] == ["base", "alt"]
    assert plan["replications"] == 3
    with pytest.raises(KeyError):
        service.get_run("exp-base-1")


def test_a_subset_of_conditions_can_be_run_now(service: GenesisService) -> None:
    result = service.execute_protocol("exp", only_conditions=["alt"])
    assert result["runs"] == ["exp-alt-1"]


def test_an_unknown_condition_is_refused_with_the_declared_ones(
    service: GenesisService,
) -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        service.execute_protocol("exp", only_conditions=["nope"])


# --- worlds -----------------------------------------------------------------------


@pytest.fixture
def empirical(tmp_path: Path) -> GenesisService:
    genesis = GenesisService(_workspace(tmp_path, empirical=True))
    yield genesis
    genesis.close()


def test_each_world_is_its_own_experiment_so_its_draws_differ(
    empirical: GenesisService,
) -> None:
    """Matched seeds key on the experiment, so a world is the natural unit."""
    result = empirical.execute_protocol(
        "exp",
        replications=1,
        initializations=["data/world-a.json", "data/world-b.json"],
    )
    assert len(result["runs"]) == 4  # 2 worlds x 2 conditions
    experiments = {
        empirical.get_run(r)["manifest"]["randomness_inputs"]["experiment_id"]
        for r in result["runs"]
    }
    assert experiments == {"exp-world-a", "exp-world-b"}
    seeds = {
        json.dumps(empirical.get_run(r)["manifest"]["seeds"], sort_keys=True)
        for r in result["runs"]
    }
    assert len(seeds) == 4


def test_a_world_seeds_the_population_it_names(empirical: GenesisService) -> None:
    empirical.execute_protocol(
        "exp", initializations=[{"id": "w-b", "data_source": "data/world-b.json"}]
    )
    history = empirical.persistence.list_state_history("exp-w-b-base-1")
    seeded = history[0][1]["population"]
    assert seeded["data_source"] == "data/world-b.json"
    assert seeded["rows"][0]["seed"] == 2


def test_a_world_naming_an_asset_the_build_does_not_carry_is_refused(
    empirical: GenesisService,
) -> None:
    with pytest.raises(ValueError, match="does not carry"):
        empirical.execute_protocol("exp", initializations=["data/world-z.json"])


def test_an_empty_world_list_is_refused_rather_than_read_as_none(
    empirical: GenesisService,
) -> None:
    """Otherwise 'no worlds' would silently mean 'the package's own'."""
    with pytest.raises(ValueError, match="omit it"):
        empirical.execute_protocol("exp", initializations=[])


def test_duplicate_world_ids_are_refused(empirical: GenesisService) -> None:
    with pytest.raises(ValueError, match="duplicate or reserved"):
        empirical.execute_protocol(
            "exp",
            initializations=[
                {"id": "same", "data_source": "data/world-a.json"},
                {"id": "same", "data_source": "data/world-b.json"},
            ],
        )


# --- a build is nameable the way it was created -----------------------------------


def test_a_run_may_name_its_build_the_way_compile_was_asked_for_it(
    service: GenesisService,
) -> None:
    """``/compile`` takes 'builds/x' and recorded the absolute path, so naming
    the build the same way it was created was refused as unknown."""
    service.create_run({"id": "relative", "study_id": "exec-study", "build": "builds/exec-study"})
    result = service.execute_protocol("relative")
    assert result["runs"] == ["relative-base-1", "relative-alt-1"]


def test_a_build_outside_the_workspace_is_still_refused(service: GenesisService) -> None:
    with pytest.raises(ValueError):
        service.create_run({"id": "escapee", "study_id": "exec-study", "build": "../../elsewhere"})


# --- the package has no say in how many draws to take -----------------------------


def test_a_package_declaring_a_draw_count_is_refused_with_somewhere_to_put_it(
    tmp_path: Path,
) -> None:
    """Dropping it silently would quietly change how much of the study runs."""
    from genesis.compiler import StudyCompiler

    source = tmp_path / "legacy"
    source.mkdir()
    for name in ("study", "openness", "theory", "domain", "protocol", "outcomes", "models"):
        (source / f"{name}.yaml").write_text('schema_version: "1.0"\nstudy_id: legacy\n')
    (source / "protocol.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: legacy\n'
        "time_model:\n  type: rounds\n  end: 2\nreplications: 3\n"
    )
    with pytest.raises(ValueError) as raised:
        StudyCompiler(source).compile(tmp_path / "build")
    message = str(raised.value)
    assert "SPEC_KEY_RETIRED:protocol.replications" in message
    assert "chosen when the study is run" in message
    # Reported once, as a retirement -- not also as an anonymous extra key.
    assert "extra_forbidden" not in message


def test_the_authoring_api_refuses_it_for_the_same_stated_reason(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        with pytest.raises(ValueError, match="chosen when the study is run"):
            service.create_specification(
                {
                    "id": "legacy-study",
                    "title": "legacy",
                    "protocol": {
                        "time_model": {"type": "rounds", "end": 2},
                        "replications": 3,
                    },
                }
            )
    finally:
        service.close()


def test_a_build_carries_no_draw_count_at_all(service: GenesisService) -> None:
    """So the build digest means 'one study', not 'one sample size'."""
    protocol = json.loads(
        (service.workspace / "builds" / "exec-study" / "protocol.json").read_text()
    )
    assert "replications" not in protocol


# --- the elicitation no longer spends turns on numbers ----------------------------


def _experiment_stage():
    from genesis.elicitation import WorkflowRegistry
    from genesis.service import _workflows_root

    return WorkflowRegistry(_workflows_root()).get("three-layer-study").stage("experiment-design")


def test_the_experiment_stage_no_longer_elicits_operational_settings() -> None:
    """A researcher states these as numbers; there is nothing to convert."""
    stage = _experiment_stage()
    decisions = {decision.id for decision in stage.critical_decisions}
    assert "operational-controls" not in decisions
    targets = {path for decision in stage.critical_decisions for path in decision.target_paths}
    for operational in (
        "/protocol/replications",
        "/protocol/budgets",
        "/protocol/checkpoints",
        "/protocol/replay_retention",
    ):
        assert operational not in targets


def test_it_still_elicits_what_makes_the_cells_comparable() -> None:
    """Streams and matching are scientific: they say what is held identical."""
    stage = _experiment_stage()
    decision = next(d for d in stage.critical_decisions if d.id == "randomness-and-matching")
    assert set(decision.target_paths) == {"/protocol/random_streams", "/protocol/matching"}
    assert decision.required


def test_the_stage_says_plainly_that_counts_are_not_its_business() -> None:
    stage = _experiment_stage()
    assert "not when it is specified" in stage.instructions
    assert "replications" not in stage.ambiguity_topics


# --- the Run view can ask for a plan before spending anything ----------------------


def test_the_protocol_route_plans_and_runs(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    workspace = _workspace(tmp_path)
    client = TestClient(create_app(workspace))
    planned = client.post("/runs/exp/protocol", json={"replications": 2, "plan": True})
    assert planned.status_code == 200
    assert planned.json()["runs"] == 4
    assert client.get("/runs/exp-base-1").status_code != 200  # nothing dispatched

    ran = client.post("/runs/exp/protocol", json={"replications": 2})
    assert ran.status_code == 200
    assert len(ran.json()["runs"]) == 4


def test_the_protocol_route_refuses_an_option_it_does_not_support(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    client = TestClient(create_app(_workspace(tmp_path)))
    response = client.post("/runs/exp/protocol", json={"draws": 3})
    assert response.status_code >= 400
    assert "draws" in response.json()["error"]["message"]


def test_planning_writes_nothing_at_all(tmp_path: Path) -> None:
    """Asking what a choice would cost must not commit to it."""
    genesis = GenesisService(_workspace(tmp_path))
    try:
        before = len(genesis.persistence.list_runs())
        genesis.execute_protocol("exp", replications=5, plan_only=True)
        assert len(genesis.persistence.list_runs()) == before
        with pytest.raises(KeyError):
            genesis.persistence.get_experiment("exp")
    finally:
        genesis.close()


# --- an event cap is a spend guard the run sets, and it must leave a mark --------


def test_a_run_is_capped_by_the_run_not_the_package(service: GenesisService) -> None:
    result = service.execute_protocol("exp", max_events=1)
    capped = service.get_run(result["runs"][0])
    assert capped["max_events"] == 1


def test_the_plan_reports_the_cap_before_anything_is_spent(service: GenesisService) -> None:
    plan = service.execute_protocol("exp", max_events=25, plan_only=True)
    assert plan["max_events"] == 25
    assert service.execute_protocol("exp", plan_only=True)["max_events"] is None


def test_a_cap_below_one_event_is_refused(service: GenesisService) -> None:
    with pytest.raises(ValueError, match="max_events must be at least 1"):
        service.execute_protocol("exp", max_events=0)


def test_a_run_cut_short_by_its_cap_still_reads_as_completed(tmp_path: Path) -> None:
    """The status cannot distinguish them -- hitting the cap marks the run
    completed, exactly as reaching the declared termination does -- so the mark
    beside it is the only thing that can."""
    natural, _ = _capped_run(tmp_path, rounds=3, cap=None)
    events, marks = _capped_run(tmp_path, rounds=3, cap=natural - 1)
    assert events < natural
    assert marks == ["max_events"]


def test_a_run_that_reached_its_own_termination_carries_no_such_mark(
    service: GenesisService,
) -> None:
    result = service.execute_protocol("exp")
    run = service.get_run(result["runs"][0])
    assert run["status"] == "completed"
    assert all("stopped_by" not in entry for entry in run.get("executions", []))


def test_the_protocol_route_passes_the_cap_through(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    client = TestClient(create_app(_workspace(tmp_path)))
    planned = client.post("/runs/exp/protocol", json={"max_events": 3, "plan": True})
    assert planned.status_code == 200
    assert planned.json()["max_events"] == 3


# --- declarations that were never read ------------------------------------------


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        ("budgets", "pass max_events to the run"),
        ("checkpoints", "nothing has ever read this"),
        ("replay_retention", "nothing has ever read this"),
    ],
)
def test_a_protocol_declaring_a_retired_operational_key_is_refused(
    tmp_path: Path, key: str, reason: str
) -> None:
    from genesis.compiler import StudyCompiler

    source = tmp_path / key
    source.mkdir()
    for name in ("study", "openness", "theory", "domain", "protocol", "outcomes", "models"):
        (source / f"{name}.yaml").write_text('schema_version: "1.0"\nstudy_id: legacy\n')
    (source / "protocol.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: legacy\n'
        f"time_model:\n  type: rounds\n  end: 2\n{key}: {{}}\n"
    )
    with pytest.raises(ValueError) as raised:
        StudyCompiler(source).compile(tmp_path / f"build-{key}")
    assert f"SPEC_KEY_RETIRED:protocol.{key}" in str(raised.value)
    assert reason in str(raised.value)


def test_nothing_in_the_system_reads_a_checkpoint_or_retention_declaration() -> None:
    """The reason they were retired rather than moved: there was no consumer to
    move them to, so declaring one promised a policy that did not exist."""
    import genesis.service as service_module
    from genesis.specification.models import ProtocolSpec

    assert not {"budgets", "checkpoints", "replay_retention"} & set(ProtocolSpec.model_fields)
    source = Path(service_module.__file__).read_text()
    assert 'protocol.get("budgets"' not in source


# --- the manifest says how the study was actually realised ------------------------


def test_the_manifest_states_the_cap_including_when_there_is_none(
    service: GenesisService,
) -> None:
    """Stated, not implied by absence: a reader must not have to guess whether
    an old manifest predates the field or the run simply had no cap."""
    capped = service.execute_protocol("exp", max_events=7, only_conditions=["base"])
    assert service.get_run(capped["runs"][0])["manifest"]["realisation"]["max_events"] == 7
    plain = service.execute_protocol("exp", only_conditions=["alt"])
    realisation = service.get_run(plain["runs"][0])["manifest"]["realisation"]
    assert realisation["max_events"] is None
    assert realisation["initialization"] is None
    assert realisation["replication"] == 1


def test_the_manifest_records_the_world_the_run_was_actually_seeded_from(
    empirical: GenesisService,
) -> None:
    """data_provenance held the package's declaration, so a reader could not
    tell which population produced these numbers."""
    result = empirical.execute_protocol("exp", initializations=["data/world-b.json"])
    manifest = empirical.get_run(result["runs"][0])["manifest"]
    assert manifest["realisation"]["initialization"]["data_source"] == "data/world-b.json"
    provenance = manifest["data_provenance"]
    assert provenance["run_initialization"]["data_source"] == "data/world-b.json"
    # The package's own declaration is still there, and still says world-a.
    assert provenance["initialization"]["data_source"] == "data/world-a.json"


def test_a_capped_run_is_not_the_same_effective_configuration_as_an_uncapped_one() -> None:
    """Held at the resolver so only the cap varies: comparing two conditions
    would differ in their digests whether or not the cap counted."""
    from genesis.execution_manifest import resolve_execution_manifest, scientific_config_digest

    base: dict = dict(
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
    uncapped = resolve_execution_manifest(**base)
    capped = resolve_execution_manifest(**base, max_events=7)
    other_cap = resolve_execution_manifest(**base, max_events=8)
    assert "max_events" not in uncapped
    assert capped["max_events"] == 7
    assert scientific_config_digest(capped) != scientific_config_digest(uncapped)
    assert scientific_config_digest(capped) != scientific_config_digest(other_cap)


def test_a_world_changes_the_effective_configuration_at_the_resolver_too() -> None:
    from genesis.execution_manifest import resolve_execution_manifest, scientific_config_digest

    base: dict = dict(
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
    plain = resolve_execution_manifest(**base)
    world_a = resolve_execution_manifest(**base, initialization_digest="aaa")
    world_b = resolve_execution_manifest(**base, initialization_digest="bbb")
    assert "initialization_digest" not in plain
    assert scientific_config_digest(world_a) != scientific_config_digest(plain)
    assert scientific_config_digest(world_a) != scientific_config_digest(world_b)


def test_two_worlds_are_not_the_same_effective_configuration(
    empirical: GenesisService,
) -> None:
    """The world used to reach the digest only through the experiment id, which
    scientific_config_digest drops as lineage."""
    from genesis.execution_manifest import scientific_config_digest

    result = empirical.execute_protocol(
        "exp", initializations=["data/world-a.json", "data/world-b.json"]
    )
    digests = {
        scientific_config_digest(empirical.get_run(run)["manifest"]["execution"])
        for run in result["runs"]
        if empirical.get_run(run)["manifest"]["condition_id"] == "base"
    }
    assert len(digests) == 2


def test_a_run_choosing_neither_keeps_the_digest_it_had_before(service: GenesisService) -> None:
    """Same back-compatibility contract as executor_code_digest."""
    result = service.execute_protocol("exp", only_conditions=["base"])
    execution = service.get_run(result["runs"][0])["manifest"]["execution"]
    assert "max_events" not in execution
    assert "initialization_digest" not in execution


def test_an_exported_bundle_carries_how_the_run_actually_went(tmp_path: Path) -> None:
    """The manifest is frozen before execution, so a truncation can only be
    reported here; a bundle without it reads as a complete run."""
    service, run_id = _truncated_run(tmp_path)
    try:
        service.export_run(run_id, "exports/capped")
        bundle = service.workspace / "exports" / "capped"
        execution = json.loads((bundle / "run_execution.json").read_text())
        assert any(entry.get("stopped_by") == "max_events" for entry in execution)
    finally:
        service.close()


# --- the cap marks only a run it actually cut short --------------------------------


def _repeating_study(workspace: Path, rounds: int) -> GenesisService:
    """A study whose natural length is more than one event, so a cap can bite."""
    service = GenesisService(workspace)
    try:
        service.create_specification(
            {
                "id": "cap-study",
                "title": "cap",
                "processes": [
                    {
                        "id": "tick",
                        "executor": {},
                        "context_policy": "public",
                        "trigger": {"phase": 0, "repeat": True},
                        "state_effects": [{"field": "counter", "op": "increment"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {"states": [{"id": "counter", "value_type": "integer", "initial": 0}]},
                "protocol": {
                    "time_model": {"type": "rounds", "end": rounds},
                    "conditions": [{"id": "base"}],
                },
                "outcomes": [],
                "models": [],
            }
        )
        version = service.get_specification("cap-study")["version"]
        service.approve_specification("cap-study", version, "researcher")
        compiled = service.compile_study(None, "builds/cap", specification_id="cap-study")
        service.create_run({"id": "e", "study_id": "cap-study", "build": compiled["path"]})
    except Exception:
        service.close()
        raise
    return service


def _capped_run(tmp_path: Path, rounds: int, cap: int | None) -> tuple[int, list[str]]:
    """Run that study under a cap; report how far it got and what was recorded."""
    service = _repeating_study(tmp_path / f"cap-{rounds}-{cap}", rounds)
    try:
        run_id = service.execute_protocol("e", max_events=cap)["runs"][0]
        run = service.get_run(run_id)
        marks = [e.get("stopped_by") for e in run.get("executions", []) if e.get("stopped_by")]
        return len(service.trace_run(run_id)), marks
    finally:
        service.close()


def _truncated_run(tmp_path: Path) -> tuple[GenesisService, str]:
    """A genuinely cut-short run, with its service left open for inspection."""
    natural, _ = _capped_run(tmp_path, rounds=3, cap=None)
    service = _repeating_study(tmp_path / "truncated", 3)
    run_id = service.execute_protocol("e", max_events=natural - 1)["runs"][0]
    return service, run_id


def test_a_cap_equal_to_the_natural_length_is_not_a_truncation(tmp_path: Path) -> None:
    """The check fired on the last natural event, so a complete run was marked
    truncated -- and the only test of the feature passed because of it."""
    natural, _ = _capped_run(tmp_path, rounds=3, cap=None)
    assert natural > 1, "the probe needs a run long enough to cut short"
    events, marks = _capped_run(tmp_path, rounds=3, cap=natural)
    assert events == natural
    assert marks == []


def test_a_cap_above_the_natural_length_is_not_a_truncation(tmp_path: Path) -> None:
    natural, _ = _capped_run(tmp_path, rounds=3, cap=None)
    events, marks = _capped_run(tmp_path, rounds=3, cap=natural + 1)
    assert (events, marks) == (natural, [])


def test_a_cap_below_the_natural_length_is_a_truncation(tmp_path: Path) -> None:
    natural, _ = _capped_run(tmp_path, rounds=3, cap=None)
    events, marks = _capped_run(tmp_path, rounds=3, cap=natural - 1)
    assert events == natural - 1
    assert marks == ["max_events"]


# --- the cap is checked wherever it enters ----------------------------------------


@pytest.mark.parametrize("bad", [0, -3, True, "5", 2.5])
def test_a_cap_that_is_not_a_positive_count_is_refused(
    service: GenesisService, bad: object
) -> None:
    """Only the protocol route checked this, and it stored what it had not
    coerced: "5" passed as an int and was stored as a string, so the plan
    reported a cap the run never applied."""
    with pytest.raises(ValueError, match="max_events"):
        service.execute_protocol("exp", max_events=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -3, True, "5", 2.5])
def test_the_direct_run_path_refuses_the_same_caps(service: GenesisService, bad: object) -> None:
    with pytest.raises(ValueError, match="max_events"):
        service.create_run(
            {
                "id": f"direct-{bad}",
                "study_id": "exec-study",
                "build": "builds/exec-study",
                "max_events": bad,
            }
        )


def test_a_cap_that_passes_is_stored_as_the_number_it_was_checked_as(
    service: GenesisService,
) -> None:
    result = service.execute_protocol("exp", max_events=5, only_conditions=["base"])
    assert service.get_run(result["runs"][0])["max_events"] == 5


def test_re_running_an_existing_run_under_new_settings_is_refused(
    service: GenesisService,
) -> None:
    """Reuse is how a study is extended, but it silently kept the old run's
    world and cap while reporting the new ones as applied."""
    service.execute_protocol("exp", only_conditions=["alt"])
    with pytest.raises(ValueError, match="different max_events"):
        service.execute_protocol("exp", max_events=2, only_conditions=["alt"])


def test_extending_a_study_with_unchanged_settings_still_works(service: GenesisService) -> None:
    first = service.execute_protocol("exp", replications=1)
    second = service.execute_protocol("exp", replications=3)
    assert set(first["runs"]) < set(second["runs"])


# --- a world may only be the build's own data -------------------------------------


def _foreign_asset(tmp_path: Path) -> Path:
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"seed": 999, "actors": ["intruder"]}))
    return outside


def test_an_absolute_path_outside_the_build_is_refused(
    empirical: GenesisService, tmp_path: Path
) -> None:
    """'build_path / x' is just x when x is absolute, so an existence check let
    any readable JSON on the machine become the study's population."""
    with pytest.raises(ValueError, match="does not carry"):
        empirical.execute_protocol(
            "exp", initializations=[str(_foreign_asset(tmp_path))], plan_only=True
        )


def test_a_relative_path_that_climbs_out_of_the_build_is_refused(
    empirical: GenesisService, tmp_path: Path
) -> None:
    outside = _foreign_asset(tmp_path)
    escape = os.path.relpath(outside, empirical.workspace / "builds" / "exec-study")
    with pytest.raises(ValueError, match="does not carry"):
        empirical.execute_protocol("exp", initializations=[escape], plan_only=True)


def test_a_run_created_directly_cannot_smuggle_a_world_in(
    empirical: GenesisService, tmp_path: Path
) -> None:
    """This path never passes through world resolution, so it had no check."""
    outside = _foreign_asset(tmp_path)
    empirical.create_run(
        {
            "id": "smuggled",
            "study_id": "exec-study",
            "build": "builds/exec-study",
            "initialization": {"id": "x", "data_source": str(outside)},
        }
    )
    with pytest.raises(ValueError, match="RUN_INITIALIZATION"):
        empirical.execute_run("smuggled")


def test_the_build_s_own_asset_is_still_accepted(empirical: GenesisService) -> None:
    result = empirical.execute_protocol("exp", initializations=["data/world-b.json"])
    history = empirical.persistence.list_state_history(result["runs"][0])
    assert history[0][1]["population"]["rows"][0]["seed"] == 2


# --- a replay must reproduce what was realised ------------------------------------


def test_a_replay_is_seeded_from_the_world_its_source_ran_on(empirical: GenesisService) -> None:
    """A FULL replay of a world-b run was seeded from the package's world-a:
    different data, different digest, while claiming to reproduce the run."""
    from genesis.replay import ReplayMode

    source = empirical.execute_protocol(
        "exp", initializations=["data/world-b.json"], only_conditions=["base"]
    )["runs"][0]
    child = empirical.replay_run(source, mode=ReplayMode.FULL, justification="check")
    seeded = empirical.persistence.list_state_history(child["run_id"])[0][1]["population"]
    assert seeded["data_source"] == "data/world-b.json"
    assert seeded["rows"][0]["seed"] == 2


def test_a_replay_inherits_the_cap_its_source_ran_under(tmp_path: Path) -> None:
    from genesis.replay import ReplayMode

    service, run_id = _truncated_run(tmp_path)
    try:
        cap = service.get_run(run_id)["max_events"]
        child = service.replay_run(run_id, mode=ReplayMode.FULL, justification="check")
        assert service.get_run(child["run_id"])["max_events"] == cap
    finally:
        service.close()


def test_planning_refuses_a_bad_cap_before_any_run_exists(service: GenesisService) -> None:
    """Run creation also validates, so only the planning path proves the
    protocol route checks the cap itself."""
    with pytest.raises(ValueError, match="max_events"):
        service.execute_protocol("exp", max_events=0, plan_only=True)
    with pytest.raises(ValueError, match="max_events"):
        service.execute_protocol("exp", max_events="5", plan_only=True)  # type: ignore[arg-type]


def test_reaching_the_cap_is_only_a_truncation_when_work_was_denied() -> None:
    """The rule both dispatch paths share. The concurrent batch path reaches it
    with its own evidence (a non-empty queue), which no test here exercises."""
    from genesis.runtime import RunController

    controller = RunController.__new__(RunController)
    controller.budget_exhausted = False
    controller.status = "running"
    controller._stop_on_budget(work_remains=False)
    assert (controller.status, controller.budget_exhausted) == ("completed", False)
    controller._stop_on_budget(work_remains=True)
    assert controller.budget_exhausted is True


def test_an_empty_condition_list_is_refused_rather_than_read_as_none(
    service: GenesisService,
) -> None:
    """It dispatched nothing and reported completed, which is what a finished
    study looks like."""
    with pytest.raises(ValueError, match="omit it"):
        service.execute_protocol("exp", only_conditions=[])


def test_a_build_compiled_before_the_split_is_refused_not_silently_ignored(
    service: GenesisService,
) -> None:
    """Such a build still declares replications/budgets and nothing reads them,
    so it would run one draw where it says N, and uncapped where it says a cap."""
    # A build's files are immutable, so stage a copy the way a legacy build
    # would have been compiled.
    import shutil

    legacy = service.workspace / "builds" / "legacy"
    shutil.copytree(service.workspace / "builds" / "exec-study", legacy)
    protocol_path = legacy / "protocol.json"
    protocol_path.chmod(0o644)
    protocol = json.loads(protocol_path.read_text())
    protocol_path.write_text(json.dumps({**protocol, "replications": 3}))
    service.create_run({"id": "legacy-run", "study_id": "exec-study", "build": str(legacy)})
    with pytest.raises(ValueError, match="which the run now decides"):
        service.execute_protocol("legacy-run", plan_only=True)


def test_a_world_means_nothing_to_a_package_that_seeds_nothing(service: GenesisService) -> None:
    """It was accepted and written into the manifest, so the provenance named a
    data asset the run never read."""
    with pytest.raises(ValueError, match="does not initialise from data"):
        service.execute_protocol("exp", initializations=["data/world-a.json"], plan_only=True)


def _legacy_build(service: GenesisService, **settings: object) -> Path:
    """A build as an older compiler produced it, integrity manifest included."""
    import hashlib
    import shutil

    legacy = service.workspace / "builds" / f"legacy-{'-'.join(sorted(settings))}"
    shutil.copytree(service.workspace / "builds" / "exec-study", legacy)
    for entry in legacy.rglob("*"):
        if entry.is_file():
            entry.chmod(0o644)
    protocol_path = legacy / "protocol.json"
    protocol_path.write_text(
        json.dumps({**json.loads(protocol_path.read_text()), **settings}, indent=2, sort_keys=True)
    )
    manifest_path = legacy / "integrity_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol.json"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return legacy


def test_a_legacy_build_is_refused_on_the_direct_run_path_too(service: GenesisService) -> None:
    """The refusal lived only in execute_protocol, so the same build that the
    protocol route rejected ran uncapped through create_run + execute_run --
    silently losing a spend guard the package declared."""
    legacy = _legacy_build(service, budgets={"max_events": 1})
    service.create_run({"id": "direct", "study_id": "exec-study", "build": str(legacy)})
    with pytest.raises(ValueError, match="which the run now decides"):
        service.execute_run("direct")
    # And refused before the transition, so the run is left runnable.
    assert service.get_run("direct")["status"] == "created"
