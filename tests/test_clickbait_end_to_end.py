from __future__ import annotations

import json
import shutil
from pathlib import Path

from genesis.runtime import ProcessResult
from genesis.service import GenesisService

PACKAGE = Path(__file__).parent / "golden_studies" / "clickbait_mini"


def _context_inputs(invocation) -> dict[str, object]:
    data = invocation.context.data
    return dict(data.get("inputs", {})) if "inputs" in data else {}


def _records(invocation, artifact_type: str) -> list[tuple[str, dict]]:
    return [
        (instance_id, dict(record))
        for instance_id, record in _context_inputs(invocation).items()
        if record.get("artifact_type") == artifact_type
    ]


def _overrides() -> dict[str, object]:
    def form_strategy(invocation):
        creator = invocation.actor_ids[0]
        return {"creator-strategy": {"creator_id": creator, "framing": f"round-{invocation.phase}"}}

    def write_article(invocation):
        creator = invocation.actor_ids[0]
        article_id = f"article-{creator}-{invocation.phase}"
        body_instance_id = f"article-body-{invocation.invocation_id}-attempt-{invocation.attempt}"
        return {
            "article-title": {
                "article_id": article_id,
                "creator_id": creator,
                "title": f"Title by {creator} in round {invocation.phase}",
                "body_instance_id": body_instance_id,
            },
            "article-body": {
                "article_id": article_id,
                "creator_id": creator,
                "body": f"Body by {creator} in round {invocation.phase}",
            },
        }

    def detect(invocation):
        articles = [record["value"] for _id, record in _records(invocation, "article-title")]
        return {
            "clickbait-detection": {
                "detected": bool(articles),
                "items": [
                    {"article_id": article["article_id"], "detected": True, "confidence": 0.9}
                    for article in articles
                ],
            }
        }

    def govern(invocation):
        factors = invocation.condition["factors"]
        active = invocation.phase >= 4 and factors["governance"] != "none"
        return {
            "governance-decision": {
                "penalty": 0.5 if active else 0.0,
                "revenue_delta": -0.5 if active else 0.0,
                "disclosed": active and factors["governance"] == "disclosed",
                "phase": invocation.phase,
            }
        }

    def recommend(invocation):
        titles = _records(invocation, "article-title")
        return {
            "recommendation-set": {
                "user_id": invocation.actor_ids[0],
                "candidates": [
                    {
                        "title_instance_id": instance_id,
                        "body_instance_id": record["value"]["body_instance_id"],
                        "article_id": record["value"]["article_id"],
                        "creator_id": record["value"]["creator_id"],
                    }
                    for instance_id, record in titles
                ],
            }
        }

    def select(invocation):
        recommendation = _records(invocation, "recommendation-set")[0][1]["value"]
        chosen = recommendation["candidates"][0]
        return ProcessResult(
            outputs={
                "title-selection": {
                    "user_id": invocation.actor_ids[0],
                    **chosen,
                }
            },
            events=(
                {
                    "type": "article-selected",
                    "recipient_id": invocation.actor_ids[0],
                    "source_artifact_id": chosen["body_instance_id"],
                },
            ),
        )

    def interpret(invocation):
        body_instance_id, body_record = _records(invocation, "article-body")[0]
        body = body_record["value"]
        return ProcessResult(
            outputs={
                "user-action": {
                    "user_id": invocation.actor_ids[0],
                    "article_id": body["article_id"],
                    "creator_id": body["creator_id"],
                    "action": "follow",
                }
            },
            events=(
                {
                    "type": "article-read",
                    "recipient_id": invocation.actor_ids[0],
                    "source_artifact_id": body_instance_id,
                },
            ),
        )

    def update(invocation):
        data = invocation.context.data
        action = _records(invocation, "user-action")[0][1]["value"]
        follows = list(data["state"]["follows"])
        relation = {"source": action["user_id"], "target": action["creator_id"]}
        if relation not in follows:
            follows.append(relation)
        memory = dict(data["state"]["creator-memory"])
        memory[action["user_id"]] = {
            "creator_id": action["creator_id"],
            "last_phase": invocation.phase,
        }
        return ProcessResult(state_effects={"follows": follows, "creator-memory": memory})

    def expose(invocation):
        actions = _records(invocation, "user-action")
        return {
            "peer-exposure": {
                "creator_id": invocation.actor_ids[0],
                "exposed_action_ids": [instance_id for instance_id, _record in actions],
            }
        }

    return {
        "form-strategy": form_strategy,
        "write-article": write_article,
        "detect-clickbait": detect,
        "apply-governance": govern,
        "recommend-titles": recommend,
        "select-title": select,
        "interpret-article": interpret,
        "update-relations": update,
        "expose-peers": expose,
    }


def _service(tmp_path) -> tuple[GenesisService, str]:
    workspace = tmp_path / "workspace"
    source = workspace / "studies" / "clickbait-mini"
    source.parent.mkdir(parents=True)
    shutil.copytree(PACKAGE, source)
    service = GenesisService(workspace)
    build = service.compile_study("studies/clickbait-mini", "builds/clickbait-mini")
    return service, build["path"]


def test_disclosed_high_scenario_runs_six_rounds_with_sequential_exposure(tmp_path) -> None:
    service, build = _service(tmp_path)
    service.create_run(
        {
            "id": "disclosed-high-run",
            "study_id": "clickbait-mini",
            "build": build,
            "condition_id": "governance-disclosed-peer-visibility-high",
            "condition": {
                "id": "governance-disclosed-peer-visibility-high",
                "factors": {"governance": "disclosed", "peer-visibility": "high"},
            },
            "replication": 1,
        }
    )

    result = service.execute_run("disclosed-high-run", executor_overrides=_overrides())

    assert result["status"] == "completed"
    events = [
        event
        for event in service.trace_run("disclosed-high-run")
        if event["kind"] == "process_completed"
    ]
    assert {event["phase"] for event in events} == {1, 2, 3, 4, 5, 6}
    assert len([event for event in events if event["process_id"] == "form-strategy"]) == 24
    detector_events = [event for event in events if event["process_id"] == "detect-clickbait"]
    assert all("latent-strategies" not in json.dumps(event["context"]) for event in detector_events)
    later_strategy_events = [
        event for event in events if event["process_id"] == "form-strategy" and event["phase"] >= 2
    ]
    assert all(event["context"]["state"]["creator-memory"] for event in later_strategy_events)
    interpretation_events = [
        event for event in events if event["process_id"] == "interpret-article"
    ]
    assert len(interpretation_events) == 48
    assert all(len(event["exposures"]) == 1 for event in interpretation_events)
    assert all(
        event["input_refs"] == [event["exposures"][0]["source_artifact_id"]]
        for event in interpretation_events
    )
    assert all(
        event["exposures"][0]["recipient_ids"] == event["actors"] for event in interpretation_events
    )
    decisions = [
        json.loads(row["payload"])
        for row in service.persistence.list_artifacts("disclosed-high-run")
        if row["artifact_id"].startswith("governance-decision-")
    ]
    assert [decision["value"]["penalty"] for decision in decisions] == [
        0.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.5,
    ]
    _version, state = service.persistence.latest_json_state("disclosed-high-run")
    assert len(state["follows"]) == 8
    assert set(state["creator-memory"]) == {f"user-{index}" for index in range(1, 9)}
    outcomes = service.evaluate_outcomes("disclosed-high-run")
    assert {outcome["outcome_id"] for outcome in outcomes} == {
        "clickbait-rate",
        "creator-revenue",
        "user-action-rate",
    }
    assert all(
        outcome["group_0"] == "governance-disclosed-peer-visibility-high"
        and outcome["group_1"] in {1, 2, 3, 4, 5, 6}
        for outcome in outcomes
    )
    clickbait_rows = [row for row in outcomes if row["outcome_id"] == "clickbait-rate"]
    assert len(clickbait_rows) == 6
    assert all(row["detected_mean"] == 1 for row in clickbait_rows)
    penalty_rows = [row for row in outcomes if row["outcome_id"] == "creator-revenue"]
    assert [row["revenue_delta_sum"] for row in penalty_rows] == [0.0, 0.0, 0.0, -0.5, -0.5, -0.5]
    action_rows = [row for row in outcomes if row["outcome_id"] == "user-action-rate"]
    assert len(action_rows) == 6
    assert all(row["value_count"] == 8 for row in action_rows)
    service.close()


def test_factorial_scenarios_share_matched_seed_and_vary_only_declared_context(tmp_path) -> None:
    service, build = _service(tmp_path)
    service.create_run({"id": "clickbait-experiment", "study_id": "clickbait-mini", "build": build})

    result = service.execute_protocol("clickbait-experiment", executor_overrides=_overrides())

    assert len(result["runs"]) == 6
    trials = [service.get_run(run_id) for run_id in result["runs"]]
    assert len({trial["manifest"]["seeds"]["conventional"] for trial in trials}) == 1
    assert all(trial["manifest"]["model_versions"] for trial in trials)
    assert all(trial["manifest"]["prompt_versions"] for trial in trials)
    initial_worlds = [service.persistence.list_state_history(trial["id"])[0][1] for trial in trials]
    assert all(world == initial_worlds[0] for world in initial_worlds)
    low_trials = [
        trial for trial in trials if trial["condition"]["factors"]["peer-visibility"] == "low"
    ]
    high_trials = [
        trial for trial in trials if trial["condition"]["factors"]["peer-visibility"] == "high"
    ]
    assert all(
        not event["exposures"]
        for trial in low_trials
        for event in service.trace_run(trial["id"])
        if event.get("process_id") == "expose-peers"
    )
    assert all(
        event["exposures"]
        for trial in high_trials
        for event in service.trace_run(trial["id"])
        if event.get("process_id") == "expose-peers"
    )
    detector_counts = {
        trial["id"]: len(
            [
                event
                for event in service.trace_run(trial["id"])
                if event.get("process_id") == "detect-clickbait"
            ]
        )
        for trial in trials
    }
    assert set(detector_counts.values()) == {6}

    def decisions(trial):
        return [
            json.loads(row["payload"])["value"]
            for row in service.persistence.list_artifacts(trial["id"])
            if row["artifact_id"].startswith("governance-decision-")
        ]

    opaque_trials = [
        trial for trial in trials if trial["condition"]["factors"]["governance"] == "opaque"
    ]
    none_trials = [
        trial for trial in trials if trial["condition"]["factors"]["governance"] == "none"
    ]
    assert all(
        decision["penalty"] == 0.5 and decision["disclosed"] is False
        for trial in opaque_trials
        for decision in decisions(trial)[3:]
    )
    assert all(decision["penalty"] == 0.0 for trial in none_trials for decision in decisions(trial))
    service.close()
