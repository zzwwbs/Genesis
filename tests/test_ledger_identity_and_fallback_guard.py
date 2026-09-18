"""What the ledger records, and which results cross the output-commit boundary.

Two defects from the 2026-09-14 full-scale review. The recorded ``state_delta``
is the authoritative source a partial replay applies, and it was computed with
``!=``: a value that changed only in type or sign was recorded as unchanged, so
a replay diverged from its source in exactly the bytes the commit identity is
digested from. Separately, a ``skip_with_event`` failure policy commits its
declared fallback outputs, but the engine-field guard and the output schema
check were gated on ``succeeded`` alone, so those outputs committed with neither
actor identity nor schema validation.
"""

from __future__ import annotations

from typing import Any

from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    StateStore,
)


class _Ledger:
    """Records what the controller commits, the way persistence would."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []

    def __init_artifacts__(self) -> None:  # pragma: no cover - documentation only
        pass

    def commit_process_result(self, event, state, artifacts, fail_after=None, **_: Any) -> None:
        self.events.append(dict(event))
        self.artifacts.extend(dict(a) for a in artifacts or [])

    def latest_json_state(self, run_id):  # pragma: no cover - not exercised here
        return None

    def list_events(self, run_id):
        return list(self.events)

    def list_artifacts(self, run_id):
        return []


def test_a_transition_that_changes_only_type_or_sign_reaches_the_ledger() -> None:
    delta = RunController._state_delta
    assert delta({"x": -0.0}, {"x": 0.0}) == {"x": 0.0}
    assert delta({"x": 1}, {"x": 1.0}) == {"x": 1.0}
    assert delta({"x": True}, {"x": 1}) == {"x": 1}
    # Genuinely unchanged values stay out, and a removed key is still recorded.
    assert delta({"x": 1}, {"x": 1}) == {}
    assert delta({"x": 1}, {}) == {"x": None}


def test_the_recorded_delta_replays_to_the_committed_state() -> None:
    """A round that flips -0.0 to 0.0 must be replayable from its own event."""
    # x starts at 0.0; round 1 writes -0.0 and round 2 writes 0.0 back. Both
    # change the stored bytes and neither changes the value.
    values = {1: -0.0, 2: 0.0}

    class Writer:
        def execute(self, invocation):
            return ProcessResult(
                outputs={},
                state_effects=[{"field": "x", "op": "set", "value": values[int(invocation.phase)]}],
            )

    ledger = _Ledger()
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "w",
                    "actors": None,
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": True},
                    "state_effects": [{"field": "x", "op": "set"}],
                }
            ]
        ),
        ExecutorRegistry({"w": Writer()}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({"x": float}, {"x": 0.0}),
        persistence=ledger,
    )
    controller.run("ledger-run", phase_limit=3, seed=1)
    deltas = [
        event["state_delta"]
        for event in ledger.events
        if event.get("process_id") == "w" and "state_delta" in event
    ]
    # Every round changed the stored bytes, so every round records a delta.
    # With ``!=`` the second was recorded as {} and a replay lost the sign.
    assert deltas == [{"x": -0.0}, {"x": 0.0}]


def test_a_skipped_turn_commits_no_outputs_at_all() -> None:
    """The 2026-09-14 review reported the engine-field guard bypassed here.

    It is bypassed, but nothing escapes through it: the skipped branch commits
    an event with an empty state delta and no artifacts, so a skip_with_event
    policy's fallback_outputs never reach the record. Running the guard on this
    path would only turn a clean skip into a failure. Pinned so the claim is not
    re-raised, and so a future change that *does* commit skipped outputs has to
    face the guard first.
    """
    process = {
        "id": "paired",
        "actors": {
            "source": "population.rows.0.users",
            "id_field": "id",
            "role": "user",
            "per": {
                "source": "clicks.${actor}.opened",
                "id_field": "article_id",
                "role": "article",
            },
        },
        "context_policy": "private",
        "trigger": {"phase": 1, "repeat": True},
        "retry_policy": {
            "max_attempts": 1,
            "failure_policy": "skip_with_event",
            "fallback_outputs": {"reader-response": {"user_id": "wrong", "article_id": "wrong"}},
        },
        "outputs": [
            {
                "artifact_type": "reader-response",
                "schema_ref": "response-schema",
                "actor_fields": {"user_id": "user", "article_id": "article"},
                "phase_fields": ["phase"],
            }
        ],
    }

    class AlwaysFails:
        def execute(self, invocation):
            return ProcessResult(status="failed", outputs={}, metadata={"code": "BOOM"})

    state = {
        "population": {"rows": [{"users": [{"id": "u1"}]}]},
        "clicks": {"u1": {"opened": [{"article_id": "a-1-w2"}]}},
    }
    seen: list[tuple[str, object]] = []
    ledger = _Ledger()
    controller = RunController(
        Scheduler([process]),
        ExecutorRegistry({"paired": AlwaysFails()}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({"population": dict, "clicks": dict}, dict(state)),
        output_schema_validator=lambda ref, value: seen.append((ref, value)) or [],
        persistence=ledger,
    )
    controller.run("skip-run", phase_limit=2, seed=1)

    assert [event["kind"] for event in ledger.events] == ["process_skipped"]
    assert ledger.artifacts == []
    assert ledger.events[0]["state_delta"] == {}
    # Nothing was committed, so nothing needed validating.
    assert seen == []
