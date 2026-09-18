"""Pressing Pause during a model call, and resuming after a failed attempt.

Two defects from the 2026-09-14 full-scale review, in the feature whose whole
promise is that a run the provider cannot serve pauses instead of failing --
because a failed run cannot be resumed.

The pause path sets the run's shared cancel event, so the in-flight provider
call raises ``PROVIDER_CANCELLED``. That matched no pause branch, so the most
ordinary use of the feature recorded a failed attempt. And the retry loop
restarted at attempt 1 on every dispatch while the ledger already held that
attempt id, so a resume re-emitted it under a different dispatch order and
persistence refused the commit -- leaving the run permanently unresumable.
"""

from __future__ import annotations

from typing import Any

from genesis.provider_errors import is_provider_cancellation, provider_pause_reason
from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
)


class _Ledger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def commit_process_result(self, event, state, artifacts, fail_after=None, **_: Any) -> None:
        event_id = str(event.get("event_id"))
        if any(str(held.get("event_id")) == event_id for held in self.events):
            # What persistence does: the same attempt id committed twice with a
            # different dispatch order is a differing commit, not an idempotent
            # retry.
            raise ValueError("IDEMPOTENCY: process-result commit differs")
        self.events.append(dict(event))

    def latest_json_state(self, run_id):
        return None

    def list_events(self, run_id):
        return list(self.events)

    def list_artifacts(self, run_id):
        return []


PROCESS = {
    "id": "call",
    "actors": None,
    "context_policy": "private",
    "trigger": {"phase": 1, "repeat": True},
    "retry_policy": {"max_attempts": 2},
}


def _controller(executor: Any, ledger: _Ledger, status: list[str]) -> RunController:
    return RunController(
        Scheduler([dict(PROCESS)]),
        ExecutorRegistry({"call": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=ledger,
        status_provider=lambda: status[0],
    )


def test_a_cancellation_is_recognised_as_deliberate() -> None:
    cancelled = ValueError("PROVIDER_CANCELLED: provider request aborted by researcher")
    assert is_provider_cancellation(cancelled)
    # It is not a provider fault, so it carries no provider pause reason.
    assert provider_pause_reason(cancelled) is None
    assert not is_provider_cancellation(ValueError("PROVIDER_HTTP: provider returned HTTP 400"))


def test_pausing_during_a_call_pauses_the_run_instead_of_failing_it() -> None:
    status = ["running"]

    class Paused:
        def execute(self, invocation):
            # What the API does: persist the pause, which sets the run's cancel
            # event, and the in-flight call aborts.
            status[0] = "paused"
            raise ValueError("PROVIDER_CANCELLED: provider request aborted by researcher")

    ledger = _Ledger()
    controller = _controller(Paused(), ledger, status)
    controller.run("pause-run", phase_limit=2, seed=1)

    assert controller.status == "paused"
    assert controller.pause_reason is not None
    assert controller.pause_reason["kind"] == "researcher_paused"
    # No attempt was recorded: nothing it would have produced was committed.
    assert [event["kind"] for event in ledger.events] == []
    assert controller.failures == []


def test_a_cancel_during_a_call_stays_cancelled() -> None:
    status = ["running"]

    class Cancelled:
        def execute(self, invocation):
            status[0] = "cancelled"
            raise ValueError("PROVIDER_CANCELLED: provider request aborted by researcher")

    ledger = _Ledger()
    controller = _controller(Cancelled(), ledger, status)
    controller.run("cancel-run", phase_limit=2, seed=1)

    # Cancelled is final; it must not be downgraded to paused.
    assert controller.status == "cancelled"
    assert ledger.events == []


def test_a_run_paused_after_a_failed_attempt_resumes_at_the_next_attempt() -> None:
    """The case the previous tests missed: a failed attempt 1, then a pause."""
    status = ["running"]
    calls: list[int] = []

    class FailThenOutage:
        def __init__(self) -> None:
            self.seen = 0

        def execute(self, invocation):
            self.seen += 1
            calls.append(invocation.attempt)
            if invocation.attempt == 1:
                # A plain bad response: recorded as a failed attempt.
                return ProcessResult(status="failed", outputs={}, metadata={"code": "BAD"})
            raise ValueError("PROVIDER_HTTP: provider returned HTTP 402: no credit")

    ledger = _Ledger()
    first = _controller(FailThenOutage(), ledger, status)
    first.run("retry-run", phase_limit=2, seed=1)

    assert first.status == "paused"
    assert first.pause_reason["kind"] == "provider_credit"
    # Attempt 1 is on the ledger as a failure; attempt 2 committed nothing.
    assert [(event["attempt"], event["kind"]) for event in ledger.events] == [(1, "process_failed")]

    # Resume in a fresh controller against the same ledger, as a restart does.
    resumed_calls: list[int] = []

    class Succeeds:
        def execute(self, invocation):
            resumed_calls.append(invocation.attempt)
            return ProcessResult(outputs={})

    status[0] = "running"
    second = _controller(Succeeds(), ledger, status)
    second.run("retry-run", phase_limit=2, seed=1)

    # It continues at attempt 2 rather than re-emitting attempt 1's event id.
    assert resumed_calls[0] == 2
    assert second.status != "failed"
    assert [(event["attempt"], event["kind"]) for event in ledger.events] == [
        (1, "process_failed"),
        (2, "process_completed"),
    ]
