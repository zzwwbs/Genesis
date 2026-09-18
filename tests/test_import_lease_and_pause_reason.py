"""Three findings from the 2026-09-14 full-scale review.

An imported run's id was interpolated into a build path unchecked, so a bundle
could write a complete executable build outside the workspace. A lease lost in
the final phase left a finished run 'running' and reported the cell as failed.
And a pause reason carried the provider's response body, which routinely echoes
the request -- and so the prompt -- into the persisted run record.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from genesis.provider_errors import provider_pause_reason
from genesis.service import GenesisService

EXECUTABLE_FILES = (
    "processes.json",
    "process_graph.json",
    "context_policies.json",
    "state_model.json",
    "artifact_catalog.json",
    "outcome_plan.json",
    "protocol.json",
    "build_manifest.json",
    "validation_report.json",
)


def _bundle(root: Path) -> Path:
    source = root / "bundle"
    source.mkdir(parents=True)
    for name in EXECUTABLE_FILES:
        payload = {"build_hash": "deadbeefcafe"} if name == "build_manifest.json" else {"p": 1}
        (source / name).write_text(json.dumps(payload))
    return source


def test_an_imported_run_id_cannot_escape_the_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = _bundle(root)
        service = GenesisService(root / "ws")
        try:
            with pytest.raises(ValueError, match="IMPORT_RUN"):
                service._reconstruct_imported_build(source, None, "../../pwned")
            # An ordinary id still reconstructs its build.
            build_ref, restored = service._reconstruct_imported_build(source, None, "good-run")
        finally:
            service.close()
        assert restored
        assert Path(build_ref).name == "good-run-imported-deadbeef"
        assert not (root / "pwned-imported-deadbeef").exists()


@pytest.mark.parametrize("name", ["..", ".", "a/b", "a\\b", ""])
def test_every_unsafe_id_shape_is_refused(name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = _bundle(root)
        service = GenesisService(root / "ws")
        try:
            with pytest.raises(ValueError, match="IMPORT_RUN"):
                service._reconstruct_imported_build(source, None, name)
        finally:
            service.close()


def test_a_pause_reason_carries_no_provider_body() -> None:
    """A provider echoing the request must not put the prompt in the record."""
    secret = "TOP-SECRET-PROMPT-abc123"
    body = json.dumps({"error": "no credit", "request": {"messages": [{"content": secret}]}})
    reason = provider_pause_reason(ValueError(f"PROVIDER_HTTP: provider returned HTTP 402: {body}"))
    assert reason is not None
    assert reason["kind"] == "provider_credit"
    assert reason["status"] == 402
    assert secret not in reason["error"]
    # What GENESIS itself wrote survives, so the researcher still knows why.
    assert reason["error"] == "PROVIDER_HTTP: provider returned HTTP 402"

    retried = provider_pause_reason(
        ValueError(f"PROVIDER_HTTP: provider returned HTTP 429 after 3 retries: {body}")
    )
    assert retried is not None
    assert secret not in retried["error"]
    assert retried["error"].endswith("after 3 retries")

    unavailable = provider_pause_reason(
        ValueError(f"PROVIDER_UNAVAILABLE: provider unreachable after 3 retries: {body}")
    )
    assert unavailable is not None
    assert secret not in unavailable["error"]


def _run_until_lease_lost(terminal_phase: int, lose_after_phase: int) -> Any:
    """Run until the lease moves on just after the given phase."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
    )

    class Works:
        def execute(self, invocation):
            return ProcessResult(outputs={})

    status = ["running"]
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "actors": None,
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": True},
                }
            ]
        ),
        ExecutorRegistry({"p": Works()}),
        ContextEngine({"private": {"allow": []}}),
        status_provider=lambda: status[0],
    )
    original = controller._poll_external_status

    def poll() -> bool:
        if int(controller._next_phase or 0) > lose_after_phase:
            status[0] = "lease_lost"
        return original()

    controller._poll_external_status = poll  # type: ignore[method-assign]
    controller.run("lease-run", phase_limit=6, seed=1, terminal_phase=terminal_phase)
    return controller


def test_a_lease_lost_after_the_last_phase_leaves_no_work_remaining() -> None:
    controller = _run_until_lease_lost(terminal_phase=2, lose_after_phase=2)
    assert controller.lease_lost
    # The declared work is done, so _dispatch_run settles the run instead of
    # raising RUN_LEASE_LOST and leaving it 'running'.
    assert controller.stopped_with_work_remaining is False


def test_a_lease_lost_mid_run_still_reports_work_remaining() -> None:
    controller = _run_until_lease_lost(terminal_phase=4, lose_after_phase=1)
    assert controller.lease_lost
    assert controller.stopped_with_work_remaining is True
