"""End-to-end smoke check of the command line, run in CI.

The test suite exercises the library in-process. This exercises what a user
runs: it creates a workspace, then validates, compiles, runs and evaluates a
small declarative study through the ``genesis`` command line, with no model
provider and no network. It exits non-zero on the first failure and names the
failing step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROUNDS = 3
SOURCE = Path(__file__).resolve().parents[1] / "src"


def fail(message: str) -> None:
    raise SystemExit(f"smoke check failed: {message}")


def genesis(*args: str) -> dict[str, Any]:
    """Run one CLI command and return the JSON it prints last."""
    env = dict(os.environ)
    if (SOURCE / "genesis").is_dir():
        # Prefer the checkout over any other installed copy.
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(SOURCE), env.get("PYTHONPATH", "")) if part
        )
    command = " ".join(args)
    completed = subprocess.run(
        [sys.executable, "-m", "genesis", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        fail(f"`genesis {command}` exited {completed.returncode}\n{completed.stderr[-2000:]}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        fail(f"`genesis {command}` printed nothing")
    try:
        result: dict[str, Any] = json.loads(lines[-1])
    except json.JSONDecodeError:
        fail(f"`genesis {command}` did not print JSON: {lines[-1][:200]}")
        raise
    return result


def write_study(package: Path) -> None:
    """A counter incremented once per round by a declarative executor."""
    base = {"schema_version": "1.0", "study_id": "smoke"}
    files: dict[str, Any] = {
        "study": {**base, "title": "Smoke check"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "tick",
                    "executor": {
                        "mode": "state-transition",
                        "parameters": {
                            "operations": [{"op": "increment", "state": "count", "value": 1}]
                        },
                    },
                    "context_policy": "counter",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "count", "op": "set"}],
                }
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "counter", "allow": ["count"]}],
            "states": [{"id": "count", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": ROUNDS - 1}},
        "outcomes": {
            **base,
            "datasets": [{"id": "final", "source": {"kind": "state", "snapshot": "final"}}],
            "outcomes": [
                {
                    "id": "total",
                    "source": "final",
                    "grouping": [],
                    "aggregation": {"op": "sum", "field": "count"},
                }
            ],
        },
        "models": {**base, "models": []},
    }
    package.mkdir(parents=True)
    for name, content in files.items():
        (package / f"{name}.yaml").write_text(yaml.safe_dump(content, sort_keys=False))


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        workspace = Path(root) / "workspace"
        package = workspace / "studies" / "smoke"
        build = workspace / "builds" / "smoke"

        if genesis("init", str(workspace)).get("status") != "initialized":
            fail("init did not initialise the workspace")
        write_study(package)
        if genesis("validate", str(package)).get("valid") is not True:
            fail("validate rejected the study")
        if not genesis("compile", str(package), "--output", str(build)).get("build_hash"):
            fail("compile produced no build hash")
        run = genesis("run", str(workspace), "--run-id", "smoke-1", "--output", str(build))
        if run.get("status") != "completed":
            fail(f"run ended as {run.get('status')!r}")
        outcomes = genesis("outcomes", str(workspace), "--run-id", "smoke-1").get("outcomes")
        expected = [{"outcome_id": "total", "count_sum": ROUNDS, "count_missing": 0}]
        if outcomes != expected:
            fail(f"outcomes were {outcomes!r}, expected {expected!r}")
    print(f"smoke check passed: {ROUNDS} rounds evaluated to a total of {ROUNDS}")


if __name__ == "__main__":
    main()
