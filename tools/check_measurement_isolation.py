"""Compiled-context check: measurement processes must never enter causal context.

A measurement process observes the simulated world to produce a research
observable; its output must not become an input to the behaviour it measures.
This check enforces that for ANY package, driven by the package's own
declarations rather than hardcoded study identifiers:

* a process is measurement when it declares ``measurement: true`` (or names a
  measurement role) in openness.yaml;
* its declared outputs are measurement artifacts;
* those artifacts must not appear in any context policy other than the
  measurement processes' own input-only policies;
* a measurement process must declare no state effects (which would feed
  downstream causal state) and must read only ``inputs``.

Runs against the SOURCED package yaml (pre-compile) via the same serialization
path the compiler uses; exits non-zero on violation.

    python tools/check_measurement_isolation.py [package_dir ...]

With no argument every package under demos/ and tests/golden_studies/ that
declares at least one measurement process is checked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Declarations that mark a process as measurement rather than causal.
MEASUREMENT_KEYS = ("measurement", "is_measurement")
MEASUREMENT_ROLES = {"measurement", "observer", "detector", "coder"}


def _default_packages() -> list[Path]:
    candidates = [
        *sorted((ROOT / "demos").glob("*-package")),
        *sorted((ROOT / "tests" / "golden_studies").glob("*")),
    ]
    return [path for path in candidates if (path / "openness.yaml").is_file()]


def _is_measurement(process: dict[str, Any]) -> bool:
    for key in MEASUREMENT_KEYS:
        if process.get(key) is True:
            return True
    role = str(process.get("role", "") or process.get("process_role", "") or "").lower()
    return role in MEASUREMENT_ROLES


def _declared_outputs(process: dict[str, Any]) -> set[str]:
    outputs = set()
    for output in process.get("outputs") or []:
        if isinstance(output, dict) and output.get("artifact_type"):
            outputs.add(str(output["artifact_type"]))
        elif isinstance(output, str):
            outputs.add(output)
    return outputs


def check_package(package: Path) -> list[str]:
    """Return the isolation violations for one package (empty when clean)."""
    domain = yaml.safe_load((package / "domain.yaml").read_text()) or {}
    openness = yaml.safe_load((package / "openness.yaml").read_text()) or {}

    visibility = {
        str(v["id"]): set(v.get("allow") or ())
        for v in domain.get("visibility", [])
        if isinstance(v, dict) and v.get("id")
    }
    processes = {
        str(p["id"]): p
        for p in openness.get("processes", [])
        if isinstance(p, dict) and p.get("id")
    }
    measurement = {pid: p for pid, p in processes.items() if _is_measurement(p)}
    if not measurement:
        return []

    measurement_artifacts: set[str] = set()
    for process in measurement.values():
        measurement_artifacts |= _declared_outputs(process)
    measurement_policies = {
        str(process.get("context_policy"))
        for process in measurement.values()
        if process.get("context_policy")
    }

    violations: list[str] = []
    # 1) measurement artifacts absent from every context policy except the
    #    measurement processes' own input-only policies
    for policy_id, allowed in visibility.items():
        if policy_id in measurement_policies:
            continue
        leaks = sorted(measurement_artifacts & allowed)
        if leaks:
            violations.append(f"policy {policy_id} exposes measurement artifacts {leaks}")

    # 2) measurement processes: no state effects, narrow policies, inputs only
    for pid, process in sorted(measurement.items()):
        if process.get("state_effects"):
            violations.append(f"{pid} declares state effects: {process['state_effects']}")
        policy = str(process.get("context_policy", ""))
        extra = sorted(visibility.get(policy, set()) - {"inputs"})
        if extra:
            violations.append(f"{pid} policy {policy} allows non-input context {extra}")

    # 3) no measurement artifact may be a declared input of a causal process
    for pid, process in sorted(processes.items()):
        if pid in measurement:
            continue
        declared_inputs = {str(item) for item in process.get("inputs") or []}
        consumed = sorted(measurement_artifacts & declared_inputs)
        if consumed:
            violations.append(f"causal process {pid} consumes measurement artifacts {consumed}")
    return violations


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    packages = [Path(arg) for arg in argv] if argv else _default_packages()
    checked = 0
    failures = 0
    for package in packages:
        if not (package / "openness.yaml").is_file():
            print(f"SKIP {package}: no openness.yaml")
            continue
        violations = check_package(package)
        openness = yaml.safe_load((package / "openness.yaml").read_text()) or {}
        has_measurement = any(
            _is_measurement(p) for p in openness.get("processes", []) if isinstance(p, dict)
        )
        if not has_measurement:
            print(f"SKIP {package.name}: declares no measurement process")
            continue
        checked += 1
        if violations:
            failures += 1
            print(f"MEASUREMENT_ISOLATION_VIOLATIONS ({package.name}):")
            for item in violations:
                print(" -", item)
        else:
            print(
                f"MEASUREMENT_ISOLATION_OK ({package.name}): measurement outputs never "
                "enter causal context; measurement processes hold no state effects."
            )
    if not checked:
        print("MEASUREMENT_ISOLATION: no package declares a measurement process")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
