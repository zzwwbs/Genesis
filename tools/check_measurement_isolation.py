"""Compiled-context check: measurement processes must never enter causal context.

`classify-article` and `evaluate-clickbait` are measurement processes in the
no-governance design: their labels, scores, and rationales must not appear in
any creator- or user-facing context policy, and they must not declare state
effects (which would feed downstream causal state).

Runs against the SOURCED package yaml (pre-compile) via the same serialization
path the compiler uses; exits non-zero on violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "demos" / "large-chain-package"

MEASUREMENT = ("classify-article", "evaluate-clickbait")
MEASUREMENT_POLICY_IDS = ("classifier-context", "detector-context")

# policies whose allowed state/artifact names a measurement-only process may use
ALLOWED_FOR_MEASUREMENT = {"inputs"} | set(MEASUREMENT_POLICY_IDS)


def main() -> int:
    domain = yaml.safe_load((PACKAGE / "domain.yaml").read_text())
    openness = yaml.safe_load((PACKAGE / "openness.yaml").read_text())

    visibility = {v["id"]: set(v.get("allow") or ()) for v in domain["visibility"]}
    processes = {p["id"]: p for p in openness["processes"]}

    violations: list[str] = []
    # 1) measurement artifacts absent from every context policy except the
    #    measurement processes' own input-only policies
    measurement_artifacts = {"classification", "detection"}
    for policy_id, allowed in visibility.items():
        leaks = sorted(measurement_artifacts & allowed)
        if leaks and policy_id not in ALLOWED_FOR_MEASUREMENT:
            violations.append(f"policy {policy_id} exposes measurement artifacts {leaks}")

    # 2) measurement processes: no state effects, narrow policies, inputs only
    for pid in MEASUREMENT:
        process = processes.get(pid)
        if process is None:
            violations.append(f"missing process {pid}")
            continue
        if process.get("state_effects"):
            violations.append(f"{pid} declares state effects: {process['state_effects']}")
        policy = process.get("context_policy")
        allowed = visibility.get(policy, set())
        extra = sorted(allowed - {"inputs"})
        if extra:
            violations.append(f"{pid} policy {policy} allows non-input context {extra}")

    # 3) creator/user-facing policies never see measurement outputs
    for policy_id, allowed in visibility.items():
        if policy_id in ALLOWED_FOR_MEASUREMENT:
            continue
        if allowed & measurement_artifacts:
            violations.append(
                f"{policy_id} allow-list still contains {sorted(allowed & measurement_artifacts)}"
            )

    if violations:
        print("MEASUREMENT_ISOLATION_VIOLATIONS:")
        for item in violations:
            print(" -", item)
        return 1
    print(
        "MEASUREMENT_ISOLATION_OK: classification/detection never enter "
        "creator/user context; classify/evaluate have no state effects."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
