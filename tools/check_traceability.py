"""Validate the requirement-to-evidence matrix required by SDD Appendix B."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REQUIREMENT_ID = re.compile(r"\b(?:ACC|IEL|SYS|LIFE|L1|L2|L3|REP|AST|PROV|OUTC|XL)-\d{3}\b")


def _load_matrix(path: Path) -> dict[str, Any]:
    """Load the matrix through PyYAML, with a small dependency-free fallback."""
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except ModuleNotFoundError:
        return _load_matrix_fallback(path)


def _load_matrix_fallback(path: Path) -> dict[str, Any]:
    """Handle the repository matrix when optional tooling is not installed locally."""
    entries: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    pending_key: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = re.match(r"\s+(ACC-\d{3}):", line)
        if requirement:
            current = requirement.group(1)
            entries[current] = {}
            inline = re.search(r"\{(.*?)\}", line)
            if inline:
                for key, value in re.findall(r"(tests|inspections):\s*\[(.*?)\]", inline.group(1)):
                    entries[current][key] = [
                        item.strip() for item in value.split(",") if item.strip()
                    ]
            pending_key = None
            continue
        evidence = re.match(r"\s+(tests|inspections):\s*\[(.*?)\]", line)
        if evidence and current:
            values = [
                item.strip().strip("'\"") for item in evidence.group(2).split(",") if item.strip()
            ]
            entries[current][evidence.group(1)] = values
            pending_key = None
            continue
        evidence_map = re.match(r"\s+(tests|inspections):\s*$", line)
        if evidence_map and current:
            pending_key = evidence_map.group(1)
            entries[current][pending_key] = []
            continue
        item = re.match(r"\s+-\s+(.+?)\s*$", line)
        if item and current and pending_key:
            entries[current][pending_key].append(item.group(1).strip("'\""))
    return {"requirements": entries}


def check_traceability(
    specification: Path, matrix: Path, repository_root: Path | None = None
) -> list[str]:
    """Return deterministic errors for missing or empty normative-requirement evidence.

    Every normative family (ACC, IEL, SYS, LIFE, L1, L2, L3, REP, AST, PROV,
    OUTC, XL) referenced in the specification must resolve in the matrix.
    """
    required = sorted(set(REQUIREMENT_ID.findall(specification.read_text(encoding="utf-8"))))
    entries = _load_matrix(matrix).get("requirements", {})
    if not isinstance(entries, dict):
        entries = {}
    root = (repository_root or Path.cwd()).resolve()
    errors: list[str] = []
    for requirement_id in required:
        evidence = entries.get(requirement_id)
        if not isinstance(evidence, dict):
            errors.append(f"{requirement_id}: missing traceability entry")
            continue
        tests = evidence.get("tests", [])
        inspections = evidence.get("inspections", [])
        if not tests and not inspections:
            errors.append(f"{requirement_id}: no test or inspection evidence")
            continue
        for evidence_path in [*tests, *inspections]:
            if not isinstance(evidence_path, str):
                errors.append(f"{requirement_id}: evidence path is not a string")
                continue
            candidate = (root / evidence_path).resolve()
            if not candidate.is_relative_to(root) or not candidate.exists():
                errors.append(f"{requirement_id}: evidence path does not exist: {evidence_path}")
    return errors


def main() -> int:
    """Validate the repository's current specification matrix."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("GENESIS_Complete_System_Design_Specification.md")
    )
    parser.add_argument("--matrix", type=Path, default=Path("docs/requirements-traceability.yaml"))
    args = parser.parse_args()
    errors = check_traceability(args.spec, args.matrix)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
