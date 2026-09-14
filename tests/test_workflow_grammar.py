"""The guided workflow must only ask for things the schema can hold.

A stage's `target_paths` say where an answer will be written. `/protocol/interventions`
named a section ProtocolSpec has no field for, so a researcher's answer about how a
treatment reaches an actor went to a path the authoring API drops -- silently, because
nothing checked the two against each other.

The parse test is here for a duller reason: a malformed stage file fails every test
that builds a service, which is 300+ failures pointing anywhere but the broken file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from genesis.compiler import CANONICAL
from genesis.service import _workflows_root

ROOT = Path(_workflows_root()) / "three-layer-study"
STAGES = sorted((ROOT / "stages").glob("*.yaml"))


@pytest.mark.parametrize("path", [*STAGES, *sorted((ROOT / "templates").glob("*.yaml"))])
def test_every_workflow_file_parses(path: Path) -> None:
    assert yaml.safe_load(path.read_text()) is not None, path.name


@pytest.mark.parametrize("path", STAGES)
def test_every_target_path_names_a_field_the_schema_holds(path: Path) -> None:
    stage = yaml.safe_load(path.read_text())
    unknown: list[str] = []
    for decision in stage.get("critical_decisions") or ():
        for target in decision.get("target_paths") or ():
            section, _, field = str(target).lstrip("/").partition("/")
            model = CANONICAL.get(section)
            if model is None or not field:
                continue
            root = field.split("/", 1)[0]
            if root not in model.model_fields:
                unknown.append(f"{decision['id']} -> {target}")
    assert not unknown, f"{path.name} writes answers nowhere: {unknown}"


def test_the_workflow_never_teaches_a_gate_spelling_that_cannot_fire() -> None:
    """Every instruction and template wrote `condition.<factor>`, which the
    scheduler never resolves -- which is how a real study's treatments came to
    be inert. A factor's level is under `condition.factors`."""
    import re

    offenders = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".yaml", ".yml"}:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                for match in re.finditer(r"condition\.([a-z<][\w<>-]*)", line):
                    if (
                        match.group(1) not in {"factors", "id"}
                        and "never fires" not in line
                        and "resolves to nothing" not in line
                    ):
                        offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
