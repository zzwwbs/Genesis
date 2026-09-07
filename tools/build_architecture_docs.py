"""Regenerate the module-inventory table inside the architecture document.

Scans ``src/genesis/*.py`` and refreshes the section between the
``<!-- GEN:module-table -->`` marker pair in
``docs/GENESIS_Architecture.md``. The table reports module names, line
counts, the module docstring headline, and the top-level class/function
names, so the architecture document never drifts from the source tree.

Usage::

    python tools/build_architecture_docs.py   # refreshes the table in place
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "GENESIS_Architecture.md"
SOURCE = ROOT / "src" / "genesis"

BEGIN = "<!-- GEN:module-table -->"
END = "<!-- /GEN:module-table -->"

CLASS_RE = re.compile(r"^class (\w+)", re.MULTILINE)
FUNCTION_RE = re.compile(r"^def (\w+)", re.MULTILINE)


def _module_docstring_headline(text: str) -> str:
    """First sentence of the module docstring, or an em dash when absent."""
    match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if not match:
        return "—"
    first_line = match.group(1).strip().splitlines()[0].strip()
    return first_line[:160]


def _key_types(text: str) -> list[str]:
    classes = CLASS_RE.findall(text)
    functions = [name for name in FUNCTION_RE.findall(text) if not name.startswith("_")]
    return [*classes[:6], *functions[:3]]


def build_table() -> str:
    rows = []
    for path in sorted(SOURCE.glob("*.py")):
        if path.name in {"__init__.py", "__main__.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        loc = len(lines)
        headline = _module_docstring_headline(text)
        types = ", ".join(_key_types(text)) or "—"
        esc_headline = headline.replace("|", "\\|")
        esc_types = types.replace("|", "\\|")
        rows.append(f"| `{path.name}` | {loc} | {esc_headline} | {esc_types} |")
    table = [
        "| Module | LOC | Purpose (docstring) | Key types |",
        "|---|---|---|---|",
        *rows,
        (
            "| `workflows/three-layer-study/` | — | Data-driven workflow package: "
            "stages, questions, templates, invalidation rules, JSON contracts, "
            "theory templates | `WorkflowRegistry`, `WorkflowDefinition`, "
            "`WorkflowStage` |"
        ),
    ]
    return "\n".join(table)


def main() -> int:
    doc = DOC.read_text(encoding="utf-8")
    if BEGIN not in doc or END not in doc:
        print(f"markers not found in {DOC}; add {BEGIN!r} / {END!r} first")
        return 1
    table = build_table()
    head, tail = doc.split(BEGIN, 1)
    _, tail = tail.split(END, 1)
    DOC.write_text(encoding="utf-8", data=head + BEGIN + "\n" + table + "\n" + END + tail)
    print(f"refreshed module table in {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
