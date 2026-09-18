"""Two lows and one export contract from the 2026-09-14 full-scale review."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "GENESIS_Architecture.md"


@pytest.mark.skipif(not DOC.is_file(), reason="docs/ is local-only")
def test_the_architecture_module_table_matches_the_source_tree() -> None:
    """The doc claims the table never drifts; nothing checked that it hadn't.

    It listed 15 rows against 26 modules, missing twelve including
    state_encoding and provider_errors.
    """
    text = DOC.read_text(encoding="utf-8")
    block = text.split("<!-- GEN:module-table -->")[1].split("<!-- /GEN:module-table -->")[0]
    listed = {line.split("`")[1] for line in block.splitlines() if line.startswith("| `")}
    modules = {
        path.name
        for path in (ROOT / "src" / "genesis").glob("*.py")
        if path.name not in {"__init__.py", "__main__.py"}
    }
    missing = sorted(modules - listed)
    assert not missing, (
        f"{len(missing)} modules missing from the table: {', '.join(missing)}; "
        "run python tools/build_architecture_docs.py"
    )


def test_the_ui_covers_every_action_the_service_advertises() -> None:
    """edit_draft was emitted in allowed_actions and silently skipped by the UI."""
    import re

    service_text = (ROOT / "src" / "genesis" / "service.py").read_text(encoding="utf-8")
    ui_text = (ROOT / "src" / "genesis" / "static" / "ui.html").read_text(encoding="utf-8")
    advertised = set()
    for match in re.finditer(r"actions\.(?:append|extend)\((.*?)\)", service_text, re.DOTALL):
        advertised.update(re.findall(r'"([a-z_]+)"', match.group(1)))
    handled = set(re.findall(r"^\s*([a-z_]+):", ui_text.split("const actionIds = {")[1], re.M))
    handled.update(re.findall(r"^\s*([a-z_]+):", ui_text.split("const actionHints = {")[1], re.M))
    unhandled = sorted(advertised - handled)
    assert not unhandled, f"advertised but not handled by the UI: {', '.join(unhandled)}"
