"""The researcher-facing page has to be wired to the workspace it drives.

Every test here started as a defect found by clicking through the page, not by
reading it. The service layer was fine in each case; the page addressed the
wrong element, read a field on a tab the researcher could not see, or offered a
control that no code path ever enabled. None of it was reachable from the
service tests, because those call the API directly with canonical inputs.

So the audits that found them are kept here as rules.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "src" / "genesis" / "static" / "ui.html"
TABS = ("models", "specification", "run", "outputs")


@pytest.fixture(scope="module")
def page() -> str:
    return UI.read_text()


def _tab_of_line(page: str) -> dict[int, str | None]:
    """Which tab panel each line of markup belongs to (None once script starts)."""
    mapping: dict[int, str | None] = {}
    current: str | None = None
    for number, line in enumerate(page.splitlines(), 1):
        found = re.search(r'<section id="tab-([a-z]+)" class="tab-panel', line)
        if found:
            current = found.group(1)
        if line.strip().startswith("<script"):
            current = None
        mapping[number] = current
    return mapping


def _elements(page: str, pattern: str) -> dict[str, str | None]:
    tabs = _tab_of_line(page)
    found: dict[str, str | None] = {}
    for number, line in enumerate(page.splitlines(), 1):
        for match in re.finditer(pattern, line):
            found[match.group(1)] = tabs[number]
    return found


def _handlers(page: str) -> dict[str, str]:
    """Each top-level function name mapped to its body, by brace matching."""
    bodies: dict[str, str] = {}
    for match in re.finditer(r"(?:async )?function ([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{", page):
        start = match.end()
        depth, index = 1, start
        while index < len(page) and depth:
            depth += {"{": 1, "}": -1}.get(page[index], 0)
            index += 1
        bodies[match.group(1)] = page[start:index]
    return bodies


def _buttons(page: str) -> set[tuple[str, str]]:
    """(tab, handler) for every onclick inside a tab panel."""
    tabs = _tab_of_line(page)
    found = set()
    for number, line in enumerate(page.splitlines(), 1):
        if tabs[number] is None:
            continue
        for match in re.finditer(r'onclick="([a-zA-Z0-9_]+)\(', line):
            found.add((tabs[number], match.group(1)))
    return found


def test_a_control_reports_its_result_on_the_tab_it_lives_on(page: str) -> None:
    """Seven controls wrote into a <pre> on a tab the researcher was not looking
    at, so clicking them did nothing visible."""
    homes = _elements(page, r'<pre id="([a-z0-9-]+)"')
    bodies = _handlers(page)
    misrouted = [
        f"{tab}/{handler} writes to {target}, which lives in {homes[target]}"
        for tab, handler in sorted(_buttons(page))
        for target in sorted(set(re.findall(r"show\('([a-z0-9-]+)'", bodies.get(handler, ""))))
        if homes.get(target, tab) != tab
    ]
    assert not misrouted, "\n".join(misrouted)


def test_every_offered_control_can_be_reached(page: str) -> None:
    """'Compile approved draft' was rendered disabled and nothing anywhere ever
    enabled it, so the Run tab's primary action was unreachable by any path."""
    disabled = set(re.findall(r'<button id="([a-zA-Z0-9-]+)"[^>]*\bdisabled\b', page))
    assert disabled, "the audit only means something while some control starts disabled"
    # Compare the assigned value rather than using a lookahead: '\s*' backtracks
    # to zero width, so '(?!true)' happily matches at the space before 'true'.
    unreachable = []
    for button in sorted(disabled):
        assigned = re.findall(rf"getElementById\('{button}'\)\.disabled\s*=\s*([^;]+);", page)
        if not any(value.strip() != "true" for value in assigned):
            unreachable.append(button)
    assert not unreachable, f"never enabled by any code path: {unreachable}"


def test_the_study_identity_is_visible_wherever_it_is_used(page: str) -> None:
    """Handlers on three tabs read this field. While it sat on the Run tab, a
    researcher who imported a package silently addressed the default study."""
    inputs = _elements(page, r'<(?:input|select|textarea) id="([a-zA-Z0-9-]+)"')
    assert inputs.get("study-id") is None, "study-id must live in the header, not inside one tab"
    assert 'id="study-id"' in page.split("</header>")[0]


def test_no_control_depends_on_a_field_the_researcher_cannot_see(page: str) -> None:
    bodies = _handlers(page)
    header_fields = {"study-id"}
    homes = _elements(page, r'<(?:input|select|textarea) id="([a-zA-Z0-9-]+)"')
    offenders = []
    for tab, handler in sorted(_buttons(page)):
        body = bodies.get(handler, "")
        read = set(re.findall(r"getElementById\('([a-zA-Z0-9-]+)'\)\.value(?!\s*=)", body))
        if re.search(r"\bid\(\)", body):
            read.add("study-id")
        offenders += [
            f"{tab}/{handler} reads {field}, which lives in {homes[field]}"
            for field in sorted(read - header_fields)
            if homes.get(field, tab) not in (tab, None)
        ]
    assert not offenders, "\n".join(offenders)


def test_the_outputs_picker_prefers_a_run_that_actually_holds_something() -> None:
    """A protocol's template run never executes; it sorted first and the tab
    opened on an empty trace."""
    body = _handlers(UI.read_text())["refreshRuns"]
    assert "status === 'completed'" in body


def test_the_default_world_is_not_shown_as_its_internal_sentinel() -> None:
    from genesis.service import _DEFAULT_WORLD

    assert _DEFAULT_WORLD not in UI.read_text()
    source = (Path(__file__).resolve().parents[1] / "src" / "genesis" / "service.py").read_text()
    assert "the population the package carries" in source


# --- the checks we built must be reachable from the page --------------------------


def test_the_intent_check_is_shown_and_can_be_acknowledged(page: str) -> None:
    """The gate refuses approval on an unacknowledged contradiction, and the
    page had nothing that could satisfy it: no findings, no acknowledgement."""
    assert "showElicitationTab('intent')" in page
    assert 'id="intent-findings"' in page
    bodies = _handlers(page)
    # Only a contradiction is acknowledgeable; an unsupported finding advises.
    assert "finding.verdict === 'contradicted'" in bodies["renderIntentFindings"]
    assert "acknowledged_findings: acknowledgedIntentFindings()" in bodies["approveElicitation"]


def test_a_blocking_finding_is_not_left_for_the_researcher_to_discover(page: str) -> None:
    bodies = _handlers(page)
    assert "refreshIntentBadge" in bodies["renderElicitation"]
    assert "'intent'" in bodies["renderElicitation"]


def test_the_build_readback_can_be_asked_for_from_the_page(page: str) -> None:
    """readback_build existed in the service with no route and no control."""
    assert "readbackBuild()" in page
    body = _handlers(page)["readbackBuild"]
    assert "/readback" in body
    assert "model_profile_id" in body
    # A reading that cannot be tied to a build and a model is not evidence.
    assert "record.model_profile" in body and "record.build_hash" in body


def test_no_control_calls_a_route_the_app_does_not_serve(page: str) -> None:
    """'Review patch' GET /specifications/{id}/draft, which has never existed,
    so it threw before reaching its own fallback and printed 'Not Found'."""
    import re as _re

    from genesis.app import create_app

    # Read from the app itself: routes registered in a loop (pause, cancel)
    # carry no decorator for a source scan to find.
    served = {getattr(route, "path", "") for route in create_app().routes}

    # A served route may take parameters, so match calls against patterns
    # rather than comparing strings: '/elicitation/workflows/three-layer-study'
    # is a legitimate call to '/elicitation/workflows/{workflow_id}'.
    patterns = [
        _re.compile("^" + _re.sub(r"\\\{[^}]*\\\}", "[^/]+", _re.escape(route)) + "/?$")
        for route in served
    ]

    def served_by_app(call: str) -> bool:
        concrete = _re.sub(r"\$\{[^}]*\}", "x", call)
        return any(pattern.match(concrete) for pattern in patterns)

    called = set(_re.findall(r"request\(`([^`?]+)`", page)) | set(
        _re.findall(r"request\('([^'?]+)'", page)
    )
    missing = sorted({route for route in called if not served_by_app(route)})
    assert not missing, f"the page calls routes the app does not serve: {missing}"


def test_a_compile_failure_attributed_to_an_earlier_stage_is_shown(page: str) -> None:
    """compile_warnings were written into every preview and read by nothing."""
    assert "compile_warnings" in _handlers(page)["showElicitationTab"]


def test_no_value_from_the_wire_is_interpolated_into_markup(page: str) -> None:
    """Suggestion labels are model-produced text; interpolating them into
    innerHTML ran whatever they contained in the workspace's own origin."""
    import re as _re

    offenders = [
        line.strip()
        for line in page.splitlines()
        if _re.search(r"\.innerHTML\s*=", line) and _re.search(r"\$\{", line)
    ]
    assert not offenders, "build these as nodes with textContent:\n" + "\n".join(offenders)
