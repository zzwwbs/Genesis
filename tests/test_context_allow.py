"""An allow entry must resolve to something the context engine can deliver.

The engine skips a path that is not there, so an entry naming an artifact type,
an attribute or the protocol read as a grant and delivered nothing. The clickbait
detector's policy allowed only `article`; every detector call saw `{}`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from genesis.compiler import _validate_context_allow
from genesis.runtime import CONTEXT_ROOTS, ContextEngine
from genesis.specification.models import DomainSpec


def _domain(*allow: str) -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "s",
            "states": [{"id": "articles", "value_type": "list"}],
            "artifacts": [{"id": "article", "schema_ref": "article-schema"}],
            "attributes": [{"id": "trust"}],
            "visibility": [{"id": "ctx", "allow": list(allow)}],
        }
    )


def _messages(*allow: str) -> list[str]:
    return [error["message"] for error in _validate_context_allow(_domain(*allow))]


def test_resolvable_entries_pass() -> None:
    assert _messages("articles", "state.articles", "state", *CONTEXT_ROOTS, "inputs.article") == []


@pytest.mark.parametrize(
    ("entry", "says"),
    [
        ("article", "artifact type"),
        ("protocol.phase", "{phase}"),
        ("attributes", "attribute"),
        ("trust", "attribute"),
        ("state.nope", "not a declared state"),
        ("settlement", "neither a context namespace"),
    ],
)
def test_unresolvable_entries_are_refused(entry: str, says: str) -> None:
    [message] = _messages("articles", entry)
    assert f"'{entry}'" in message and says in message


def test_roots_match_what_the_engine_delivers() -> None:
    """Every root the check accepts is one the engine actually reads."""
    invocation = SimpleNamespace(
        inputs={"x": 1},
        condition={"x": 1},
        actor_ids=("a",),
        event_history=[{"x": 1}],
        feedback_slots={"x": 1},
        exchanges=[{"x": 1}],
        phase=1,
        invocation_id="i",
    )
    for root in CONTEXT_ROOTS:
        engine = ContextEngine({"p": {"allow": [root]}})
        data = engine.build("p", invocation, {"x": 1}).data  # type: ignore[arg-type]
        assert root in data, root
