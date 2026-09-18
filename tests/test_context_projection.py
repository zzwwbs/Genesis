"""A declared projection must hold wherever the records are actually nested.

The nested-path projection was written and tested against a record whose nested
field is a mapping. The case it exists for -- "an article without its author's
private strategy text" -- is a list of author records, and there the drop was a
silent no-op: the policy compiled, read as a privacy guarantee, and handed the
actor the field it promised to withhold.

So these probe both shapes, and a leak is asserted against the whole projected
value rather than against the field the author happened to think of.
"""

from __future__ import annotations

from typing import Any

import pytest

from genesis.runtime import _project_value

ARTICLE_NESTED_LIST: list[dict[str, Any]] = [
    {"id": "a1", "title": "t", "authors": [{"name": "n1", "private": "SECRET"}]}
]
ARTICLE_NESTED_MAPPING: list[dict[str, Any]] = [
    {"id": "a1", "title": "t", "author": {"name": "n1", "private": "SECRET"}}
]


@pytest.mark.parametrize(
    ("records", "path"),
    [(ARTICLE_NESTED_LIST, "authors.private"), (ARTICLE_NESTED_MAPPING, "author.private")],
)
def test_a_dropped_nested_field_does_not_reach_the_actor(
    records: list[dict[str, Any]], path: str
) -> None:
    projected = _project_value(records, "board", {"drop": [path]})
    assert "SECRET" not in str(projected), projected


@pytest.mark.parametrize(
    ("records", "keep"),
    [
        (ARTICLE_NESTED_LIST, ["id", "authors.name"]),
        (ARTICLE_NESTED_MAPPING, ["id", "author.name"]),
    ],
)
def test_a_kept_nested_field_survives(records: list[dict[str, Any]], keep: list[str]) -> None:
    """The other half: keep silently emptied the nested collection instead."""
    projected = _project_value(records, "board", {"keep": keep})
    assert "SECRET" not in str(projected)
    assert "n1" in str(projected), projected


def test_dropping_the_whole_collection_still_works() -> None:
    projected = _project_value(ARTICLE_NESTED_LIST, "board", {"drop": ["authors"]})
    assert projected == [{"id": "a1", "title": "t"}]


def test_a_record_that_is_not_nested_is_unaffected() -> None:
    records = [{"id": "a1", "title": "t"}]
    assert _project_value(records, "board", {"drop": ["authors.private"]}) == records


def test_order_and_shape_are_preserved_so_the_context_digest_is_stable() -> None:
    projected = _project_value(ARTICLE_NESTED_LIST, "board", {"drop": ["authors.private"]})
    assert [entry["id"] for entry in projected] == ["a1"]
    assert list(projected[0]) == ["id", "title", "authors"]
    assert isinstance(projected[0]["authors"], list)


def test_a_projection_reaches_records_nested_inside_a_list_of_lists() -> None:
    """One level of recursion left the rule a silent no-op for this shape.

    A list whose entries are lists of records failed the guard, so the value was
    handed over whole -- declared, compiled, and delivering the field the
    projection promised to withhold.
    """
    from genesis.runtime import _dropped_paths, _kept_paths

    paths = frozenset({"authors.private"})
    nested = {"authors": [[{"name": "n1", "private": "SECRET"}]]}
    assert _dropped_paths(nested, paths) == {"authors": [[{"name": "n1"}]]}
    assert _kept_paths(nested, frozenset({"authors.name"})) == {"authors": [[{"name": "n1"}]]}
    # The flat shape, which already worked, still does.
    assert _dropped_paths({"authors": [{"name": "n1", "private": "S"}]}, paths) == {
        "authors": [{"name": "n1"}]
    }
