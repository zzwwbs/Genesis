"""An outcome the evaluator cannot compute as declared is refused at compile.

The clickbait study declared ten outcomes. Seven named sources nothing defined
and produced no rows; three declared a join that matched every keyless row to
every other (636 events became 767,016 rows); and `measure: proportion` and its
kin were never read, so each was counted. All compiled clean (2026-09-14).
"""

from __future__ import annotations

from typing import Any

import pytest

from genesis.compiler import _validate_outcomes
from genesis.outcome_plan import materialize_datasets
from genesis.specification.models import DomainSpec, OutcomesSpec

DOMAIN = DomainSpec.model_validate(
    {
        "schema_version": "1.0",
        "study_id": "s",
        "artifacts": [{"id": "detection-result", "schema_ref": "detection-schema"}],
    }
)


def _codes(outcome: dict[str, Any], datasets: list[dict[str, Any]] | None = None) -> list[str]:
    spec = OutcomesSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "s",
            "datasets": datasets or [],
            "outcomes": [{"id": "o", **outcome}],
        }
    )
    return [error["code"] for error in _validate_outcomes(spec, DOMAIN)]


def test_computable_outcomes_pass() -> None:
    assert _codes({"source": "events", "aggregation": {"op": "count", "field": "phase"}}) == []
    assert (
        _codes({"source": "detection-result", "aggregation": {"op": "mean", "field": "score"}})
        == []
    )
    dataset = [
        {"id": "detections", "source": {"kind": "artifacts", "artifact_type": "detection-result"}}
    ]
    assert (
        _codes(
            {
                "source": "detections",
                "grouping": ["phase"],
                "aggregation": {"op": "sum", "field": "score"},
                "missingness": {"policy": "zero"},
                "window": {"time_field": "phase", "start": 1, "end": 10},
            },
            dataset,
        )
        == []
    )


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        ({"source": "detection-rows", "aggregation": {"op": "count"}}, "OUTCOME_SOURCE_UNKNOWN"),
        (
            {"source": "events", "aggregation": {"measure": "proportion", "numerator": "positive"}},
            "OUTCOME_KEY_UNREAD",
        ),
        (
            {"source": "events", "aggregation": {"op": "difference_in_differences", "field": "x"}},
            "OUTCOME_AGGREGATION_UNSUPPORTED",
        ),
        ({"source": "events", "aggregation": {"op": "mean"}}, "OUTCOME_FIELD_MISSING"),
        (
            {"source": "events", "missingness": {"policy": "retain_null"}},
            "OUTCOME_KEY_UNREAD",
        ),
        ({"source": "events", "window": {"type": "round_bins", "size": 10}}, "OUTCOME_KEY_UNREAD"),
        (
            {"source": "events", "join": {"left": "events", "right": "settlements", "on": "id"}},
            "OUTCOME_SOURCE_UNKNOWN",
        ),
    ],
)
def test_an_outcome_computed_as_something_else_is_refused(
    outcome: dict[str, Any], code: str
) -> None:
    assert code in _codes(outcome)


def test_a_join_never_matches_rows_missing_the_key() -> None:
    from genesis.service import _hash_join

    left = [{"invocation_id": None, "a": 1}, {"invocation_id": "i1", "a": 2}]
    right = [{"invocation_id": None, "b": 1}, {"invocation_id": "i1", "b": 2}]
    assert _hash_join(left, right, "invocation_id") == [{"invocation_id": "i1", "a": 2, "b": 2}]


def test_a_process_filtered_dataset_reads_declared_artifacts() -> None:
    """Declared artifacts record producer_process, which the filter never read."""
    plan = {
        "datasets": [
            {
                "id": "d",
                "source": {
                    "kind": "artifacts",
                    "artifact_type": "detection-result",
                    "process": "detect-clickbait",
                },
            }
        ]
    }
    artifacts = [
        {
            "artifact_id": "a1",
            "payload": {
                "declared_artifact_id": "detection-result",
                "producer_process": "detect-clickbait",
                "value": {"score": 3},
            },
        }
    ]
    rows = materialize_datasets(plan, {"artifacts": artifacts, "events": [], "state": []})["d"]
    assert [row["score"] for row in rows] == [3]


def test_two_outcomes_joining_one_pair_on_different_keys_keep_their_own() -> None:
    """The relation name carries `on`, so one declaration cannot win for both.

    Every plan is built before any is evaluated, so a second outcome joining
    events+artifacts on process_id used to overwrite the first's relation and
    both then read the last-declared key (2026-09-14 M5).
    """
    from genesis.service import _joined_relation_name

    by_invocation = _joined_relation_name("events", "artifacts", "invocation_id")
    by_process = _joined_relation_name("events", "artifacts", "process_id")
    assert by_invocation != by_process
    sources = {by_invocation: ["a"], by_process: ["b"]}
    assert len(sources) == 2
