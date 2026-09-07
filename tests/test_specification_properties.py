import pytest

hypothesis = pytest.importorskip("hypothesis")
# ruff: noqa
from hypothesis import given, strategies as st

from genesis.specification import StudySpec


@given(st.from_regex(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", fullmatch=True))
def test_study_id_accepts_every_generated_stable_identifier(study_id: str):
    value = StudySpec.model_validate(
        {"schema_version": "1.0", "study_id": study_id, "title": "property test"}
    )
    assert value.study_id == study_id
