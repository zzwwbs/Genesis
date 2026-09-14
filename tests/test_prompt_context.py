"""A model can only see the context its prompt puts in front of it.

The request a model receives is the rendered prompt and nothing else: context
enters only through `{context}` or `{context.<path>}`. Every prompt in the
clickbait packages was a structured prompt *specification* written to disk as a
Python dict repr, with no placeholder -- so each model answered blind, with no
round number, no feed, no leaderboard and no governance notice, whatever its
context policy allowed. A paid run surfaced it: the model filled `phase` from
the only thing that looked like one, `'process': 'publish-article'`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.compiler import StudyCompiler, ValidationIssue
from tests.test_engine_gaps import _write_package


def _codes(tmp_path: Path, prompt: str, policy: str = "view") -> list[str]:
    source = _write_package(
        tmp_path,
        {
            "openness": {
                "processes": [
                    {
                        "id": "write",
                        "executor": {"mode": "generative", "model_profile": "mp"},
                        "openness_rationale": "what is written is the phenomenon",
                        "closure_rationale": "the schema closes it",
                        "context_policy": policy,
                        "prompt_ref": "write",
                        "trigger": {"type": "phase", "phase": 0},
                        "outputs": [{"artifact_type": "note", "schema_ref": "note"}],
                    }
                ]
            },
            "domain": {
                "states": [{"id": "feed", "value_type": "array", "initial": []}],
                "visibility": [{"id": "view", "allow": ["feed"]}],
                "artifacts": [{"id": "note", "artifact_type": "note", "schema_ref": "note"}],
            },
            "models": {"models": [{"id": "mp", "provider": "openai-compatible", "model": "m"}]},
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 0}},
        },
        "prompt-study",
    )
    (source / "schemas").mkdir(exist_ok=True)
    (source / "schemas" / "note.yaml").write_text("type: object\nproperties: {t: {type: string}}\n")
    (source / "prompts").mkdir(exist_ok=True)
    (source / "prompts" / "write.txt").write_text(prompt)
    try:
        StudyCompiler(source).compile(tmp_path / "build")
    except ValidationIssue as issue:
        return [item.code for item in issue.issues]
    return []


def test_a_prompt_that_never_shows_the_context_is_refused(tmp_path: Path) -> None:
    assert "PROMPT_OMITS_CONTEXT" in _codes(tmp_path, "Write one note for round {phase}.")


def test_a_prompt_spec_saved_as_a_dict_repr_is_refused(tmp_path: Path) -> None:
    """The shape every clickbait prompt had."""
    spec = "{'prompt_ref': 'write', 'objective': 'Write one note.', 'output_schema': 'note'}"
    assert "PROMPT_OMITS_CONTEXT" in _codes(tmp_path, spec)


@pytest.mark.parametrize(
    "prompt",
    ["Write from {context}", "Your feed:\n{context.feed}\nWrite one note."],
)
def test_a_prompt_that_shows_its_context_compiles(tmp_path: Path, prompt: str) -> None:
    assert "PROMPT_OMITS_CONTEXT" not in _codes(tmp_path, prompt)


@pytest.mark.parametrize("policy", ["none", "private", "public"])
def test_a_process_granted_no_context_may_omit_it(tmp_path: Path, policy: str) -> None:
    """The built-in policies allow nothing, so there is nothing to omit."""
    assert "PROMPT_OMITS_CONTEXT" not in _codes(tmp_path, "Write one note.", policy=policy)


# --- the draft path writes prompts as text, or refuses them -----------------------


def test_a_structured_prompt_from_a_draft_is_refused_not_stringified(tmp_path: Path) -> None:
    from genesis.elicitation import write_candidate_prompts

    spec = {"objective": "Write one note.", "output_schema": "note"}
    errors = write_candidate_prompts(tmp_path / "prompts", {"write": spec, "ok": "Use {context}"})
    assert [error["code"] for error in errors] == ["PROMPT_CONTENT"]
    assert not (tmp_path / "prompts" / "write.txt").exists(), "a repr was written as the prompt"
    assert (tmp_path / "prompts" / "ok.txt").read_text() == "Use {context}"


def test_an_empty_prompt_from_a_draft_is_refused(tmp_path: Path) -> None:
    from genesis.elicitation import write_candidate_prompts

    assert [e["code"] for e in write_candidate_prompts(tmp_path / "p", {"w": "  "})] == [
        "PROMPT_CONTENT"
    ]
