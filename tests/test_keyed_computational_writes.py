"""A keyed write declared by study code must survive the fan-out.

The compiler was relaxed to accept ``key: actor`` on a non-model effect and call
it composable, but the runtime honoured a declared op/key only when an executor
returned explicit ``ProcessResult.state_effects``. The normal shape -- study
code returning a mapping -- was written whole, so each actor overwrote the last
under sequential timing, and under simultaneous timing collided at commit after
every call had already run.

This drives a real run rather than re-deriving the branch the fix changed: a
test that restates the implementation agrees with it however wrong it is, which
is how the defect survived being written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from genesis.runtime import _declares_composable_write
from genesis.service import GenesisService
from tests.test_engine_gaps import _write_package

ACTORS = ["u1", "u2", "u3"]


def _package(workspace: Path, effects: list[dict[str, Any]], timing: str) -> Path:
    return _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    {
                        "id": "distribute",
                        "actors": ACTORS,
                        "information_timing": {"mode": timing, "order": "shuffled"},
                        "openness_rationale": "what each user is shown is the phenomenon",
                        "closure_rationale": "the feed is computed by study code",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "sees-feeds",
                        "trigger": {"type": "phase", "phase": 0, "repeat": True},
                        "state_effects": effects,
                    }
                ]
            },
            "domain": {
                "visibility": [{"id": "sees-feeds", "allow": ["feeds"]}],
                "states": [{"id": "feeds", "value_type": "object", "initial": {}}],
            },
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 0}},
        },
        "feed-study",
    )


def _run(
    tmp_path: Path, effects: list[dict[str, Any]], timing: str, value: Any = None
) -> dict[str, Any]:
    service = GenesisService(tmp_path / "ws")
    try:
        source = _package(tmp_path / "ws", effects, timing)
        build = service.compile_study(source, "builds/feed")["path"]
        service.create_run({"id": "r", "study_id": "feed-study", "build": build})
        # Study code returning a plain mapping: the shape that lost the key.
        result = service.execute_run(
            "r",
            executor_overrides={
                "distribute": lambda inv: {
                    "feeds": value(inv.actor_ids[0]) if value else [f"for-{inv.actor_ids[0]}"]
                }
            },
        )
        assert result["status"] == "completed", result
        return dict(list(service.persistence.list_state_history("r"))[-1][1])
    finally:
        service.close()


@pytest.mark.parametrize("timing", ["sequential", "simultaneous"])
def test_each_fanned_out_actor_keeps_its_own_entry(tmp_path: Path, timing: str) -> None:
    state = _run(tmp_path, [{"field": "feeds", "op": "put", "key": "actor"}], timing)
    assert state["feeds"] == {actor: [f"for-{actor}"] for actor in ACTORS}


def test_a_whole_field_write_is_unchanged(tmp_path: Path) -> None:
    """Only a key or an accumulating op needs the declaration interpreted; a
    bare set must keep writing the field whole."""
    state = _run(
        tmp_path,
        [{"field": "feeds", "op": "set"}],
        "sequential",
        value=lambda actor: {"whole": actor},
    )
    assert state["feeds"] == {"whole": ACTORS[-1]}


def test_only_a_key_or_an_accumulating_op_needs_interpreting() -> None:
    assert _declares_composable_write([{"field": "feeds", "op": "put", "key": "actor"}])
    assert _declares_composable_write([{"field": "notes", "op": "append"}])
    assert not _declares_composable_write([{"field": "feeds", "op": "set"}])
    assert not _declares_composable_write(["feeds"])
    assert not _declares_composable_write([])


# --- exactly which shapes this changed --------------------------------------------


def test_an_executor_returning_explicit_effects_is_untouched() -> None:
    """The commit that made this change claimed no existing package changes
    behaviour. That was asserted without checking and is false as a statement
    about the code: a process declaring a non-set op and returning a plain
    mapping does change. It is true of the packages on disk, and this records
    why -- every affected declaration there is served by an executor that
    returns explicit state_effects, or by a no-op, or names a module that is not
    present.
    """
    import inspect

    from genesis import runtime

    # These build their own effects, so the branch never applies to them.
    assert "state_effects=" in inspect.getsource(runtime.StateTransitionExecutor)
    # A rule executor does not, so a rule declaring a non-set op is a shape that
    # this change does alter. No package on disk uses one.
    assert "state_effects=" not in inspect.getsource(runtime.RuleExecutor)


def test_the_change_is_scoped_to_declarations_that_ask_for_it() -> None:
    """A bare set keeps whole-field semantics, so a package that declares
    nothing unusual cannot be affected however its executor returns."""
    assert not _declares_composable_write([{"field": "f", "op": "set"}])
    assert not _declares_composable_write([{"field": "f"}])
    assert not _declares_composable_write(["f"])
    assert _declares_composable_write([{"field": "f", "op": "append"}])
    assert _declares_composable_write([{"field": "f", "key": "actor"}])
