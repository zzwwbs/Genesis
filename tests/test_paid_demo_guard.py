"""The paid Mode-B runners must not spend merely because someone ran them."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# demos/ is gitignored: the paid runners and their guard exist only in a local
# checkout that has them. On a clean clone there is nothing to guard, so this
# skips rather than failing on files the repository never carried.
pytestmark = pytest.mark.skipif(
    not (ROOT / "demos" / "paid_guard.py").is_file(),
    reason="demos/ is local-only and absent from this checkout",
)
PAID = ["run_mode_b.py", "run_mode_b_full.py", "run_mode_b_large.py"]


def test_the_guard_refuses_without_the_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tested directly, never by executing a runner: if the guard regressed,
    running the suite would itself make the paid calls it exists to prevent."""
    from demos.paid_guard import require_paid_opt_in

    monkeypatch.delenv("GENESIS_ENABLE_PAID_DEMOS", raising=False)
    with pytest.raises(SystemExit) as refused:
        require_paid_opt_in("run_mode_b.py")
    assert refused.value.code == 2
    monkeypatch.setenv("GENESIS_ENABLE_PAID_DEMOS", "1")
    require_paid_opt_in("run_mode_b.py")


@pytest.mark.parametrize("runner", PAID)
def test_the_guard_runs_before_anything_else(runner: str) -> None:
    source = (ROOT / "demos" / runner).read_text()
    main_block = source.split('if __name__ == "__main__":', 1)[1]
    assert main_block.index("require_paid_opt_in") < main_block.index("main()")
