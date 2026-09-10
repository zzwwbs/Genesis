"""Executors for the review-finding regressions: a state field and a nested one."""


def stamp(invocation):
    phase = invocation.phase or 0
    return {
        "topic": f"t{phase}",
        "profile": {"tier": "premium" if phase % 2 else "basic", "phase": phase},
    }
