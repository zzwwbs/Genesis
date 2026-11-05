"""The replay modes a run can be replayed in."""

from __future__ import annotations

from enum import StrEnum


class ReplayMode(StrEnum):
    FULL = "full"
    ARTIFACT = "artifact"
    PARTIAL = "partial"
    BRANCH = "branch"
