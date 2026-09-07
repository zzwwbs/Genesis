"""Replay requests and immutable lineage results."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


class ReplayMode(StrEnum):
    FULL = "full"
    ARTIFACT = "artifact"
    PARTIAL = "partial"
    BRANCH = "branch"


@dataclass(frozen=True)
class ReplayRequest:
    source_run_id: str
    mode: ReplayMode
    artifact_ids: tuple[str, ...] = ()
    boundary: str | None = None
    overrides: dict[str, Any] | None = None
    justification: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_ids", tuple(self.artifact_ids))
        object.__setattr__(self, "overrides", _freeze(dict(self.overrides or {})))


@dataclass(frozen=True)
class ReplayResult:
    run_id: str
    source_run_id: str
    mode: ReplayMode
    artifacts: dict[str, Any]
    overrides: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _freeze(dict(self.artifacts)))
        object.__setattr__(self, "overrides", _freeze(dict(self.overrides)))


class ReplayManager:
    @staticmethod
    def _hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def replay(self, request: ReplayRequest, recorded_artifacts: dict[str, Any]) -> ReplayResult:
        if request.mode == ReplayMode.ARTIFACT and not request.artifact_ids:
            raise ValueError("artifact replay requires artifact_ids")
        if request.mode == ReplayMode.PARTIAL and not request.boundary:
            raise ValueError("partial replay requires a boundary")
        if request.mode == ReplayMode.BRANCH:
            if not request.boundary:
                raise ValueError("branch replay requires a boundary")
            if not request.justification:
                raise ValueError("branch replay requires a justification")
        if request.mode == ReplayMode.FULL and (request.artifact_ids or request.boundary):
            raise ValueError("full replay does not accept artifact_ids or a boundary")
        selected = (
            dict(recorded_artifacts)
            if not request.artifact_ids
            else {key: recorded_artifacts[key] for key in request.artifact_ids}
        )
        for key, value in selected.items():
            if not isinstance(value, dict) or "payload" not in value or "hash" not in value:
                raise ValueError(f"recorded artifact hash is required: {key}")
            if self._hash(value["payload"]) != value["hash"]:
                raise ValueError(f"artifact integrity failure: {key}")
            selected[key] = value["payload"]
        return ReplayResult(
            str(uuid.uuid4()),
            request.source_run_id,
            request.mode,
            selected,
            dict(request.overrides or {}),
        )
