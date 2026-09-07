"""Trusted extension manifests and an explicit local registry boundary."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][0-9A-Za-z.-]+)?$")


def _version(value: str) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(value)
    if not match:
        raise ValueError(f"invalid semantic version: {value}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def _satisfies(value: str, expression: str) -> bool:
    current = _version(value)
    for clause in (part.strip() for part in expression.split(",") if part.strip()):
        for operator in (">=", "<=", ">", "<", "=="):
            if clause.startswith(operator):
                target = _version(clause[len(operator) :])
                checks = {
                    ">=": current >= target,
                    "<=": current <= target,
                    ">": current > target,
                    "<": current < target,
                    "==": current == target,
                }
                if not checks[operator]:
                    return False
                break
        else:
            if current != _version(clause):
                return False
    return True


@dataclass(frozen=True)
class ExtensionManifest:
    id: str
    version: str
    genesis_range: str
    schema_range: str
    capabilities: tuple[str, ...]
    entry_point: str
    integrity_hash: str

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError("extension id must be a stable lowercase identifier")
        _version(self.version)
        if not self.entry_point or ":" not in self.entry_point:
            raise ValueError("extension entry_point must use module:attribute form")
        if not self.integrity_hash:
            raise ValueError("extension integrity_hash is required")


class ExtensionRegistry:
    def __init__(self, *, genesis_version: str, schema_version: str):
        _version(genesis_version)
        _version(schema_version)
        self.genesis_version = genesis_version
        self.schema_version = schema_version
        self._entries: dict[str, tuple[ExtensionManifest, Callable[..., Any]]] = {}

    def register(
        self,
        manifest: ExtensionManifest,
        factory: Callable[..., Any],
        *,
        enabled: bool = False,
        integrity_material: bytes | str | None = None,
    ) -> None:
        if not enabled:
            raise PermissionError("workspace extensions require explicit enablement")
        if not _satisfies(self.genesis_version, manifest.genesis_range) or not _satisfies(
            self.schema_version, manifest.schema_range
        ):
            raise ValueError("extension is not compatible with this GENESIS/schema version")
        if manifest.id in self._entries:
            raise ValueError(f"extension already registered: {manifest.id}")
        if integrity_material is None:
            raise ValueError("extension integrity material is required for digest verification")
        material = (
            integrity_material.encode()
            if isinstance(integrity_material, str)
            else integrity_material
        )
        if hashlib.sha256(material).hexdigest() != manifest.integrity_hash:
            raise ValueError("extension digest does not match integrity_hash")
        self._entries[manifest.id] = (manifest, factory)

    def get(self, extension_id: str) -> Callable[..., Any]:
        try:
            return self._entries[extension_id][1]
        except KeyError:
            raise KeyError(f"extension unavailable: {extension_id}") from None

    def manifests(self) -> tuple[ExtensionManifest, ...]:
        return tuple(manifest for manifest, _ in self._entries.values())
