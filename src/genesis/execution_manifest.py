"""Immutable package closure and effective execution manifest (G4/G1 foundation).

The package closure snapshots the approved package's canonical specification
files, prompts, schemas and empirical data assets as original bytes with
content digests, so a later export or replay never reads the editable package.
The effective execution manifest resolves one immutable configuration before
run creation or provider calls, with a scientific configuration digest that is
independent of the local run ID, import IDs and creation times.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Versioned canonical hashing rules (spec §2.1): sorted manifest entries,
# UTF-8 JSON, deterministic key order and number representation, SHA-256 of
# original asset bytes. Workspace location and timestamps are never identity
# inputs, so relocating identical contents preserves the digest.
MANIFEST_VERSION = 1
RNG_ALGORITHM_VERSION = "genesis-rng-v1"
STREAM_SCHEME_VERSION = 1
RUNTIME_CONTRACT_VERSION = 1

# Assets that must never enter a package closure (spec §2.1): credentials and
# API keys, and the workspace-internal metadata registry.
_EXCLUDED_NAMES = {".credentials", ".credentials.yaml", ".credentials.yml", "credentials.json"}
_CREDENTIAL_MARKERS = ("credential", "secret", "private_key", "api_key", ".key", "creds")


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "text/yaml"
    if suffix == ".json":
        return "application/json"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".txt":
        return "text/plain"
    if suffix == ".parquet":
        return "application/vnd.apache.parquet"
    return "application/octet-stream"


def _is_excluded(relative: Path) -> bool:
    """Exclude credential-like files from the closure (spec §2.1)."""
    name = relative.name.lower()
    if name in _EXCLUDED_NAMES:
        return True
    if any(marker in name for marker in _CREDENTIAL_MARKERS):
        return True
    return False


def _safe_relative(root: Path, path: Path) -> Path:
    """Return the package-relative path, rejecting traversal and symlink escape."""
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise ValueError(
            f"package asset escapes the package directory: {path} (symlink or traversal)"
        )
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"package asset is outside the package: {path}") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unacceptable package asset path: {relative}")
    return relative


@dataclass(frozen=True)
class PackageClosure:
    """The content-addressed immutable closure of one approved package."""

    source: str
    digest: str
    manifest: dict[str, Any]

    @property
    def assets(self) -> dict[str, dict[str, Any]]:
        return {str(asset["path"]): asset for asset in self.manifest.get("assets", [])}


def canonical_json(value: Any) -> str:
    """Deterministic UTF-8 JSON with sorted keys and compact separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_package_closure(source: str | Path) -> PackageClosure:
    """Resolve one approved package into an immutable content-addressed closure.

    Collects canonical specification files, prompts, schemas and empirical data
    assets as original bytes, recording each normalized path, digest, size and
    media type. Rejects path traversal, absolute paths and symlink escapes, and
    never includes credentials, API keys or unrelated workspace files.
    """
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"package directory does not exist: {root}")
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates: list[Path] = []
    for pattern in ("*.yaml", "*.yml", "*.json", "*.txt"):
        candidates.extend(sorted(root.glob(pattern)))
    for subdir in ("prompts", "schemas", "data"):
        base = root / subdir
        if base.is_dir():
            candidates.extend(sorted(p for p in base.rglob("*") if p.is_file()))
    for path in candidates:
        relative = _safe_relative(root, path)
        normalized = relative.as_posix()
        if normalized in seen:
            continue
        if _is_excluded(relative):
            continue
        seen.add(normalized)
        content = path.read_bytes()
        assets.append(
            {
                "path": normalized,
                "digest": hashlib.sha256(content).hexdigest(),
                "size": len(content),
                "media_type": _media_type(path),
            }
        )
    assets.sort(key=lambda asset: asset["path"])
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "digest_algorithm": "sha256",
        "assets": assets,
    }
    return PackageClosure(
        source=str(root),
        digest=hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        manifest=manifest,
    )


@dataclass(frozen=True)
class ExecutionManifest:
    """One immutable effective configuration resolved before run creation."""

    manifest_version: int = MANIFEST_VERSION
    package_digest: str = ""
    build_digest: str = ""
    protocol_digest: str = ""
    condition_id: str = "base"
    factors: Mapping[str, Any] = field(default_factory=dict)
    replication: int = 1
    origin_experiment_id: str | None = None
    randomness_algorithm: str = RNG_ALGORITHM_VERSION
    stream_scheme_version: int = STREAM_SCHEME_VERSION
    model_configuration_digest: str = ""
    runtime_contract_version: int = RUNTIME_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "package_digest": self.package_digest,
            "build_digest": self.build_digest,
            "protocol_digest": self.protocol_digest,
            "condition": {
                "id": self.condition_id,
                "factors": dict(self.factors),
            },
            "replication": self.replication,
            "origin_experiment_id": self.origin_experiment_id,
            "randomness": {
                "algorithm_version": self.randomness_algorithm,
                "stream_scheme_version": self.stream_scheme_version,
            },
            "model_configuration_digest": self.model_configuration_digest,
            "runtime_contract_version": self.runtime_contract_version,
        }

    def scientific_digest(self) -> str:
        """Digest of the effective configuration, excluding run-local identity.

        Run IDs, import IDs, creation times, workspace locations and lineage
        (origin experiment) must not silently alter a configuration digest:
        lineage is provenance, not effective configuration (spec §2.2).
        """
        payload = dict(self.to_dict())
        payload.pop("origin_experiment_id", None)
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def resolve_execution_manifest(
    *,
    condition_id: str,
    factors: Mapping[str, Any] | None,
    replication: int,
    build_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    package_closure_digest: str,
    protocol_digest: str,
    model_configuration_digest: str,
    outcome_plan_digest: str,
    origin_experiment_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the effective configuration from build and protocol facts.

    This is the manifest resolved *before* run creation or provider calls; the
    persisted effective manifest must include or immutably reference every
    execution-affecting parameter. Returns the dict form stored in a run
    manifest so callers treat one authoritative serialization as the source.
    """
    build_digest = str(build_manifest.get("build_hash") or build_manifest.get("build_digest") or "")
    if not package_closure_digest:
        raise ValueError("EXECUTION_MANIFEST: package closure digest is required")
    if not protocol_digest:
        raise ValueError("EXECUTION_MANIFEST: protocol digest is required")
    _ = outcome_plan_digest
    _ = protocol
    return ExecutionManifest(
        package_digest=package_closure_digest,
        build_digest=build_digest,
        protocol_digest=protocol_digest,
        condition_id=str(condition_id or "base"),
        factors=dict(factors or {}),
        replication=int(replication or 1),
        origin_experiment_id=origin_experiment_id,
        model_configuration_digest=model_configuration_digest,
    ).to_dict()


def scientific_config_digest(manifest: ExecutionManifest | dict[str, Any]) -> str:
    """Scientific configuration digest of a resolved manifest.

    Accepts either a :class:`ExecutionManifest` or the dict form (e.g. the
    ``execution`` block stored inside a run manifest) and ignores run-local
    identity fields.
    """
    if isinstance(manifest, ExecutionManifest):
        return manifest.scientific_digest()
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_id", "generated_at", "seeds", "origin_experiment_id"}
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
