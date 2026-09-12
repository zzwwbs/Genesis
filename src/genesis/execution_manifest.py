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
_CREDENTIAL_MARKERS = (
    "credential",
    "secret",
    "private_key",
    "private-key",
    "api_key",
    "api-key",
    "apikey",
    "api_token",
    "api-token",
    "access_token",
    "access-token",
    "auth_token",
    "auth-token",
    ".key",
    "creds",
)
# Whole file or directory names that denote secrets, compared without suffix.
_CREDENTIAL_STEMS = {"key", "keys", "token", "tokens", "password", "passwords", "env"}
# The asset types each package directory may contribute (spec §2.1). An
# allowlist, so a stray file -- an .env, a key, an editor cache -- cannot enter a
# closure, and with it every build and export, merely by being in the tree.
_ALLOWED_SUFFIXES = {
    "prompts": frozenset({".txt"}),
    "schemas": frozenset({".yaml", ".yml", ".json"}),
    "data": frozenset({".csv", ".json", ".parquet"}),
}


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


def _exclusion_reason(relative: Path) -> str | None:
    """Why a package file must stay out of the closure, or ``None`` (spec §2.1).

    Every path component is checked, not only the file name: a secret is just as
    much a secret inside ``prompts/.credentials/``.
    """
    parts = [part.lower() for part in relative.parts]
    if any(part.startswith(".") for part in parts):
        return "hidden"
    for part in parts:
        if part in _EXCLUDED_NAMES or Path(part).stem in _CREDENTIAL_STEMS:
            return "credential"
        if any(marker in part for marker in _CREDENTIAL_MARKERS):
            return "credential"
    allowed = _ALLOWED_SUFFIXES.get(parts[0]) if len(parts) > 1 else None
    if allowed is not None and relative.suffix.lower() not in allowed:
        return "unsupported file type"
    return None


def _is_excluded(relative: Path) -> bool:
    """Exclude credential-like and unsupported files from the closure (spec §2.1)."""
    return _exclusion_reason(relative) is not None


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
    # Package files left out, with the reason, so an omission is visible rather
    # than silent. Not part of the digest: the closure is what was included.
    excluded: tuple[tuple[str, str], ...] = ()

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
    excluded: list[tuple[str, str]] = []
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
        reason = _exclusion_reason(relative)
        if reason is not None:
            seen.add(normalized)
            excluded.append((normalized, reason))
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
        excluded=tuple(sorted(excluded)),
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
    outcome_plan_digest: str = ""
    runtime_contract_version: int = RUNTIME_CONTRACT_VERSION
    # Source identity of the study code computational and stochastic executors
    # run. Empty when a build runs none, and then left out, so a configuration
    # without study code keeps the digest it had before this field existed.
    executor_code_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self._core_dict()
        if self.executor_code_digest:
            payload["executor_code_digest"] = self.executor_code_digest
        return payload

    def _core_dict(self) -> dict[str, Any]:
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
            # The predeclared measurement plan is part of the effective
            # configuration: two runs measuring different observables are not
            # realizations of the same configured study.
            "outcome_plan_digest": self.outcome_plan_digest,
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
    executor_code_digest: str = "",
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
        outcome_plan_digest=outcome_plan_digest,
        executor_code_digest=executor_code_digest,
    ).to_dict()


EXECUTOR_CODE_MODES = frozenset({"computational", "stochastic"})


def executor_code_reference(process: Mapping[str, Any]) -> str | None:
    """The ``module:attribute`` a computational or stochastic process runs, or None."""
    binding = process.get("executor")
    if not isinstance(binding, Mapping):
        return None
    mode = str(binding.get("mode", ""))
    if mode not in EXECUTOR_CODE_MODES:
        return None
    parameters = binding.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    key = "function" if mode == "stochastic" else "entry_point"
    return str(parameters.get(key) or "") or None


def executor_code_digest(identity: Mapping[str, Mapping[str, Any]]) -> str:
    """One digest over the study code a build runs, where it could be located.

    A process whose module could not be found contributes nothing: whether a
    module happens to be importable is a property of the machine, not of the
    configuration, and it must not change the configuration's identity.
    """
    resolved = {
        process_id: record
        for process_id, record in identity.items()
        if isinstance(record, Mapping) and record.get("source_sha256")
    }
    if not resolved:
        return ""
    return hashlib.sha256(canonical_json(resolved).encode()).hexdigest()


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
