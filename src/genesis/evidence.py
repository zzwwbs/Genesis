"""Capability-labelled evidence exchange (G4).

Exploration and reproducibility are distinct, capability-labelled bundles:
``exploration`` lets a researcher inspect retained evidence and previously
calculated results, while ``reproducibility`` additionally requires the
run-pinned package closure, build, execution manifest and checkpoint evidence
so a compatible local runtime could reconstruct inputs. Each bundle carries a
machine-readable capability evaluation; a requested full export fails with a
completeness report rather than silently downgrading.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BUNDLE_VERSION = 1

# The capability vocabulary (spec §4.2). Each is individually evaluated.
CAPABILITY_ORDER = (
    "inspect",
    "read_outcome_snapshot",
    "recompute_outcomes",
    "reexecute",
    "replay_recorded",
    "branch_at_checkpoint",
)


class ExportMode:
    EXPLORATION = "exploration"
    REPRODUCIBILITY = "reproducibility"


def evaluate_capabilities(
    *,
    has_build: bool,
    has_closure: bool,
    has_recorded_outputs: bool,
    has_checkpoint_evidence: bool,
    has_outcomes: bool,
) -> list[dict[str, Any]]:
    """Individually evaluate every capability with missing prerequisites.

    A bundle may support reexecution without exact recorded replay when
    provider responses were not retained, and vice versa (spec §4.2).
    """
    capabilities: list[dict[str, Any]] = []

    def add(capability: str, available: bool, missing: list[str], reason: str) -> None:
        capabilities.append(
            {
                "capability": capability,
                "available": available,
                "missing": missing,
                "reason": reason,
            }
        )

    add(
        "inspect",
        True,
        [],
        "events and artifacts are retained for inspection",
    )
    add(
        "read_outcome_snapshot",
        has_outcomes,
        [] if has_outcomes else ["outcomes"],
        "stored outcome rows are available for reading"
        if has_outcomes
        else "no stored outcome snapshot was retained",
    )
    missing_recompute: list[str] = []
    if not has_build:
        missing_recompute.append("build")
    if not has_outcomes:
        missing_recompute.append("outcomes")
    add(
        "recompute_outcomes",
        has_build and has_outcomes,
        missing_recompute,
        "outcome rows can be recomputed from the pinned outcome plan and retained evidence"
        if has_build and has_outcomes
        else "the build (outcome plan) or stored outcomes are missing",
    )
    missing_reexecute = []
    if not has_closure:
        missing_reexecute.append("package_closure")
    if not has_build:
        missing_reexecute.append("build")
    add(
        "reexecute",
        has_build and has_closure,
        missing_reexecute,
        "a compatible local runtime could reconstruct supported execution inputs"
        if has_build and has_closure
        else "executable build or package closure is missing",
    )
    missing_replay = []
    if not has_build:
        missing_replay.append("build")
    if not has_recorded_outputs:
        missing_replay.append("recorded_outputs")
    add(
        "replay_recorded",
        has_build and has_recorded_outputs,
        missing_replay,
        "recorded invocations can be reused under the pinned build"
        if has_build and has_recorded_outputs
        else "recorded outputs or build are missing",
    )
    missing_branch = []
    if not has_build:
        missing_branch.append("build")
    if not has_checkpoint_evidence:
        missing_branch.append("checkpoint_evidence")
    add(
        "branch_at_checkpoint",
        has_build and has_checkpoint_evidence,
        missing_branch,
        "recorded checkpoints support branching"
        if has_build and has_checkpoint_evidence
        else "checkpoint evidence is not available in this bundle",
    )
    return capabilities


def write_bundle_manifest(
    destination: Path,
    *,
    export_mode: str,
    run_id: str,
    source_run_id: str,
    local_import_id: str | None,
    package_digest: str,
    build_digest: str,
    scientific_config_digest: str,
    capabilities: list[dict[str, Any]],
    omissions: list[str],
    retention_policy: str,
) -> Path:
    """EVD-001: one machine-readable bundle manifest with digest-backed members."""
    members = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        relative = path.relative_to(destination).as_posix()
        members.append(
            {
                "path": relative,
                "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
                "media_type": _media_type(path),
            }
        )
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "export_mode": export_mode,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "local_import_id": local_import_id,
        "package_digest": package_digest,
        "build_digest": build_digest,
        "scientific_config_digest": scientific_config_digest,
        "members": members,
        "capabilities": capabilities,
        "omissions": omissions,
        "retention_policy": retention_policy,
    }
    target = destination / "bundle_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target


def _is_contained_relative(relative: str) -> bool:
    """A member path must be a clean, bundle-relative POSIX path (EVD-T05)."""
    if not relative or relative.startswith(("/", "\\")):
        return False
    parts = Path(relative).parts
    if not parts or ".." in parts or "." in parts:
        return False
    if os.path.normpath(relative) != relative:
        return False
    return True


def verify_bundle_manifest(destination: Path) -> dict[str, Any]:
    """Verify every member against the bundle manifest digest table (EVD-004/T05).

    Requires complete member coverage (every file in the bundle directory must
    be listed), rejects unsafe/absolute paths and symlinks, and verifies actual
    member sizes so an omitted member or a size-limit bypass cannot pass
    preflight while other files enter the workspace.
    """
    manifest_path = destination / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("IMPORT_BUNDLE: bundle_manifest.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("IMPORT_BUNDLE: bundle manifest is invalid JSON") from exc
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            f"IMPORT_BUNDLE: unsupported bundle_version "
            f"{manifest.get('bundle_version')}; expected {BUNDLE_VERSION}"
        )
    seen: set[str] = set()
    for member in manifest.get("members", []):
        if not isinstance(member, Mapping):
            raise ValueError("IMPORT_BUNDLE: member entries must be objects")
        relative = str(member.get("path", ""))
        if not _is_contained_relative(relative):
            raise ValueError(f"IMPORT_BUNDLE: unsafe member path '{relative}'")
        if relative in seen:
            raise ValueError(f"IMPORT_BUNDLE: duplicate member path '{relative}'")
        seen.add(relative)
        asset = destination / relative
        if not asset.is_file() or asset.is_symlink():
            raise ValueError(f"IMPORT_BUNDLE: member '{relative}' is missing or a link")
        expected = str(member.get("digest", ""))
        actual = hashlib.sha256(asset.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"IMPORT_BUNDLE: member '{relative}' fails its digest")
        declared_size = member.get("size")
        if not isinstance(declared_size, int) or declared_size < 0:
            raise ValueError(f"IMPORT_BUNDLE: member '{relative}' has an invalid size")
        if asset.stat().st_size != declared_size:
            raise ValueError(
                f"IMPORT_BUNDLE: member '{relative}' size differs from the declared size"
            )
    # Complete coverage: every file in the bundle directory (except the
    # manifest itself) must be listed, otherwise a manifest that omits members
    # bypasses integrity and size enforcement for the remaining files.
    present = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    }
    unlisted = sorted(present - seen)
    if unlisted:
        raise ValueError(
            "IMPORT_BUNDLE: incomplete member coverage; unlisted files: " + ", ".join(unlisted[:10])
        )
    return dict(manifest)


def verify_bundle_manifest_and_size(
    destination: Path, *, size_limit_bytes: int
) -> dict[str, Any] | None:
    """Preflight a bundle for import: digests, paths, sizes (EVD-004/T05).

    Returns the verified bundle manifest when the directory carries one;
    otherwise ``None`` (a legacy bundle without bundle_manifest.json is
    accepted for backward compatibility, subject to the legacy integrity
    check). Sizes are summed from the verified member coverage.
    """
    manifest_path = destination / "bundle_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = verify_bundle_manifest(destination)
    total = 0
    for member in manifest.get("members", []):
        total += int(member.get("size", 0))
        if total > size_limit_bytes:
            raise ValueError(f"IMPORT_SIZE: bundle exceeds {size_limit_bytes} bytes")
    return manifest


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


def publish_bundle(staging: Path, destination: Path) -> None:
    """Atomically publish a fully written bundle within one filesystem (spec §4.3).

    Refuses to overwrite an existing destination by default; never merges into
    an old directory and leaves stale files behind.
    """
    if destination.exists():
        raise ValueError(
            f"EXPORT_DESTINATION: refuses to overwrite existing destination {destination}"
        )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)


def stage_bundle(destination: Path) -> Path:
    """Return a fresh staging directory beside the final destination."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    return staging
