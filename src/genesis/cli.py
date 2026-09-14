"""Command-line entry point for GENESIS."""

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from genesis import __version__

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def warn_non_local_binding(host: str) -> bool:
    """Return True and warn when the server is bound to a non-local interface."""
    if host in _LOCAL_HOSTS:
        return False
    print(
        f"WARNING: binding to non-local interface '{host}'. GENESIS is a local-first "
        "research system; confirm this exposure is intended.",
        file=sys.stderr,
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genesis", description="GENESIS research system")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="Print the installed GENESIS version")
    serve = subparsers.add_parser("serve", help="Run the local GENESIS API server")
    serve.add_argument("--host", default="127.0.0.1", help="Interface to bind")
    serve.add_argument("--port", default=8000, type=int, help="TCP port to bind")
    serve.add_argument("--workspace", default=".", help="Local GENESIS workspace")
    for name, help_text in {
        "init": "Create a study workspace",
        "validate": "Validate canonical study files",
        "compile": "Compile a canonical study",
        "run": "Create or execute a local run",
        "status": "Show run status",
        "pause": "Pause a run",
        "resume": "Resume a run",
        "cancel": "Cancel a run",
        "trace": "Show run trace",
        "replay": "Replay a run",
        "outcomes": "Evaluate outcomes",
        "export": "Export local data",
        "import": "Import local data",
        "doctor": "Check local service prerequisites",
        "backup": "Back up the local database",
        "gc": "Report unreferenced object files; remove them with --apply",
        "integrity-check": "Verify a compiled build",
    }.items():
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("path", nargs="?", default=".")
        command.add_argument("--output", "-o", default=None)
        command.add_argument("--run-id", default=None)
        if name == "replay":
            # Parity with the API: the CLI can reach partial and branch replay,
            # not only whole-trajectory reuse.
            command.add_argument(
                "--mode",
                default="full",
                choices=["full", "artifact", "partial", "branch"],
                help="Replay mode",
            )
            command.add_argument(
                "--boundary", default=None, help="Freeze boundary, e.g. 'phase:3' or 'event:<id>'"
            )
            command.add_argument(
                "--artifact-id",
                action="append",
                default=None,
                dest="artifact_ids",
                help="Recorded artifact to retrieve (artifact mode; repeatable)",
            )
            command.add_argument(
                "--justification", default=None, help="Required rationale for a branch"
            )
            command.add_argument(
                "--override",
                action="append",
                default=None,
                dest="overrides",
                metavar="FACTOR=VALUE",
                help="Branchable factor override (branch mode; repeatable)",
            )
        if name == "gc":
            command.add_argument(
                "--apply", action="store_true", help="Remove the unreferenced object files"
            )
        if name == "run":
            command.add_argument(
                "--max-concurrency",
                action="append",
                default=None,
                dest="max_concurrency",
                metavar="N|PROFILE=N",
                help="Concurrent model calls: N for every profile, or PROFILE=N (repeatable)",
            )
        if name == "export":
            command.add_argument(
                "--mode",
                default="exploration",
                choices=["exploration", "reproducibility"],
                help="Bundle capability level",
            )
            command.add_argument(
                "--experiment",
                default=None,
                help="Export every run of this experiment as stacked analysis tables",
            )
            command.add_argument(
                "--analysis-build",
                default=None,
                help="Build whose datasets and outcomes to apply (execution must be identical)",
            )
    return parser


def _max_concurrency(values: list[str] | None) -> int | dict[str, int] | None:
    """Parse ``--max-concurrency`` values; ranges and profiles are checked by the
    service before the run is created (CON-003)."""
    if not values:
        return None
    shared: int | None = None
    per_profile: dict[str, int] = {}
    for value in values:
        profile, separator, number = value.rpartition("=")
        # Plain digits only: int() would also accept "4_0" and " 4".
        if not re.fullmatch(r"[0-9]+", number):
            raise SystemExit(f"--max-concurrency expects N or PROFILE=N, got '{value}'")
        limit = int(number)
        if separator:
            if profile in per_profile:
                raise SystemExit(f"--max-concurrency sets '{profile}' more than once")
            per_profile[profile] = limit
        elif shared is not None:
            raise SystemExit("--max-concurrency N may be given only once")
        else:
            shared = limit
    if shared is not None and per_profile:
        raise SystemExit("--max-concurrency takes either N or PROFILE=N entries, not both")
    return shared if shared is not None else per_profile


def _replay(service: Any, run_id: str, args: Any) -> dict[str, Any]:
    """Run one replay, confirming the digest-bound preview where required."""
    from genesis.replay import ReplayMode

    mode = ReplayMode(str(getattr(args, "mode", "full")))
    overrides: dict[str, Any] = {}
    for item in getattr(args, "overrides", None) or []:
        if "=" not in item:
            raise SystemExit(f"replay override must be FACTOR=VALUE: {item}")
        factor, value = item.split("=", 1)
        overrides[factor] = value
    boundary = getattr(args, "boundary", None)
    artifact_ids = tuple(getattr(args, "artifact_ids", None) or ())
    justification = getattr(args, "justification", None)
    preview_token = None
    if mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH}:
        preview = service.replay_preview(
            run_id,
            mode=mode,
            boundary=boundary,
            overrides=overrides or None,
            justification=justification,
        )
        preview_token = preview["preview_token"]
    return cast(
        "dict[str, Any]",
        service.replay_run(
            run_id,
            mode=mode,
            artifact_ids=artifact_ids,
            boundary=boundary,
            overrides=overrides or None,
            justification=justification,
            preview_token=preview_token,
        ),
    )


def _remove_build(target: Path) -> None:
    """Remove a build directory; a build's integrity files are written read-only."""
    import shutil
    import stat

    def _writable(function: Any, path: str, _exc: Any) -> None:
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        function(path)

    shutil.rmtree(target, onexc=_writable)


def main(argv: Sequence[str] | None = None) -> None:
    """Invoke the GENESIS CLI."""
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(__version__)
    elif args.command == "serve":
        import uvicorn

        from genesis.app import create_app

        warn_non_local_binding(args.host)
        uvicorn.run(create_app(args.workspace), host=args.host, port=args.port)
    elif args.command == "init":
        from genesis.service import GenesisService

        service = GenesisService(args.path)
        try:
            print(json.dumps(service.initialize()))
        finally:
            service.close()
    elif args.command == "validate":
        import shutil
        import tempfile

        from genesis.compiler import StudyCompiler

        # Validation is compilation without retaining a build artifact. Only a
        # directory this command created is removed: deleting whatever --output
        # named wiped a user's existing directory even when compile refused to
        # write into it.
        scratch = None if args.output else Path(tempfile.mkdtemp(prefix="genesis-validate-"))
        target = Path(args.output) if args.output else cast(Path, scratch) / "build"
        created = not target.exists()
        try:
            build = StudyCompiler(args.path).compile(target)
            print(
                json.dumps(
                    {"valid": True, "study_id": build.study_id, "build_hash": build.build_hash}
                )
            )
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
            elif created and target.exists():
                _remove_build(target)
    elif args.command == "compile":
        if not args.output:
            raise SystemExit("compile requires --output")
        source, output = Path(args.path).resolve(), Path(args.output).resolve()
        workspace = Path(os.path.commonpath([source, output]))
        if not (workspace / ".genesis").is_dir():
            # The directory the two paths share is a workspace only if one was
            # initialised there. Treating any shared ancestor as one created
            # `.genesis` wherever they happened to meet -- a home directory, or on
            # macOS `/private`, which is not writable -- and testing only for `/`
            # missed every other case. Without a workspace, compile directly.
            from genesis.compiler import StudyCompiler
            from genesis.elicitation import WorkflowRegistry
            from genesis.service import _workflows_root

            build = StudyCompiler(
                source,
                theory_templates=WorkflowRegistry(_workflows_root()).theory_templates(),
            ).compile(output)
            print(
                json.dumps(
                    {
                        "study_id": build.study_id,
                        "build_hash": build.build_hash,
                        "path": str(build.path),
                        "manifest": build.manifest,
                    }
                )
            )
            return
        from genesis.service import GenesisService

        service = GenesisService(workspace)
        try:
            print(json.dumps(service.compile_study(source, output)))
        finally:
            service.close()
    elif args.command == "integrity-check":
        from genesis.compiler import StudyCompiler

        print(json.dumps({"valid": StudyCompiler.verify_build(args.path)}))
    elif args.command == "doctor":
        from genesis.service import GenesisService

        service = GenesisService(args.path)
        try:
            print(json.dumps(service.doctor()))
        finally:
            service.close()
    elif args.command == "gc":
        from genesis.service import GenesisService

        service = GenesisService(args.path)
        try:
            print(json.dumps(service.collect_garbage(apply=bool(args.apply))))
        finally:
            service.close()
    elif args.command == "backup":
        if not args.output:
            raise SystemExit("backup requires --output")
        from genesis.service import GenesisService

        service = GenesisService(args.path)
        try:
            print(json.dumps(service.backup_run(args.run_id or "run-1", args.output)))
        finally:
            service.close()
    elif args.command in {
        "run",
        "status",
        "pause",
        "resume",
        "cancel",
        "trace",
        "replay",
        "outcomes",
    }:
        from genesis.service import GenesisService

        service = GenesisService(args.path)
        run_id = args.run_id or "run-1"
        try:
            if args.command == "run":
                # The override is parsed and fully validated -- range and profiles --
                # before the run is created, so a rejected value never leaves a stray
                # run behind under the (default) run id.
                max_concurrency = _max_concurrency(getattr(args, "max_concurrency", None))
                existing: dict[str, Any] | None
                try:
                    existing = service.get_run(run_id)
                except KeyError:
                    existing = None
                payload: dict[str, Any] = dict(existing) if existing else {"id": run_id}
                if existing is None and args.output:
                    payload["build"] = args.output
                try:
                    service.validate_execution_options(payload, max_concurrency=max_concurrency)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from None
                if existing is None:
                    service.create_run(payload)
                result = service.execute_run(run_id, max_concurrency=max_concurrency)
            elif args.command == "status":
                result = service.get_run(run_id)
            elif args.command in {"pause", "resume", "cancel"}:
                current = service.get_run(run_id)
                target_status = {
                    "pause": "paused",
                    "resume": "running",
                    "cancel": "cancelled",
                }[args.command]
                result = service.transition_run(run_id, target_status, current["version"])
            elif args.command == "trace":
                result = {"run_id": run_id, "events": service.trace_run(run_id)}
            elif args.command == "outcomes":
                result = {"run_id": run_id, "outcomes": service.evaluate_outcomes(run_id)}
            else:
                result = _replay(service, run_id, args)
            print(json.dumps(result))
        finally:
            service.close()
    elif args.command == "export":
        if not args.output:
            raise SystemExit("export requires --output")
        from genesis.service import GenesisService

        service = GenesisService(args.path)
        try:
            paths = (
                service.export_experiment(
                    args.experiment, args.output, analysis_build=args.analysis_build
                )
                if args.experiment
                else service.export_run(
                    args.run_id or "run-1",
                    args.output,
                    mode=str(getattr(args, "mode", "exploration")),
                )
            )
            print(json.dumps({"paths": [str(path) for path in paths], "status": "exported"}))
        finally:
            service.close()
    elif args.command == "import":
        from genesis.service import GenesisService

        if not args.output:
            raise SystemExit("import requires --output")
        service = GenesisService(args.path)
        try:
            result = service.import_package(args.output, specification_id=args.run_id)
            print(json.dumps({**result, "status": "imported"}))
        finally:
            service.close()
    else:
        build_parser().print_help()
