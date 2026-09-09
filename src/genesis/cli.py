"""Command-line entry point for GENESIS."""

import argparse
import json
import os
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
        if name == "export":
            command.add_argument(
                "--mode",
                default="exploration",
                choices=["exploration", "reproducibility"],
                help="Bundle capability level",
            )
    return parser


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
        from genesis.compiler import StudyCompiler

        # Validation is compilation without retaining a build artifact.
        target = Path(args.output or (Path(args.path) / ".genesis-build-check"))
        try:
            build = StudyCompiler(args.path).compile(target)
            print(
                json.dumps(
                    {"valid": True, "study_id": build.study_id, "build_hash": build.build_hash}
                )
            )
        finally:
            if target.exists():
                import shutil

                shutil.rmtree(target)
    elif args.command == "compile":
        if not args.output:
            raise SystemExit("compile requires --output")
        source, output = Path(args.path).resolve(), Path(args.output).resolve()
        workspace = Path(os.path.commonpath([source, output]))
        if workspace == Path(workspace.anchor):
            # A source under /Users and output under /tmp share only `/`; using
            # it as a service workspace would attempt to create `/.genesis`.
            # Explicit CLI package compilation is safe to perform directly.
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
                try:
                    service.get_run(run_id)
                except KeyError:
                    payload = {"id": run_id}
                    if args.output:
                        payload["build"] = args.output
                    service.create_run(payload)
                result = service.execute_run(run_id)
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
            paths = service.export_run(
                args.run_id or "run-1", args.output, mode=str(getattr(args, "mode", "exploration"))
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
