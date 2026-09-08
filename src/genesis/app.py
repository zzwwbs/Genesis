"""Local-first HTTP API backed by the durable GENESIS application service."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from genesis import __version__
from genesis.compiler import ValidationIssue
from genesis.replay import ReplayMode
from genesis.service import GenesisService

_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

_CORRELATION_ID: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def _correlation() -> str | None:
    return _CORRELATION_ID.get()


def _error(code: str, message: str, status: int, **details: Any) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": code,
                "message": message,
                "correlation_id": _correlation(),
                **details,
            }
        },
        status_code=status,
    )


_REMEDIATION: dict[str, str] = {
    "NOT_FOUND": "The resource does not exist; list the collection to find valid identifiers.",
    "ALREADY_EXISTS": "Use the existing resource or choose a different identifier.",
    "EXPECTED_VERSION": "Re-fetch the resource and retry with If-Match set to its current version.",
    "RUN_TRANSITION": "Only declared run transitions are allowed; inspect the run status first.",
    "SPECIFICATION_NOT_APPROVED": (
        "Inspect the draft, resolve validation issues, and approve the current version."
    ),
    "SPECIFICATION_INCOMPLETE": "Resolve the listed checklist items before approval.",
    "MODEL_PROFILE_DRIFT": "Re-resolve the runtime model profile against the compiled package.",
    "PROVIDER_CREDENTIAL": "Set the configured environment variable to the provider credential.",
    "INVALID_FIELD": "Remove or rename unsupported fields in the submitted document.",
}


_LOGGER = logging.getLogger("genesis.app")


def _service_error(exc: Exception) -> JSONResponse:
    _LOGGER.warning("service error: %s", exc)
    if isinstance(exc, KeyError):
        message = str(exc).strip(chr(34)).strip(chr(39))
        if message.startswith("ELICITATION_SESSION_EXPIRED"):
            return _error(
                "ELICITATION_SESSION_EXPIRED",
                str(exc).strip(chr(34)).strip(chr(39)),
                404,
                remediation=(
                    "Start a new session from the latest accepted package version; "
                    "unfinished sessions are in-memory only."
                ),
            )
        return _error(
            "NOT_FOUND",
            "resource not found",
            404,
            remediation=_REMEDIATION["NOT_FOUND"],
        )
    message = str(exc)
    code = message.split(":", 1)[0] if ":" in message else "VALIDATION_ERROR"
    status = (
        404
        if code == "PROFILE_NOT_FOUND"
        else 409
        if code
        in {
            "ALREADY_EXISTS",
            "EXPECTED_VERSION",
            "RUN_TRANSITION",
            "SPECIFICATION_NOT_APPROVED",
            "PATCH_BASE_STALE",
            "IDEMPOTENCY_CONFLICT",
        }
        else 422
    )
    return _error(
        code,
        message,
        status,
        remediation=_REMEDIATION.get(code, "Inspect the message and correct the request."),
    )


def _expected_version(value: str | None) -> int:
    if value is None:
        raise ValueError("VALIDATION_ERROR: If-Match version is required")
    try:
        version = int(value.strip('"'))
    except ValueError as exc:
        raise ValueError("VALIDATION_ERROR: If-Match must contain an integer version") from exc
    if version < 0:
        raise ValueError("VALIDATION_ERROR: If-Match version must be non-negative")
    return version


def create_app(
    workspace: str | Path | None = None, *, service: GenesisService | None = None
) -> FastAPI:
    if service is None:
        root = (
            Path(workspace) if workspace is not None else Path(tempfile.mkdtemp(prefix="genesis-"))
        )
        service = GenesisService(root)
    app = FastAPI(title="GENESIS", version=__version__)
    app.state.service = service
    app.state.idempotency = {}

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next: Any) -> Any:
        correlation_id = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        token = _CORRELATION_ID.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            _CORRELATION_ID.reset(token)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(
            "VALIDATION_ERROR",
            "request validation failed",
            422,
            details=jsonable_encoder(exc.errors()),
        )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "genesis"}

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return Path(__file__).with_name("static").joinpath("ui.html").read_text()

    @app.get("/studies")
    def list_studies() -> list[dict[str, Any]]:
        return service.list_studies()

    @app.post("/studies", status_code=201)
    def create_study(
        payload: dict[str, Any], idempotency_key: str | None = Header(None, alias="Idempotency-Key")
    ) -> Any:
        key = f"study:{idempotency_key}" if idempotency_key else None
        if key:
            try:
                cached = service.persistence.get_idempotency(key)
            except Exception:
                cached = None
            if cached is not None:
                return cached
        study_id = payload.get("id")
        if not isinstance(study_id, str) or not _ID.fullmatch(study_id):
            return _error("INVALID_ID", "id must be a stable lowercase identifier", 422)
        try:
            result = service.create_study(payload)
        except Exception as exc:
            return _service_error(exc)
        if key:
            service.persistence.record_idempotency(key, result)
        return result

    @app.get("/studies/{study_id}")
    def get_study(study_id: str) -> Any:
        try:
            return service.get_study(study_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/specifications")
    def list_specifications() -> list[dict[str, Any]]:
        return service.list_specifications()

    @app.post("/specifications", status_code=201)
    def create_specification(payload: dict[str, Any]) -> Any:
        try:
            return service.create_specification(payload)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/specifications/{specification_id}")
    def get_specification(specification_id: str) -> Any:
        try:
            return service.get_specification(specification_id)
        except Exception as exc:
            return _service_error(exc)

    @app.put("/specifications/{specification_id}")
    def update_specification(
        specification_id: str, payload: dict[str, Any], request: Request
    ) -> Any:
        try:
            expected = _expected_version(request.headers.get("If-Match"))
            return service.update_specification(specification_id, payload, expected)
        except Exception as exc:
            return _service_error(exc)

    @app.post("/specifications/{specification_id}/inspect")
    def inspect_specification(specification_id: str) -> Any:
        try:
            return service.inspect_specification(specification_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/specifications/{specification_id}/checklist")
    def get_checklist(specification_id: str) -> Any:
        try:
            return service.get_checklist(specification_id)
        except Exception as exc:
            return _service_error(exc)

    @app.put("/specifications/{specification_id}/checklist/{item_id}")
    def update_checklist_item(specification_id: str, item_id: str, payload: dict[str, Any]) -> Any:
        try:
            return service.update_checklist_item(
                specification_id,
                item_id,
                status=payload.get("status"),
                evidence=payload.get("evidence"),
                confirmed_by=payload.get("confirmed_by"),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/specifications/{specification_id}/propose-draft")
    def propose_draft(specification_id: str, payload: dict[str, Any] | None = None) -> Any:
        body = payload or {}
        try:
            return service.draft_from_model(
                specification_id,
                instruction=str(body.get("instruction", "")),
                profile_id=body.get("profile_id"),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/specifications/{specification_id}/drafts")
    def accept_draft(specification_id: str, payload: dict[str, Any]) -> Any:
        try:
            return service.accept_draft(
                specification_id,
                dict(payload.get("proposal", {})),
                confirmed_by=str(payload.get("confirmed_by", "researcher")),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/specifications/{specification_id}/approve")
    def approve_specification(
        specification_id: str, payload: dict[str, Any], request: Request
    ) -> Any:
        try:
            expected = _expected_version(request.headers.get("If-Match"))
            return service.approve_specification(
                specification_id, expected, str(payload.get("approved_by", "researcher"))
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/packages", status_code=201)
    def create_package(payload: dict[str, Any]) -> Any:
        package_id = (
            payload.get("id")
            or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        )
        try:
            return service.create_package({**payload, "id": package_id})
        except Exception as exc:
            return _service_error(exc)

    @app.get("/packages")
    def list_packages() -> list[dict[str, Any]]:
        return service.list_packages()

    @app.get("/builds")
    def list_builds() -> Any:
        try:
            return service.list_builds()
        except Exception as exc:
            return _service_error(exc)

    @app.get("/builds/{build_ref:path}/processes")
    def build_processes(build_ref: str) -> Any:
        try:
            build_path = service.resolve_path(build_ref)
            processes = json.loads((build_path / "processes.json").read_text())
            return {
                "build": build_ref,
                "processes": [
                    {
                        "id": process["id"],
                        "phase": process.get("trigger", {}).get("phase", 0),
                        "executor": process.get("executor", {}),
                        "context_policy": process.get("context_policy"),
                        "dependencies": {
                            "after": process.get("dependencies", {}).get("after", []),
                            "delay": process.get("dependencies", {}).get("delay", 0),
                        },
                    }
                    for process in processes
                ],
            }
        except Exception as exc:
            return _service_error(exc)

    @app.post("/exports")
    def export_bundle(payload: dict[str, Any]) -> Any:
        try:
            run_id = str(payload.get("run_id", payload.get("id", "")))
            output = str(payload.get("output", ""))
            mode = str(payload.get("mode", "exploration"))
            if not run_id or not output:
                return _error("VALIDATION_ERROR", "run_id and output are required", 422)
            paths = service.export_run(run_id, output, mode=mode)
            return {
                "run_id": run_id,
                "mode": mode,
                "paths": [str(path) for path in paths],
                "status": "exported",
            }
        except Exception as exc:
            return _service_error(exc)

    @app.get("/specifications/{specification_id}/versions/{version}/snapshot")
    def package_snapshot(specification_id: str, version: int) -> Any:
        try:
            return service.get_package_snapshot(specification_id, version)
        except KeyError:
            return _service_error(KeyError(specification_id))
        except Exception as exc:
            return _service_error(exc)

    @app.post("/imports", status_code=201)
    def import_package(payload: dict[str, Any]) -> Any:
        try:
            source = str(payload.get("source", ""))
            if not source:
                return _error("VALIDATION_ERROR", "source is required", 422)
            return service.import_package(source, specification_id=payload.get("specification_id"))
        except Exception as exc:
            return _service_error(exc)

    @app.get("/llm/profiles")
    def list_model_profiles() -> Any:
        try:
            return service.list_model_profiles()
        except Exception as exc:
            return _service_error(exc)

    @app.post("/llm/profiles", status_code=201)
    def create_model_profile(payload: dict[str, Any]) -> Any:
        try:
            return service.create_model_profile(payload)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/llm/profiles/{profile_id}")
    def get_model_profile(profile_id: str) -> Any:
        try:
            return service.get_model_profile(profile_id)
        except Exception as exc:
            return _service_error(exc)

    @app.put("/llm/profiles/{profile_id}")
    def update_model_profile(profile_id: str, payload: dict[str, Any], request: Request) -> Any:
        try:
            expected = _expected_version(request.headers.get("If-Match"))
            return service.update_model_profile(profile_id, payload, expected)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/llm/profiles/{profile_id}/status")
    def model_profile_status(profile_id: str) -> Any:
        try:
            return service.model_profile_status(profile_id)
        except Exception as exc:
            return _service_error(exc)

    @app.post("/llm/profiles/{profile_id}/test")
    def test_model_profile(profile_id: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            prompt = str((payload or {}).get("prompt", "Reply with OK."))
            return service.test_model_profile(profile_id, prompt)
        except Exception as exc:
            return _service_error(exc)

    @app.post("/compile")
    def compile_study(payload: dict[str, Any]) -> Any:
        source, output = payload.get("source"), payload.get("output")
        specification_id = payload.get("specification_id")
        if not output or (not source and not specification_id):
            return _error(
                "VALIDATION_ERROR", "source/specification_id and output are required", 422
            )
        try:
            return service.compile_study(source, output, specification_id=specification_id)
        except ValidationIssue as exc:
            return _error("COMPILE_INVALID", str(exc), 422, issues=[i.__dict__ for i in exc.issues])
        except Exception as exc:
            return _service_error(exc)

    @app.post("/runs", status_code=201)
    def create_run(payload: dict[str, Any]) -> Any:
        run_id = payload.get("id") or payload.get("run_id")
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
            return _error("INVALID_ID", "id must be a stable lowercase identifier", 422)
        try:
            return service.create_run(payload)
        except Exception as exc:
            return _service_error(exc)

    def transition(run_id: str, target: str, if_match: str | None) -> Any:
        try:
            run = service.get_run(run_id)
            if if_match is None:
                expected = run["version"]
            else:
                try:
                    expected = int(if_match.strip('"'))
                except ValueError:
                    return _error(
                        "VALIDATION_ERROR", "If-Match must contain an integer version", 422
                    )
            return service.transition_run(run_id, target, expected)
        except Exception as exc:
            return _service_error(exc)

    for action in ("pause", "resume", "cancel"):

        async def endpoint(run_id: str, request: Request, _action: str = action) -> Any:
            return transition(
                run_id,
                {"pause": "paused", "resume": "running", "cancel": "cancelled"}[_action],
                request.headers.get("If-Match"),
            )

        app.post(f"/runs/{{run_id}}/{action}")(endpoint)

    @app.get("/experiments")
    def list_experiments() -> Any:
        try:
            return service.persistence.list_experiments()
        except Exception as exc:
            return _service_error(exc)

    @app.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> Any:
        try:
            return service.persistence.get_experiment(experiment_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return service.list_runs()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> Any:
        try:
            return service.get_run(run_id)
        except Exception as exc:
            return _service_error(exc)

    @app.post("/runs/{run_id}/execute")
    def execute_run(run_id: str) -> Any:
        try:
            return service.execute_run(run_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/runs/{run_id}/events")
    def events(run_id: str) -> Any:
        try:
            return service.trace_run(run_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/runs/{run_id}/artifacts")
    def artifacts(run_id: str) -> Any:
        try:
            return service.artifacts_for_run(run_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/runs/{run_id}/natural-trace")
    def natural_trace(run_id: str, user: str | None = None, phase: int | None = None) -> Any:
        try:
            return service.natural_trace(
                run_id, user=user, phase=int(phase) if phase is not None else None
            )
        except Exception as exc:
            return _service_error(exc)

    @app.get("/runs/{run_id}/outcomes")
    def outcomes(run_id: str) -> Any:
        try:
            stored = service.get_run(run_id).get("outcomes", [])
            return stored or service.evaluate_outcomes(run_id)
        except Exception as exc:
            return _service_error(exc)

    for suffix in ("events", "artifacts", "outcomes"):

        @app.post(f"/runs/{{run_id}}/{suffix}", status_code=201)
        def append_collection(run_id: str, payload: dict[str, Any], _field: str = suffix) -> Any:
            try:
                return service.append_run_collection(run_id, _field, payload)
            except Exception as exc:
                return _service_error(exc)

    @app.post("/runs/import", status_code=201)
    def import_run(payload: dict[str, Any]) -> Any:
        try:
            source = str(payload.get("source", ""))
            if not source:
                return _error("VALIDATION_ERROR", "source is required", 422)
            result = service.import_run(
                source,
                run_id=payload.get("run_id"),
                size_limit_bytes=int(payload.get("size_limit_bytes") or 500 * 1024 * 1024),
            )
            return {**result, "status": "imported"}
        except Exception as exc:
            return _service_error(exc)

    @app.post("/runs/{run_id}/replays/preview")
    def replay_preview(run_id: str, payload: dict[str, Any] | None = None) -> Any:
        body = payload or {}
        try:
            return service.replay_preview(
                run_id,
                mode=ReplayMode(body.get("mode", "full")),
                artifact_ids=tuple(body.get("artifact_ids", ())),
                boundary=body.get("boundary"),
                overrides=body.get("overrides"),
                justification=body.get("justification"),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/runs/{run_id}/replays")
    def replay(run_id: str, payload: dict[str, Any] | None = None) -> Any:
        body = payload or {}
        try:
            return service.replay_run(
                run_id,
                mode=ReplayMode(body.get("mode", "full")),
                artifact_ids=tuple(body.get("artifact_ids", ())),
                boundary=body.get("boundary"),
                overrides=body.get("overrides"),
                justification=body.get("justification"),
                preview_token=body.get("preview_token"),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.get("/providers")
    def providers() -> list[dict[str, Any]]:
        return [
            {
                "id": "deterministic-mock",
                "capabilities": {"structured_output": True, "token_estimation": True},
            },
            {
                "id": "openai-compatible",
                "capabilities": {
                    "structured_output": True,
                    "token_estimation": True,
                    "cancellation": False,
                    "streaming": False,
                },
            },
        ]

    @app.post("/elicitation/sessions", status_code=201)
    def elicitation_sessions(payload: dict[str, Any]) -> Any:
        try:
            return service.start_elicitation(payload or {})
        except Exception as exc:
            return _service_error(exc)

    @app.get("/elicitation/sessions/{session_id}")
    def elicitation_session(session_id: str) -> Any:
        try:
            return service.get_elicitation(session_id)
        except Exception as exc:
            return _service_error(exc)

    @app.post("/elicitation/sessions/{session_id}/messages")
    def elicitation_messages(
        session_id: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        body = payload or {}
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="submit_message",
                payload=body,
                expected_version=body.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.submit_elicitation_message(
                    session_id,
                    str(body.get("answer", "")),
                    response_mode=str(body.get("response_mode", "free_form")),
                    suggestion_index=body.get("suggestion_index"),
                ),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.get("/elicitation/sessions")
    def elicitation_saved_sessions() -> Any:
        return {
            "items": [
                {
                    "session_id": s.session_id,
                    "specification_id": s.specification_id,
                    "current_stage": s.current_stage,
                    "status": s.status,
                    "updated_at": s.updated_at,
                }
                for s in service._elicitation_store.list_sessions()
            ]
        }

    @app.post("/elicitation/sessions/{session_id}/draft")
    def elicitation_draft(
        session_id: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        body = payload or {}
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="draft",
                payload=body,
                expected_version=body.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.draft_elicitation(session_id),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.get("/elicitation/sessions/{session_id}/preview")
    def elicitation_preview(session_id: str) -> Any:
        try:
            return service.preview_elicitation_stage(session_id)
        except Exception as exc:
            return _service_error(exc)

    @app.post("/elicitation/sessions/{session_id}/edit")
    def elicitation_edit(
        session_id: str,
        payload: dict[str, Any],
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="edit_draft",
                payload=payload,
                expected_version=payload.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.edit_elicitation_draft(
                    session_id, str(payload.get("filename", "")), str(payload.get("content", ""))
                ),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/elicitation/sessions/{session_id}/approve")
    def elicitation_approve(
        session_id: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        body = payload or {}
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="approve",
                payload=body,
                expected_version=body.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.approve_elicitation_stage(
                    session_id, approved_by=str(body.get("approved_by", "researcher"))
                ),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/elicitation/sessions/{session_id}/revise")
    def elicitation_revise(
        session_id: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        body = payload or {}
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="revise",
                payload=body,
                expected_version=body.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.revise_elicitation_stage(session_id),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/elicitation/sessions/{session_id}/reopen")
    def elicitation_reopen(
        session_id: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        body = payload or {}
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="reopen_stage",
                payload=body,
                expected_version=body.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.reopen_elicitation_stage(
                    session_id, str(body.get("stage_id", ""))
                ),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.post("/elicitation/sessions/{session_id}/cancel")
    def elicitation_cancel(
        session_id: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> Any:
        body = payload or {}
        try:
            return service.execute_elicitation_mutation(
                session_id,
                operation="cancel",
                payload=body,
                expected_version=body.get("expected_version"),
                idempotency_key=idempotency_key,
                mutation=lambda: service.cancel_elicitation(session_id),
            )
        except Exception as exc:
            return _service_error(exc)

    @app.get("/elicitation/workflows")
    def elicitation_workflows() -> Any:
        try:
            return service.list_elicitation_workflows()
        except Exception as exc:
            return _service_error(exc)

    @app.get("/elicitation/workflows/{workflow_id}")
    def elicitation_workflow(workflow_id: str) -> Any:
        try:
            return service.get_elicitation_workflow(workflow_id)
        except Exception as exc:
            return _service_error(exc)

    @app.get("/extensions")
    def extensions() -> Any:
        try:
            return service.list_extensions()
        except Exception as exc:
            return _service_error(exc)

    return app
