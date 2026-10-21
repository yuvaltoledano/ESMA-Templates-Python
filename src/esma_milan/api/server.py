"""FastAPI app for the ESMA-MILAN HTTP service.

Thin HTTP layer over the pipeline. Endpoints parse multipart input and
shape responses; all the real work - the size guardrail, materialising
uploads, running the pipeline, classifying failures - lives in
``handlers.py``. Pipeline logic does not live here.

The surface today: ``GET /api/health``, the synchronous ``POST
/api/process`` (pools up to ~30,000 loans), and the background-job
endpoints ``POST /api/jobs`` / ``GET /api/jobs/{id}`` /
``GET /api/jobs/{id}/result`` / ``DELETE /api/jobs/{id}`` for larger
pools (the job store is ``api/jobs.py``; the separate ``esma-milan-worker``
process in ``api/worker.py`` runs them). Per-IP rate limiting
(``slowapi``, in-memory backend), a consistent ``ErrorResponse`` shape on
every failure path, and transport-level input validation (per-file size
cap, Content-Type allowlist, ``deal_name`` character validation). Not
implemented: authentication.

Reverse-proxy deployment (docker-compose.yml + deploy/Caddyfile): with
``ESMA_MILAN_TRUST_PROXY=1`` the rate limiter keys on the first address
in ``X-Forwarded-For`` instead of the peer IP, which behind a proxy is
always the proxy's own. Trusting that header is safe only when nothing
but the proxy can reach this process: in the Compose deployment the app
container publishes no port, so every connection comes from Caddy, which
overwrites the header with the real client address. If the app port were
ever exposed directly, any client could forge the header and dodge the
limit, so the flag is off by default and must stay off in that case.
Host, port and worker count are read from ``ESMA_MILAN_HOST`` /
``ESMA_MILAN_PORT`` / ``ESMA_MILAN_WORKERS`` (see ``run``).

Note: this module deliberately omits ``from __future__ import
annotations``. slowapi's ``@limiter.limit`` decorator wraps the endpoint
with ``functools.wraps``; under stringized annotations FastAPI then
tries to resolve the endpoint's forward references against slowapi's
module globals (the wrapper's ``__globals__``) instead of this module's,
and fails on ``Annotated[..., Form()]``. Eager annotations - real
objects, no resolution step - sidestep that. Python is pinned to 3.12,
where ``X | None`` and ``list[str]`` evaluate fine at runtime anyway.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

import structlog
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from esma_milan.analysis import AnalysisResult
from esma_milan.api import jobs
from esma_milan.api.handlers import (
    CSV_CONTENT_TYPES,
    ApiError,
    DryRunResult,
    analysis_response_from_result,
    process_pipeline_from_bytes,
    validate_deal_name,
    validate_upload,
)
from esma_milan.api.schemas import (
    DryRunResponse,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
    JobCreated,
    JobResult,
    JobStage,
    JobStatus,
)
from esma_milan.config import DEFAULT_MIN_LOAN_ID_COVERAGE

# MIME type for .xlsx, matching the Content-Type R's output is served as.
_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Per-IP request rate limits.
#
# NOTE: slowapi's in-memory backend keeps one counter set per worker
# process, so with N uvicorn workers the effective per-IP ceiling is
# N x the configured value (uvicorn round-robins connections across
# workers). A shared backend (e.g. Redis) would make the limits
# cross-worker; that is a known limitation, not a planned change here.
#
# Module-level so a deployment can tune them without a code change: set
# ESMA_MILAN_RATE_LIMIT_PROCESS / ESMA_MILAN_RATE_LIMIT_HEALTH (slowapi
# syntax, e.g. "10/minute") to override the defaults. /api/health gets a
# higher ceiling because liveness probes are legitimately frequent.
RATE_LIMIT_PROCESS = os.environ.get("ESMA_MILAN_RATE_LIMIT_PROCESS", "10/minute")
RATE_LIMIT_HEALTH = os.environ.get("ESMA_MILAN_RATE_LIMIT_HEALTH", "60/minute")
# Job status / result / delete: the page polls status every 2 s (30/minute),
# so the budget must hold on a single worker's counter with room to spare.
RATE_LIMIT_JOBS = os.environ.get("ESMA_MILAN_RATE_LIMIT_JOBS", "120/minute")

# Uploads for background jobs are streamed to the job directory in chunks
# of this size; nothing larger than one chunk is held in memory per file.
_UPLOAD_CHUNK_BYTES = 1 << 20


def _configure_logging() -> None:
    """Configure structlog for the API process.

    Mirrors the CLI's console-renderer setup (cli.py::_configure_logging).
    With ``uvicorn --workers N`` this runs once per worker process - each
    worker imports this module independently - but workers share no
    memory, so there is no global state to coordinate.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            # Colour only when a person is watching: structlog defaults to
            # ANSI colours on Linux even for a pipe, which would litter
            # container logs with escape codes. PrintLogger writes to
            # stdout, so that is the stream to test.
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


_configure_logging()
log = structlog.get_logger(__name__)

TRUST_PROXY_ENV = "ESMA_MILAN_TRUST_PROXY"


def _rate_limit_key(request: Request) -> str:
    """Client address the rate limiter counts against.

    The peer IP (``get_remote_address``), unless ``ESMA_MILAN_TRUST_PROXY``
    is ``"1"`` and the request carries ``X-Forwarded-For``, in which case
    the first address in that header - the original client, as written
    by the proxy - is used. Read from the environment per call so the
    flag can be toggled without re-importing the module; one dict lookup
    per request. See the module docstring for when trusting the header
    is safe.
    """
    if os.environ.get(TRUST_PROXY_ENV) == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",", 1)[0].strip()
        if first:
            return first
    return get_remote_address(request)


# Rate limiter with an in-memory backend - one counter set per worker
# process (see the note above RATE_LIMIT_PROCESS). The key is the peer
# IP, or the forwarded client IP behind a trusted proxy (_rate_limit_key).
limiter = Limiter(key_func=_rate_limit_key)

app = FastAPI(
    title="ESMA-MILAN pipeline API",
    version="0.1.0",
    description=(
        "HTTP wrapper around the ESMA -> MILAN structured-finance pipeline."
    ),
)
# slowapi's @limiter.limit decorator resolves the limiter off app.state.
app.state.limiter = limiter


@app.exception_handler(ApiError)
async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Render a classified `ApiError` as a structured `ErrorResponse`.

    By the time an `ApiError` reaches here the handlers layer has
    already sanitized it - `error`/`message`/`details` are client-safe
    and no stack trace is attached. The HTTP status comes from
    `exc.status_code`; it is not echoed in the body.
    """
    body = ErrorResponse(
        error=exc.error, message=exc.message, details=exc.details
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reshape FastAPI's default 422 into a structured `ErrorResponse`.

    A missing required upload (`loans`, `collaterals`) or form field
    (`deal_name`) is a 400 `missing_field`; a field that is present but
    malformed - a non-numeric or out-of-range `min_coverage`, an
    unknown `aggregation` value - is a 422 `validation_error`. Both
    share one body shape, but the status still distinguishes "you
    forgot something" from "you sent something wrong".
    """
    errors = exc.errors()
    all_missing = bool(errors) and all(
        err.get("type") == "missing" for err in errors
    )
    status_code: int
    error_code: ErrorCode
    if all_missing:
        status_code, error_code = 400, "missing_field"
        message = "The request is missing required fields or files."
    else:
        status_code, error_code = 422, "validation_error"
        message = "One or more request fields failed validation."
    body = ErrorResponse(
        error=error_code,
        message=message,
        details={"fields": _summarise_validation_error(exc)},
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Render a tripped per-IP rate limit as a 429 `ErrorResponse`."""
    limit_str = str(exc.limit.limit) if exc.limit is not None else "unknown"
    log.warning(
        "api_rate_limit_exceeded",
        path=request.url.path,
        client=request.client.host if request.client else None,
        limit=limit_str,
    )
    body = ErrorResponse(
        error="rate_limit_exceeded",
        message="Rate limit exceeded. Slow down and retry shortly.",
        details={"limit": limit_str},
    )
    return JSONResponse(status_code=429, content=body.model_dump())


@app.exception_handler(Exception)
async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last-resort handler for anything not already classified.

    Pipeline failures are caught and turned into `ApiError`s inside the
    handlers layer; this catches the rest - an unexpected failure in the
    endpoint itself, a bug. The full exception and a stack trace go to
    the server-side structlog stream (`exc_info=True`) so the failure
    stays debuggable; the client gets only a generic message in the
    standard `ErrorResponse` shape - never a traceback, a file path, or
    a pipeline internal.
    """
    log.error(
        "api_unhandled_exception",
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        exc_info=True,
    )
    body = ErrorResponse(
        error="internal_error",
        message="An unexpected server error occurred.",
    )
    return JSONResponse(status_code=500, content=body.model_dump())


def _summarise_validation_error(exc: RequestValidationError) -> list[str]:
    """Client-safe per-field summary of which inputs failed validation."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = str(err.get("msg", "invalid"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return parts if parts else ["request validation failed"]


@app.get("/api/health")
@limiter.limit(RATE_LIMIT_HEALTH)
async def health(request: Request) -> HealthResponse:
    """Trivial liveness check - confirms the FastAPI plumbing is up.

    `request` is unused by the body but required: slowapi's rate-limit
    decorator resolves the caller's IP from it.
    """
    return HealthResponse(
        status="ok",
        max_upload_mb=jobs.max_upload_mb(),
        job_retention_hours=jobs.retention().total_seconds() / 3600,
    )


@app.post("/api/process")
@limiter.limit(RATE_LIMIT_PROCESS)
async def process(
    request: Request,
    loans: Annotated[UploadFile, File()],
    collaterals: Annotated[UploadFile, File()],
    deal_name: Annotated[str, Form()],
    aggregation: Annotated[Literal["auto", "by_loan", "by_group"], Form()] = "auto",
    min_coverage: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_MIN_LOAN_ID_COVERAGE,
    dry_run: Annotated[bool, Form()] = False,
    analysis_only: Annotated[bool, Form()] = False,
) -> Response:
    """Run the pipeline against an uploaded ESMA loans/collaterals pair.

    The two CSVs are the only files; the ESMA taxonomy is bundled with
    the package and cannot be supplied per request (a `taxonomy` form
    field from an older client is ignored, as FastAPI ignores any unknown
    multipart field). Validates the uploads at the transport layer
    (per-file size cap, Content-Type allowlist) *before* reading them
    into memory, then reads the bytes and hands off to the synchronous
    `process_pipeline_from_bytes` (which validates `deal_name` and runs
    the pipeline). Returns the composed workbook as an .xlsx download,
    or - when `dry_run` is true - a `DryRunResponse` JSON summary. All
    failure modes arrive as `ApiError` and are rendered by
    `_api_error_handler`. `request` is required by slowapi's rate-limit
    decorator.
    """
    validate_upload(
        loans, field="loans", allowed_content_types=CSV_CONTENT_TYPES
    )
    validate_upload(
        collaterals, field="collaterals", allowed_content_types=CSV_CONTENT_TYPES
    )

    loans_bytes = await loans.read()
    collaterals_bytes = await collaterals.read()

    result = process_pipeline_from_bytes(
        loans_bytes=loans_bytes,
        collaterals_bytes=collaterals_bytes,
        deal_name=deal_name,
        aggregation=aggregation,
        min_coverage=min_coverage,
        dry_run=dry_run,
        analysis_only=analysis_only,
    )

    if isinstance(result, AnalysisResult):
        return JSONResponse(content=analysis_response_from_result(result).model_dump())

    if isinstance(result, DryRunResult):
        body = DryRunResponse(
            deal_name=result.deal_name,
            chosen_aggregation=result.chosen_aggregation,
            loan_count=result.loan_count,
            property_count=result.property_count,
            group_count=result.group_count,
            warnings=result.warnings,
        )
        return JSONResponse(content=body.model_dump())

    # WorkbookResult: stream the .xlsx back as a file download. The
    # filename matches R's pattern ("<cutoff> <deal> Flattened loans
    # and collaterals.xlsx"), built by run_pipeline.
    return Response(
        content=result.workbook_bytes,
        media_type=_XLSX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{result.filename}"',
        },
    )


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


def _load_job(job_id: str) -> jobs.Job:
    """Resolve a path id to a job directory, or raise 404 `job_not_found`.

    The id is validated against ``jobs.JOB_ID_RE`` before any filesystem
    access, so a traversal-shaped id never reaches the disk.
    """
    try:
        return jobs.get_job(jobs.jobs_root(), job_id)
    except jobs.JobNotFoundError:
        raise ApiError(
            404,
            "job_not_found",
            "No such job. It may never have existed, or it has expired or been deleted.",
            details={"job_id": job_id[:64]},
        ) from None


def _job_status_body(job: jobs.Job) -> JobStatus:
    status = job.read_status()
    error = None
    if status.error:
        error = ErrorResponse(
            error=status.error.get("error", "internal_error"),
            message=str(status.error.get("message", "")),
            details=status.error.get("details"),
        )
    return JobStatus(
        job_id=job.job_id,
        state=status.state,
        mode=status.mode,
        deal_name=status.deal_name,
        created_at=status.created_at,
        started_at=status.started_at,
        finished_at=status.finished_at,
        stage=JobStage(**status.stage.__dict__) if status.stage else None,
        error=error,
        result=JobResult(**status.result) if status.result else None,
        retention_hours=jobs.retention().total_seconds() / 3600,
    )


async def _stream_upload_to(upload: UploadFile, destination: Path) -> None:
    """Copy an upload to disk in fixed-size chunks; never the whole file in memory."""
    with destination.open("wb") as fh:
        while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
            fh.write(chunk)


@app.post("/api/jobs", status_code=202)
@limiter.limit(RATE_LIMIT_PROCESS)
async def create_job(
    request: Request,
    loans: Annotated[UploadFile, File()],
    collaterals: Annotated[UploadFile, File()],
    deal_name: Annotated[str, Form()],
    aggregation: Annotated[Literal["auto", "by_loan", "by_group"], Form()] = "auto",
    min_coverage: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_MIN_LOAN_ID_COVERAGE,
    analysis_only: Annotated[bool, Form()] = False,
) -> JSONResponse:
    """Queue a background pipeline run and return 202 with its id.

    Same multipart form as `POST /api/process` minus `dry_run`
    (`analysis_only` selects the mode), the same validation (deal name,
    Content-Types, per-file cap) but with the cap taken from
    `ESMA_MILAN_MAX_UPLOAD_MB` and no loan-count limit. The two CSV
    uploads are streamed into the job directory; the worker process picks
    the job up from there and reads the bundled ESMA taxonomy itself.
    """
    cap = jobs.max_upload_bytes()
    validate_upload(loans, field="loans", allowed_content_types=CSV_CONTENT_TYPES, max_bytes=cap)
    validate_upload(
        collaterals, field="collaterals", allowed_content_types=CSV_CONTENT_TYPES, max_bytes=cap
    )
    validate_deal_name(deal_name)

    root = jobs.jobs_root()
    job_request = jobs.JobRequest(
        deal_name=deal_name,
        mode="analysis" if analysis_only else "workbook",
        aggregation=aggregation,
        min_coverage=min_coverage,
        created_at=jobs.now_iso(),
        filenames={
            "loans": loans.filename or "",
            "collaterals": collaterals.filename or "",
        },
    )
    job = jobs.begin_job(root, job_request)
    try:
        await _stream_upload_to(loans, job.inputs_dir / "loans.csv")
        await _stream_upload_to(collaterals, job.inputs_dir / "collaterals.csv")
        job = jobs.commit_job(root, job)
    except BaseException:
        jobs.abort_job(job)
        raise

    log.info(
        "api_job_created",
        job_id=job.job_id,
        deal_name=deal_name,
        mode=job_request.mode,
        loans_bytes=(job.inputs_dir / "loans.csv").stat().st_size,
        collaterals_bytes=(job.inputs_dir / "collaterals.csv").stat().st_size,
    )
    body = JobCreated(
        job_id=job.job_id,
        status_url=f"/api/jobs/{job.job_id}",
        result_url=f"/api/jobs/{job.job_id}/result",
    )
    return JSONResponse(status_code=202, content=body.model_dump())


@app.get("/api/jobs/{job_id}")
@limiter.limit(RATE_LIMIT_JOBS)
async def get_job_status(request: Request, job_id: str) -> JobStatus:
    """The job's current `status.json`; 404 `job_not_found` otherwise."""
    return _job_status_body(_load_job(job_id))


@app.get("/api/jobs/{job_id}/result")
@limiter.limit(RATE_LIMIT_JOBS)
async def get_job_result(request: Request, job_id: str) -> Response:
    """The finished job's output: the workbook as a download (same
    `Content-Disposition` convention as `/api/process`) or the analysis
    JSON. 409 `job_not_ready` while queued/running; 409 carrying the
    job's own error when failed; 404 when unknown."""
    job = _load_job(job_id)
    status = job.read_status()
    if status.state in ("queued", "running"):
        raise ApiError(
            409,
            "job_not_ready",
            "The job has not finished yet.",
            details={"state": status.state, "stage": status.stage.__dict__ if status.stage else None},
        )
    if status.state == "failed" or not status.result:
        err = status.error or {}
        raise ApiError(
            409,
            err.get("error", "internal_error"),
            str(err.get("message", "The job failed.")),
            details=err.get("details"),
        )
    filename = str(status.result["filename"])
    path = job.output_dir / filename
    if not path.is_file():
        raise ApiError(404, "job_not_found", "The job's output is no longer available.")
    if status.mode == "analysis":
        return FileResponse(path, media_type="application/json")
    return FileResponse(
        path,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/jobs/{job_id}", status_code=204)
@limiter.limit(RATE_LIMIT_JOBS)
async def delete_job(request: Request, job_id: str) -> Response:
    """Remove a finished job's directory (the "delete from server after
    download" action). 409 `job_not_ready` while queued/running."""
    job = _load_job(job_id)
    status = job.read_status()
    if status.state in ("queued", "running"):
        raise ApiError(
            409,
            "job_not_ready",
            "The job is still queued or running and cannot be deleted yet.",
            details={"state": status.state},
        )
    jobs.delete_job(job)
    log.info("api_job_deleted", job_id=job.job_id, state=status.state)
    return Response(status_code=204)


def run() -> None:
    """Entry point for ``uv run esma-milan-server``.

    Starts uvicorn with ``ESMA_MILAN_WORKERS`` worker processes (default
    4) on ``ESMA_MILAN_HOST:ESMA_MILAN_PORT`` (default ``0.0.0.0:8000``).
    Each worker is single-threaded and handles one request at a time;
    four workers serve four concurrent users. Workers are independent OS
    processes with no shared memory, which is fine because every request
    is fully independent (the per-worker rate-limit counters are the one
    consequence; see RATE_LIMIT_PROCESS). There are no CLI flags; the
    three environment variables are the only configuration, so the
    defaults keep laptop behaviour unchanged and docker-compose.yml
    sets them for the container.
    """
    uvicorn.run(
        "esma_milan.api.server:app",
        host=os.environ.get("ESMA_MILAN_HOST", "0.0.0.0"),
        port=int(os.environ.get("ESMA_MILAN_PORT", "8000")),
        workers=int(os.environ.get("ESMA_MILAN_WORKERS", "4")),
    )
