"""FastAPI app for the ESMA-MILAN HTTP service.

Thin HTTP layer over the pipeline. Endpoints parse multipart input and
shape responses; all the real work - the size guardrail, materialising
uploads, running the pipeline, classifying failures - lives in
``handlers.py``. Pipeline logic does not live here.

Day 2 hardens the surface: per-IP rate limiting (``slowapi``, in-memory
backend), a consistent ``ErrorResponse`` shape on every failure path,
and transport-level input validation (per-file size cap, Content-Type
allowlist, ``deal_name`` character validation). Still deferred:
authentication, the asynchronous job-queue endpoint for very large
pools, and reverse-proxy ``X-Forwarded-For`` handling - rate limiting
keys on the peer IP for now. See the PR description.

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
from typing import Annotated, Literal

import structlog
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from esma_milan.analysis import AnalysisResult
from esma_milan.api.handlers import (
    CSV_CONTENT_TYPES,
    XLSX_CONTENT_TYPES,
    ApiError,
    DryRunResult,
    process_pipeline_from_bytes,
    validate_upload,
)
from esma_milan.api.schemas import (
    AnalysisResponse,
    DryRunResponse,
    ErrorCode,
    ErrorResponse,
    ExecutionSummaryRow as ExecutionSummaryRowSchema,
    HealthResponse,
)
from esma_milan.api.schemas import (
    AnalysisSummary as AnalysisSummarySchema,
)
from esma_milan.api.schemas import (
    Stratification as StratificationSchema,
)
from esma_milan.api.schemas import (
    StratificationRow as StratificationRowSchema,
)
from esma_milan.api.schemas import (
    StratificationTotal as StratificationTotalSchema,
)

# MIME type for .xlsx, matching the Content-Type R's output is served as.
_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Per-IP request rate limits.
#
# NOTE: slowapi's in-memory backend keeps one counter set per worker
# process, so with N uvicorn workers the effective per-IP ceiling is
# N x the configured value (uvicorn round-robins connections across
# workers). Day 3+ migration to a shared (Redis) backend makes the
# limits truly cross-worker; until then this gap is tracked in PR #17's
# known limitations.
#
# Module-level so a deployment can tune them without a code change: set
# ESMA_MILAN_RATE_LIMIT_PROCESS / ESMA_MILAN_RATE_LIMIT_HEALTH (slowapi
# syntax, e.g. "10/minute") to override the defaults. /api/health gets a
# higher ceiling because liveness probes are legitimately frequent.
RATE_LIMIT_PROCESS = os.environ.get("ESMA_MILAN_RATE_LIMIT_PROCESS", "10/minute")
RATE_LIMIT_HEALTH = os.environ.get("ESMA_MILAN_RATE_LIMIT_HEALTH", "60/minute")


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
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


_configure_logging()
log = structlog.get_logger(__name__)

# Rate limiter with an in-memory backend - one counter set per worker
# process, which is fine for the single-VPS Day 2 deployment. A shared
# (Redis) backend is Day 3+, alongside the async job queue that needs
# cross-worker state anyway. The key is the peer IP; behind a reverse
# proxy that is the proxy's IP, so X-Forwarded-For handling is Day 3+
# when the service actually deploys behind Caddy/nginx.
limiter = Limiter(key_func=get_remote_address)

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
    return HealthResponse(status="ok")


@app.post("/api/process")
@limiter.limit(RATE_LIMIT_PROCESS)
async def process(
    request: Request,
    loans: Annotated[UploadFile, File()],
    collaterals: Annotated[UploadFile, File()],
    deal_name: Annotated[str, Form()],
    taxonomy: Annotated[UploadFile | None, File()] = None,
    aggregation: Annotated[Literal["auto", "by_loan", "by_group"], Form()] = "auto",
    min_coverage: Annotated[float, Form(ge=0.0, le=1.0)] = 0.85,
    dry_run: Annotated[bool, Form()] = False,
    analysis_only: Annotated[bool, Form()] = False,
) -> Response:
    """Run the pipeline against an uploaded ESMA loans/collaterals pair.

    Validates the uploads at the transport layer (per-file size cap,
    Content-Type allowlist) *before* reading them into memory, then
    reads the bytes and hands off to the synchronous
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
    if taxonomy is not None:
        validate_upload(
            taxonomy, field="taxonomy", allowed_content_types=XLSX_CONTENT_TYPES
        )

    loans_bytes = await loans.read()
    collaterals_bytes = await collaterals.read()
    taxonomy_bytes = await taxonomy.read() if taxonomy is not None else None

    result = process_pipeline_from_bytes(
        loans_bytes=loans_bytes,
        collaterals_bytes=collaterals_bytes,
        taxonomy_bytes=taxonomy_bytes,
        deal_name=deal_name,
        aggregation=aggregation,
        min_coverage=min_coverage,
        dry_run=dry_run,
        analysis_only=analysis_only,
    )

    if isinstance(result, AnalysisResult):
        analysis_body = AnalysisResponse(
            deal_name=result.deal_name,
            summary=AnalysisSummarySchema(
                loan_count=result.summary.loan_count,
                property_count=result.summary.property_count,
                group_count=result.summary.group_count,
                total_current_balance=result.summary.total_current_balance,
                chosen_aggregation=result.summary.chosen_aggregation,
                warnings=result.summary.warnings,
            ),
            stratifications={
                key: StratificationSchema(
                    title=strat.title,
                    type=strat.type,
                    chart_type=strat.chart_type,
                    rows=[
                        StratificationRowSchema(
                            label=row.label,
                            count=row.count,
                            count_pct=row.count_pct,
                            balance=row.balance,
                            balance_pct=row.balance_pct,
                        )
                        for row in strat.rows
                    ],
                    total=StratificationTotalSchema(
                        count=strat.total.count,
                        balance=strat.total.balance,
                    ),
                    error=strat.error,
                    note=strat.note,
                )
                for key, strat in result.stratifications.items()
            },
            execution_summary=[
                ExecutionSummaryRowSchema(label=row.label, value=row.value)
                for row in result.execution_summary
            ],
        )
        return JSONResponse(content=analysis_body.model_dump())

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


def run() -> None:
    """Entry point for ``uv run esma-milan-server``.

    Starts uvicorn with 4 worker processes. Each worker is single-
    threaded and handles one request at a time; four workers serve four
    concurrent users. Workers are independent OS processes with no
    shared memory - fine for Day 1 since every request is fully
    independent, and it forces any cross-request state added later
    (e.g. a Redis-backed job registry for the async endpoint) to live
    outside the workers by construction. Host/port/worker count are
    hard-coded Day 1 defaults; CLI-flag configurability is Day 2 polish.
    """
    uvicorn.run(
        "esma_milan.api.server:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
    )
