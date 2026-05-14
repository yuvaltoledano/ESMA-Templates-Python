"""FastAPI app for the ESMA-MILAN HTTP service.

Thin HTTP layer over the pipeline. Endpoints parse multipart input and
shape responses; all the real work - the size guardrail, materialising
uploads, running the pipeline, classifying failures - lives in
``handlers.py``. Pipeline logic does not live here.

Day 1 scope: a synchronous ``POST /api/process`` plus ``GET
/api/health``. Authentication, rate limiting, the asynchronous
job-queue endpoint for very large pools, and deployment hardening are
all explicitly deferred (see the PR description).
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import structlog
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from esma_milan.api.handlers import (
    ApiError,
    DryRunResult,
    process_pipeline_from_bytes,
)
from esma_milan.api.schemas import DryRunResponse, ErrorResponse, HealthResponse

# MIME type for .xlsx, matching the Content-Type R's output is served as.
_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


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

app = FastAPI(
    title="ESMA-MILAN pipeline API",
    version="0.1.0",
    description=(
        "HTTP wrapper around the ESMA -> MILAN structured-finance pipeline."
    ),
)


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
    """Reshape FastAPI's default 422 into a 400 `ErrorResponse`.

    A missing required file (`loans`, `collaterals`) or form field
    (`deal_name`), or an out-of-range `min_coverage`, lands here. The
    task spec calls for HTTP 400 with a structured body for the
    missing-file case, and routing every error through one shape keeps
    clients simple.
    """
    body = ErrorResponse(
        error="missing_field",
        message="The request is missing required fields or files, or a field is invalid.",
        details={"fields": _summarise_validation_error(exc)},
    )
    return JSONResponse(status_code=400, content=body.model_dump())


def _summarise_validation_error(exc: RequestValidationError) -> list[str]:
    """Client-safe per-field summary of which inputs failed validation."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = str(err.get("msg", "invalid"))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return parts if parts else ["request validation failed"]


@app.get("/api/health")
async def health() -> HealthResponse:
    """Trivial liveness check - confirms the FastAPI plumbing is up."""
    return HealthResponse(status="ok")


@app.post("/api/process")
async def process(
    loans: Annotated[UploadFile, File()],
    collaterals: Annotated[UploadFile, File()],
    deal_name: Annotated[str, Form()],
    taxonomy: Annotated[UploadFile | None, File()] = None,
    aggregation: Annotated[Literal["auto", "by_loan", "by_group"], Form()] = "auto",
    min_coverage: Annotated[float, Form(ge=0.0, le=1.0)] = 0.85,
    dry_run: Annotated[bool, Form()] = False,
) -> Response:
    """Run the pipeline against an uploaded ESMA loans/collaterals pair.

    Reads the uploads into memory, then hands off to the synchronous
    `process_pipeline_from_bytes`. Returns the composed workbook as an
    .xlsx download, or - when `dry_run` is true - a `DryRunResponse`
    JSON summary. All failure modes arrive as `ApiError` and are
    rendered by `_api_error_handler`.
    """
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
    )

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
