"""Request-handling logic for the ESMA-MILAN HTTP API.

This module is the seam between the thin FastAPI endpoints in
``server.py`` and the pipeline in ``runner.py``. Endpoints parse
multipart input and shape responses; everything in between - the
size guardrail, materialising uploads, invoking the pipeline,
classifying failures - lives here.

`process_pipeline_from_bytes` is deliberately synchronous and
side-effect-light (bytes in, a `ProcessResult` out, plus logging). That
shape is what makes the Day 3+ asynchronous job endpoint a clean
addition rather than a refactor: the same function gets called from a
background task instead of straight from an endpoint handler.

The file was originally stubbed as ``jobs.py`` in anticipation of that
async job queue. Day 1 builds only the synchronous endpoint, so it is
named ``handlers.py`` for what it actually is today; ``jobs.py`` returns
when the async queue does.

Disk note: the pipeline's input interface (`runner.run_pipeline` ->
`stage1` -> `read_csv` / `read_taxonomy`) is path-based. Refactoring
that whole chain to accept in-memory bytes would touch the most
parity-sensitive code in the repo, so Day 1 instead materialises
uploads into a per-request `tempfile.TemporaryDirectory()` that is torn
down before this function returns. The workbook *output* is true
in-memory (`runner.run_pipeline(output_dir=None)` -> bytes). A future
`Path | bytes` refactor of the read chain would close the input gap.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import structlog
from fastapi import UploadFile

from esma_milan.api.schemas import ErrorCode
from esma_milan.runner import run_pipeline

log = structlog.get_logger(__name__)

# Hard cap on pool size for the synchronous /api/process endpoint. Pools
# larger than this risk exceeding reverse-proxy HTTP timeouts on a
# synchronous request, so they are rejected with HTTP 413 and pointed at
# the (Day 3+) asynchronous job endpoint.
#
# The count is approximate: it is the data-row limit derived via newline
# counting on the uploaded loans CSV, not a full parse. With a trailing
# newline this equals the data-row count exactly; without one it
# under-counts by one; quoted newlines inside fields would over-count.
# The effective limit is therefore ~29,999 data rows after the header.
# Approximate is fine here - this is a coarse guardrail, and the precise
# pool size is known once the pipeline parses the file.
MAX_SYNC_LOAN_COUNT: int = 30_000

# Default ESMA taxonomy used when a request omits the optional taxonomy
# upload. Mirrors the CLI default in cli.py: a path relative to the
# process working directory, bundled with the R reference submodule.
DEFAULT_TAXONOMY_PATH: Path = Path("r_reference/inputs/ESMA template taxonomy.xlsx")

_AGGREGATION_CHOICES: frozenset[str] = frozenset({"auto", "by_loan", "by_group"})

# Transport-level hard cap on a single uploaded file, enforced before the
# bytes are read into memory. This is distinct from MAX_SYNC_LOAN_COUNT:
# that is a content-level guardrail (loan rows, checked after the file is
# in hand); this is a transport-level one (raw bytes) that stops a
# hostile 1 GB upload from forcing a large allocation in the first place.
MAX_UPLOAD_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MiB

# Content-Type allowlists for the uploaded files. Strict by design: a
# public API accepting only explicit content types has less attack
# surface than one that sniffs `application/octet-stream`. Clients that
# mislabel their CSVs get a 400 that names the accepted types.
CSV_CONTENT_TYPES: frozenset[str] = frozenset({"text/csv", "application/csv"})
XLSX_CONTENT_TYPES: frozenset[str] = frozenset(
    {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
)

# deal_name flows into the Content-Disposition header and the output
# workbook filename. The allowlist - alphanumerics, space, hyphen,
# underscore, period - is what makes that safe: it admits no quotes,
# newlines, null bytes, path separators or `..`, so header-injection and
# path-traversal are excluded as a class rather than blocklisted
# character by character.
MAX_DEAL_NAME_LENGTH: int = 100
_DEAL_NAME_PATTERN = re.compile(r"[A-Za-z0-9 ._-]+")


class ApiError(Exception):
    """A failure that has already been classified into an HTTP response.

    `status_code` is the HTTP status to return (it is not echoed in the
    response body); `error` is the stable machine-readable `ErrorCode`;
    `message` is a sanitized, client-safe summary; `details` is optional
    structured context that is also client-safe. Full diagnostic detail
    - stack traces, pipeline internals - goes to the server-side
    structlog stream, never into an `ApiError`.
    """

    def __init__(
        self,
        status_code: int,
        error: ErrorCode,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details


@dataclass(frozen=True)
class WorkbookResult:
    """Successful non-dry-run result: the composed 10-sheet workbook,
    in memory, ready to stream back as a file download."""

    workbook_bytes: bytes
    filename: str


@dataclass(frozen=True)
class DryRunResult:
    """Successful dry-run result: a structured summary of the pipeline
    run with no workbook composed."""

    deal_name: str
    chosen_aggregation: str
    loan_count: int
    property_count: int
    group_count: int
    warnings: list[str] = field(default_factory=list)


# Either branch of POST /api/process: a workbook download, or - when
# dry_run was requested - a structured JSON summary.
ProcessResult = WorkbookResult | DryRunResult


def _estimate_loan_row_count(loans_bytes: bytes) -> int:
    """Approximate the loans CSV's data-row count by counting newlines
    (minus the header line). See `MAX_SYNC_LOAN_COUNT` for why an
    approximation is acceptable for the size guardrail."""
    return max(loans_bytes.count(b"\n") - 1, 0)


def _first_line(text: str, *, limit: int = 300) -> str:
    """First non-empty line of `text`, trimmed to `limit` chars.

    Polars and pipeline errors are often multi-line with a diagnostic
    tail; the first line is the client-relevant summary. Truncating
    keeps error responses tidy and avoids echoing large input fragments
    back to the caller.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= limit else stripped[:limit] + "..."
    return "(no error detail)"


def validate_upload(
    file: UploadFile,
    *,
    field: str,
    allowed_content_types: frozenset[str],
) -> None:
    """Transport-level checks on an uploaded file, run before `.read()`.

    Rejects a file larger than `MAX_UPLOAD_FILE_SIZE` (413
    `file_too_large`) or one whose Content-Type is not in
    `allowed_content_types` (400 `invalid_content_type`). Size is
    checked first and before the bytes are pulled into memory, so an
    oversized upload is turned away rather than allocated.

    Args:
        file: the multipart `UploadFile` from the endpoint signature.
        field: the form field name ("loans", "collaterals", "taxonomy"),
            used in the error so the client knows which file to fix.
        allowed_content_types: the lower-cased Content-Types accepted for
            this field (`CSV_CONTENT_TYPES` or `XLSX_CONTENT_TYPES`).

    Raises:
        ApiError: 413 for an oversized file, 400 for a disallowed
            Content-Type. Both carry a client-safe `details` dict.
    """
    size = file.size
    if size is not None and size > MAX_UPLOAD_FILE_SIZE:
        raise ApiError(
            413,
            "file_too_large",
            f"Uploaded file '{field}' exceeds the per-file size limit of "
            f"{MAX_UPLOAD_FILE_SIZE // (1024 * 1024)} MB.",
            details={
                "field": field,
                "size_bytes": size,
                "limit_bytes": MAX_UPLOAD_FILE_SIZE,
            },
        )

    # Compare on the bare type, tolerating a charset/parameter suffix
    # (e.g. "text/csv; charset=utf-8") but staying strict on the type.
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in allowed_content_types:
        allowed = ", ".join(sorted(allowed_content_types))
        raise ApiError(
            400,
            "invalid_content_type",
            f"Uploaded file '{field}' has an unsupported Content-Type. "
            f"Send one of: {allowed}.",
            details={
                "field": field,
                "received": file.content_type,
                "allowed": sorted(allowed_content_types),
            },
        )


def validate_deal_name(deal_name: str) -> None:
    """Validate `deal_name` before it reaches the filename / header.

    Allowed: letters, digits, space, hyphen, underscore, period; length
    1-100 characters. `re.fullmatch` (not `re.match`) is deliberate -
    `match` with a `$` anchor would still admit a trailing newline,
    which is exactly the header-injection character this guards against.

    Raises:
        ApiError: 400 `invalid_deal_name` if the name is empty, too
            long, or contains a disallowed character.
    """
    if deal_name.strip() == "":
        raise ApiError(
            400, "invalid_deal_name", "deal_name must not be empty."
        )
    if len(deal_name) > MAX_DEAL_NAME_LENGTH:
        raise ApiError(
            400,
            "invalid_deal_name",
            f"deal_name must be at most {MAX_DEAL_NAME_LENGTH} characters.",
            details={"length": len(deal_name), "limit": MAX_DEAL_NAME_LENGTH},
        )
    if _DEAL_NAME_PATTERN.fullmatch(deal_name) is None:
        raise ApiError(
            400,
            "invalid_deal_name",
            "deal_name may only contain letters, digits, spaces, hyphens, "
            "underscores and periods.",
            details={"field": "deal_name"},
        )


def process_pipeline_from_bytes(
    *,
    loans_bytes: bytes,
    collaterals_bytes: bytes,
    taxonomy_bytes: bytes | None,
    deal_name: str,
    aggregation: str,
    min_coverage: float,
    dry_run: bool,
) -> ProcessResult:
    """Run the ESMA -> MILAN pipeline against in-memory uploads.

    Args:
        loans_bytes: raw bytes of the uploaded loans CSV.
        collaterals_bytes: raw bytes of the uploaded collaterals CSV.
        taxonomy_bytes: raw bytes of an uploaded taxonomy XLSX, or None
            to fall back to `DEFAULT_TAXONOMY_PATH`.
        deal_name: deal name used in the output filename and summary.
        aggregation: one of "auto" | "by_loan" | "by_group". "auto"
            lets the pipeline detect the method (Stage 6).
        min_coverage: minimum acceptable calc_loan_id coverage, in
            [0.0, 1.0].
        dry_run: when True, skip workbook composition and return a
            `DryRunResult` summary instead of a `WorkbookResult`.

    Returns:
        A `WorkbookResult` (dry_run=False) or `DryRunResult`
        (dry_run=True).

    Raises:
        ApiError: every failure mode, already classified into an HTTP
            status. 413 for an oversized pool; 400 for input-data
            errors (unparseable CSV, missing/duplicate columns, missing
            required fields); 500 for anything unexpected. Stack traces
            are logged server-side, never surfaced in the `ApiError`.
    """
    validate_deal_name(deal_name)
    if aggregation not in _AGGREGATION_CHOICES:
        raise ApiError(
            400,
            "validation_error",
            "aggregation must be one of: auto, by_loan, by_group.",
            details={"field": "aggregation", "received": aggregation},
        )

    # --- Size guardrail (before any parsing) ------------------------------
    estimated_rows = _estimate_loan_row_count(loans_bytes)
    if estimated_rows > MAX_SYNC_LOAN_COUNT:
        log.warning(
            "api_pool_too_large",
            deal_name=deal_name,
            estimated_rows=estimated_rows,
            limit=MAX_SYNC_LOAN_COUNT,
        )
        raise ApiError(
            413,
            "size_limit_exceeded",
            "Pool size exceeds the synchronous endpoint limit "
            "(approximately 30,000 loans; the check counts CSV lines, so "
            "the effective cap is ~29,999 data rows). For larger pools, "
            "use the asynchronous job endpoint at `POST /api/jobs` "
            "(not yet implemented).",
            details={
                "estimated_loan_rows": estimated_rows,
                "limit": MAX_SYNC_LOAN_COUNT,
            },
        )

    # "auto" -> let Stage 6 detect; otherwise pass the explicit method.
    aggregation_method = None if aggregation == "auto" else aggregation

    log.info(
        "api_process_start",
        deal_name=deal_name,
        aggregation=aggregation,
        min_coverage=min_coverage,
        dry_run=dry_run,
        estimated_rows=estimated_rows,
        taxonomy="uploaded" if taxonomy_bytes is not None else "default",
    )

    # --- Materialise uploads + run the pipeline ---------------------------
    # The temp dir holds only the *input* CSVs/XLSX; it is torn down on
    # exit from the `with` block. The workbook output never touches disk
    # (run_pipeline(output_dir=None) -> bytes), and the dry-run summary
    # is extracted into plain ints before the block exits, so nothing
    # the caller receives depends on the temp dir's lifetime.
    with tempfile.TemporaryDirectory(prefix="esma_api_") as tmp_str:
        tmp = Path(tmp_str)
        loans_path = tmp / "loans.csv"
        collaterals_path = tmp / "collaterals.csv"
        loans_path.write_bytes(loans_bytes)
        collaterals_path.write_bytes(collaterals_bytes)

        if taxonomy_bytes is not None:
            taxonomy_path = tmp / "taxonomy.xlsx"
            taxonomy_path.write_bytes(taxonomy_bytes)
        else:
            taxonomy_path = DEFAULT_TAXONOMY_PATH
            if not taxonomy_path.exists():
                log.error(
                    "api_default_taxonomy_missing",
                    path=str(taxonomy_path),
                )
                raise ApiError(
                    500,
                    "internal_error",
                    "The server's default ESMA taxonomy is unavailable. "
                    "Include a `taxonomy` file in the request to proceed.",
                )

        try:
            result = run_pipeline(
                loans_file_path=loans_path,
                collaterals_file_path=collaterals_path,
                taxonomy_file_path=taxonomy_path,
                deal_name=deal_name,
                output_dir=None,  # in-memory: workbook returned as bytes
                aggregation_method=aggregation_method,
                min_coverage=min_coverage,
                dry_run=dry_run,
                verbose=True,
            )
        except ApiError:
            raise
        except (ValueError, pl.exceptions.PolarsError) as exc:
            # Input-data errors: unparseable CSV, missing/duplicate
            # columns after the taxonomy rename, missing required
            # fields, missing pool_cutoff_date. The messages from the
            # I/O layer and validators describe the *data* problem and
            # are safe to surface (sanitized + first-line-only).
            log.warning(
                "api_process_input_error",
                deal_name=deal_name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise ApiError(
                400,
                "invalid_csv",
                "The uploaded data could not be processed.",
                details={"reason": _first_line(str(exc))},
            ) from exc
        except Exception as exc:
            # Anything else is unexpected. Log full detail server-side;
            # return a generic message with no internals leaked.
            log.error(
                "api_process_unexpected_error",
                deal_name=deal_name,
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )
            raise ApiError(
                500,
                "internal_error",
                "Pipeline error - check input data.",
            ) from exc

        if dry_run:
            dry = _build_dry_run_result(result, deal_name, aggregation)
            log.info(
                "api_process_complete",
                deal_name=deal_name,
                dry_run=True,
                loan_count=dry.loan_count,
            )
            return dry

        assert result.workbook_bytes is not None, (
            "run_pipeline(output_dir=None, dry_run=False) must return "
            "workbook_bytes"
        )
        assert result.output_filename is not None, (
            "run_pipeline(output_dir=None, dry_run=False) must return "
            "output_filename"
        )
        log.info(
            "api_process_complete",
            deal_name=deal_name,
            dry_run=False,
            filename=result.output_filename,
            n_bytes=len(result.workbook_bytes),
        )
        return WorkbookResult(
            workbook_bytes=result.workbook_bytes,
            filename=result.output_filename,
        )


def _build_dry_run_result(
    result: object,
    deal_name: str,
    aggregation: str,
) -> DryRunResult:
    """Pull the dry-run summary out of a `PipelineResult`.

    Kept separate from `process_pipeline_from_bytes` so the stage-output
    plumbing stays out of the main control flow. The asserts pin the
    invariant that `run_pipeline(dry_run=True)` populates every stage -
    if that ever changes the failure is loud and local rather than an
    `AttributeError` deep in a response handler.
    """
    from esma_milan.runner import PipelineResult

    assert isinstance(result, PipelineResult)
    assert result.stage3 is not None
    assert result.stage4 is not None
    assert result.stage6 is not None
    assert result.stage7 is not None
    assert result.chosen_aggregation_method is not None

    warnings: list[str] = []
    # Stage 6 returns "ambiguous" when it cannot confidently pick a
    # method; with no explicit override the runner falls back to
    # "by_loan". Surface that so the analyst knows the choice was a
    # default, not a detection.
    if aggregation == "auto" and result.stage6.detected_method not in (
        "by_loan",
        "by_group",
    ):
        warnings.append(
            f"Aggregation method could not be detected automatically; "
            f"defaulted to '{result.chosen_aggregation_method}'. "
            f"{result.stage6.message}"
        )

    return DryRunResult(
        deal_name=deal_name,
        chosen_aggregation=result.chosen_aggregation_method,
        loan_count=result.stage7.combined_flattened.height,
        property_count=result.stage3.properties.height,
        # Stage 4's collateral_groups frame has one row per property
        # node, so its height is the property-node count - not the
        # group count. The group count is the number of distinct
        # connected components, i.e. distinct collateral_group_id values.
        group_count=result.stage4.collateral_groups["collateral_group_id"].n_unique(),
        warnings=warnings,
    )
