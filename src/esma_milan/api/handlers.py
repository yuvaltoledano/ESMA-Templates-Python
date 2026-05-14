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

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import structlog

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


class ApiError(Exception):
    """A failure that has already been classified into an HTTP response.

    `message` is sanitized and safe to return to the client; `details`
    is optional extra context that is also client-safe. Full diagnostic
    detail - stack traces, pipeline internals - goes to the server-side
    structlog stream, never into an `ApiError`.
    """

    def __init__(
        self, status_code: int, message: str, details: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
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
    if deal_name.strip() == "":
        raise ApiError(400, "deal_name must not be empty.")
    if aggregation not in _AGGREGATION_CHOICES:
        raise ApiError(
            400,
            "aggregation must be one of: auto, by_loan, by_group.",
            details=f"received: {aggregation!r}",
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
            "Pool size exceeds the synchronous endpoint limit "
            "(approximately 30,000 loans; the check counts CSV lines, so "
            "the effective cap is ~29,999 data rows). For larger pools, "
            "use the asynchronous job endpoint at `POST /api/jobs` "
            "(not yet implemented).",
            details=f"estimated {estimated_rows} loan rows in the upload",
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
                "The uploaded data could not be processed.",
                details=_first_line(str(exc)),
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
        group_count=result.stage4.collateral_groups.height,
        warnings=warnings,
    )
