"""Pydantic response schemas for the ESMA-MILAN HTTP API.

Only *response* shapes live here. Requests to ``POST /api/process`` are
multipart (file uploads + form fields), so FastAPI's ``UploadFile`` /
``Form`` parameter types in the endpoint signature handle parsing -
there is no request body to model, and forcing a Pydantic model on it
would fight the framework.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Stable, machine-readable error codes. Clients branch on `error` rather
# than parsing `message` strings, so these are part of the API contract:
# add new codes as failure modes appear, never repurpose an existing one.
#
# The vocabulary is defined in full here even though the validators that
# raise `file_too_large`, `invalid_content_type`, and `invalid_deal_name`
# land in the Day 2 hardening commit - keeping the type stable across
# commits is cheaper than widening it twice.
ErrorCode = Literal[
    "missing_field",  # a required upload or form field was absent
    "validation_error",  # a field was present but malformed (bad type/range)
    "invalid_csv",  # uploaded CSV could not be parsed by the pipeline
    "size_limit_exceeded",  # pool exceeds the synchronous loan-count guardrail
    "file_too_large",  # a single upload exceeds the per-file byte limit
    "invalid_content_type",  # an upload's Content-Type is not accepted
    "invalid_deal_name",  # deal_name failed character/length validation
    "rate_limit_exceeded",  # per-IP request rate limit tripped
    "internal_error",  # an unexpected server-side failure
]


class HealthResponse(BaseModel):
    """Body of ``GET /api/health``."""

    status: str = "ok"


class DryRunResponse(BaseModel):
    """Body of ``POST /api/process`` when ``dry_run=true``.

    The structured result of running the pipeline without composing a
    workbook - counts and the resolved aggregation method, designed
    around what the pipeline naturally produces in its dry-run path.
    """

    deal_name: str

    chosen_aggregation: str
    """The aggregation method the pipeline actually used: ``"by_loan"``
    or ``"by_group"``. An ``"auto"`` request resolves to one of these."""

    loan_count: int
    """Rows in the final combined flattened pool (Stage 7 output)."""

    property_count: int
    """Properties surviving the Stage-3 intersection filter."""

    group_count: int
    """Distinct loan-collateral groups from the Stage-4 bipartite graph."""

    warnings: list[str] = Field(default_factory=list)
    """Non-fatal advisories surfaced by the pipeline, e.g. an ambiguous
    aggregation method that fell back to a default."""


class StratificationRow(BaseModel):
    """One row in a stratification table.

    `count_pct` / `balance_pct` are decimals in [0, 1]; the GUI formats
    them as percentages.
    """

    label: str
    count: int
    count_pct: float
    balance: float
    balance_pct: float


class StratificationTotal(BaseModel):
    """The total line at the bottom of a stratification table."""

    count: int
    balance: float


class Stratification(BaseModel):
    """One cut of the pool: a table plus a chart hint.

    See ``esma_milan.analysis.types.Stratification`` for the design
    rationale; this is the Pydantic mirror used for FastAPI response
    validation.
    """

    title: str
    type: Literal["categorical", "bucketed"]
    chart_type: Literal["pie", "bar"]
    rows: list[StratificationRow] = Field(default_factory=list)
    total: StratificationTotal
    error: str | None = None
    note: str | None = None
    weighted_average: float | None = None


class ExecutionSummaryRow(BaseModel):
    """One row of the Execution Summary (Sheet 1).

    `value` is a pre-formatted string with R-faithful formatting:
    currencies as "1,234,567.89", percentages as "12.34%", dates as
    ISO, missing as "<not available>". See
    ``esma_milan.analysis.types.ExecutionSummaryRow`` for rationale.
    """

    label: str
    value: str


class AnalysisSummary(BaseModel):
    """Top-level pool summary returned alongside the stratifications."""

    loan_count: int
    property_count: int
    group_count: int
    total_current_balance: float
    chosen_aggregation: str
    warnings: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Body of ``POST /api/process`` when ``analysis_only=true``.

    The pipeline runs through Stage 7 (the workbook itself is not
    composed); the response carries the six pool stratifications as
    pre-aggregated JSON for the GUI to render inline.
    """

    deal_name: str
    summary: AnalysisSummary
    stratifications: dict[str, Stratification]
    execution_summary: list[ExecutionSummaryRow] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Structured error body returned for every non-2xx API response.

    ``error`` is a stable machine-readable code (see ``ErrorCode``) that
    clients branch on; ``message`` is a sanitized, client-safe summary
    for humans; ``details`` is optional structured context that is also
    client-safe (e.g. which field failed, what limit was exceeded). The
    HTTP status lives on the response itself, not in the body. Stack
    traces and pipeline internals are never included here - full
    diagnostic detail goes to the server-side structlog stream.
    """

    error: ErrorCode
    message: str
    details: dict[str, object] | None = None
