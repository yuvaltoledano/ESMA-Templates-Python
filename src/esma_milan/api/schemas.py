"""Pydantic response schemas for the ESMA-MILAN HTTP API.

Only *response* shapes live here. Requests to ``POST /api/process`` are
multipart (file uploads + form fields), so FastAPI's ``UploadFile`` /
``Form`` parameter types in the endpoint signature handle parsing -
there is no request body to model, and forcing a Pydantic model on it
would fight the framework.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class ErrorResponse(BaseModel):
    """Structured error body returned for every non-2xx API response.

    ``message`` is a sanitized, client-safe summary; ``details`` is
    optional extra context that is also client-safe. Stack traces and
    pipeline internals are never included here - full diagnostic detail
    goes to the server-side structlog stream.
    """

    status_code: int
    message: str
    details: str | None = None
