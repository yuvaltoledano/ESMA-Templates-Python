"""Dataclasses describing the analysis-mode response.

These are the Python-side carriers; ``api/schemas.py`` mirrors them as
Pydantic models for FastAPI's response validation. Two parallel types
rather than one (Pydantic) type is the same shape the existing
`WorkbookResult`/`DryRunResult`/`DryRunResponse` split uses in
`handlers.py` + `schemas.py`: pipeline-facing logic stays Pydantic-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class StratificationRow:
    """One row in a stratification table.

    `count_pct` and `balance_pct` are decimals in [0, 1]; the frontend
    formats them as percentages. Keeping them as decimals here avoids
    a unit-mismatch trap (some downstream consumer doing `value * 100`
    on an already-percentaged number).
    """

    label: str
    count: int
    count_pct: float
    balance: float
    balance_pct: float


@dataclass(frozen=True)
class StratificationTotal:
    """The "Total" line at the bottom of a stratification table.

    For bucketed cuts (seasoning, LTV) `count`/`balance` equal the full
    pool size, even when some rows were excluded from the buckets for
    missing data (the exclusion is surfaced via `Stratification.note`).
    For categoricals, nulls are routed to the "Other" bucket, so the
    total equals the row sum by construction.
    """

    count: int
    balance: float


@dataclass(frozen=True)
class Stratification:
    """A single cut of the pool: title, table rows, total, chart hint.

    `error` populated means the cut couldn't be computed (typically a
    missing column on this particular dataset); rows/total are empty
    in that case and the frontend renders a muted "unavailable" tile
    rather than the table/chart.

    `note` is an optional footnote: e.g. "N loans excluded from buckets
    due to missing X". Surfaced rather than silently dropped per the
    brief.
    """

    title: str
    type: Literal["categorical", "bucketed"]
    chart_type: Literal["pie", "bar"]
    rows: list[StratificationRow] = field(default_factory=list)
    total: StratificationTotal = field(
        default_factory=lambda: StratificationTotal(count=0, balance=0.0)
    )
    error: str | None = None
    note: str | None = None
    weighted_average: float | None = None
    """Pool-level balance-weighted mean of the source column, computed
    on raw loan-level values (not bucket midpoints, which would
    introduce approximation error). Populated only for bucketed
    numeric stratifications (seasoning, current_ltv) where a single
    pool-level WA has a meaningful interpretation. None for
    categoricals (IR type, loan purpose, occupancy) and for the
    geographic stratification. Units match the source column: months
    for seasoning, decimal for LTV (0.673 not 67.3)."""


@dataclass(frozen=True)
class ExecutionSummaryRow:
    """One row of the Execution Summary table (Sheet 1).

    Value is a pre-formatted string with the same byte-shape the
    workbook writes - currencies as "1,234,567.89", percentages as
    "12.34%", dates as ISO, missing values as the "<not available>"
    sentinel. The heterogeneous types in the source sheet
    (currency / pct / count / ratio / date) make a typed numeric
    field impractical; pre-formatting server-side avoids duplicating
    the R-faithful format logic on the frontend.
    """

    label: str
    value: str


@dataclass(frozen=True)
class AnalysisSummary:
    """Top-level pool summary that sits alongside the stratifications.

    Same field set as `DryRunResult` plus `total_current_balance`, so
    the GUI can render a one-line "X loans, Y properties, EUR Z total"
    header without consuming the stratification tables.
    """

    loan_count: int
    property_count: int
    group_count: int
    total_current_balance: float
    chosen_aggregation: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisResult:
    """Analysis-mode result returned by `process_pipeline_from_bytes`.

    Parallels `WorkbookResult` and `DryRunResult` - one of the three
    branches of `ProcessResult` the handler returns. `server.py`
    serialises this via the `AnalysisResponse` Pydantic model in
    `api/schemas.py`.
    """

    deal_name: str
    summary: AnalysisSummary
    stratifications: dict[str, Stratification]
    execution_summary: list[ExecutionSummaryRow] = field(default_factory=list)
