"""Pool stratifications computed from the Stage-7 combined flattened frame.

Each function takes the Stage-7 `combined_flattened` Polars frame
(one row per loan-part after the aggregation method has been applied,
with the main property's attributes joined on) and returns a
``Stratification`` describing one cut of the pool.

The frame's geographic / occupancy columns reflect the **main property**
chosen by ``pipeline/flatten.select_main_property`` (FOWN preferred,
then highest valuation). For multi-property loans this means the whole
loan balance is attributed to the main property's region / occupancy -
the same convention the existing pipeline applies for the Execution
Summary, kept here so the GUI analysis matches what the workbook reports.
This includes cross-collateralised loans (Stage 4 Type 5): a loan whose
property pool spans regions is attributed entirely to the main property's
region. Known simplification of the existing convention; pro-rata
attribution is a future iteration if domain feedback wants it.

Each function catches its own errors and returns a ``Stratification``
with ``error`` populated rather than raising. One missing column on one
cut must not blow up the other five.

Null handling: ESMA-coded categoricals route both nulls and codes
outside the published taxonomy to a separate ``UNK`` bucket
("Unknown / not in ESMA taxonomy"), which is emitted as a row only
when its count is > 0 - clean data leaves it invisible, malformed
data surfaces it as a data-quality signal. Bucketed numeric strats
(seasoning, LTV) take the opposite route: missing-value rows are
excluded from the buckets entirely and surfaced via
``Stratification.note``. The asymmetry is intentional: an unknown
ESMA code IS a category ("unknown"), but an unknown bucketed number
isn't - there's no "we don't know seasoning" bucket on the number line.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import polars as pl

from esma_milan.analysis.labels import (
    IR_TYPE_LABELS,
    LOAN_PURPOSE_LABELS,
    OCCUPANCY_LABELS,
    UNK_LABEL,
)
from esma_milan.analysis.types import Stratification, StratificationRow, StratificationTotal

ChartType = Literal["pie", "bar"]
StratType = Literal["categorical", "bucketed"]

# Cap on the number of distinct regions shown before collapsing the tail
# into a single "Other" bucket. 10 matches the brief.
GEOGRAPHIC_TOP_N: int = 10

# Seasoning buckets, in months. Per the brief: closed="right" - the
# first bucket's left edge is -inf so it includes 0; "120+" is strictly
# greater than 120.
SEASONING_BREAKS_MONTHS: tuple[float, ...] = (12.0, 24.0, 36.0, 60.0, 120.0)
SEASONING_LABELS: tuple[str, ...] = (
    "0-12 months",
    "13-24 months",
    "25-36 months",
    "37-60 months",
    "61-120 months",
    "120+ months",
)

# Current-LTV buckets, on the same closed="right" convention. The
# combined_flattened frame carries calc_current_LTV as a decimal
# (0.0-1.x), so breaks are decimals too.
LTV_BREAKS: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 1.00)
LTV_LABELS: tuple[str, ...] = (
    "<=50%",
    "50-70%",
    "70-80%",
    "80-90%",
    "90-100%",
    ">100%",
)


def _empty_stratification(
    title: str, strat_type: StratType, chart_type: ChartType, error: str
) -> Stratification:
    """Build an empty Stratification carrying an error string.

    The frontend renders the title and the error message in place of
    the table/chart, so the tile is still visible but flagged unavailable.
    """
    return Stratification(
        title=title,
        type=strat_type,
        chart_type=chart_type,
        rows=[],
        total=StratificationTotal(count=0, balance=0.0),
        error=error,
        note=None,
    )


def _balance_col(df: pl.DataFrame) -> pl.Series:
    """Return the per-loan current balance as a Float64 series, nulls -> 0.

    Stage 7 leaves `current_principal_balance` as the loan-side numeric
    used for every balance-weighted metric in the workbook.
    """
    return (
        df["current_principal_balance"].cast(pl.Float64, strict=False).fill_null(0.0)
    )


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 on a zero denominator. Used for the
    percentage cells so an empty pool doesn't surface NaN to the GUI."""
    return numerator / denominator if denominator > 0 else 0.0


def _build_row(
    label: str, count: int, balance: float,
    total_count: int, total_balance: float,
) -> StratificationRow:
    """Assemble one stratification row with its count and balance shares.

    Percentages are decimals against the table total - safe_div returns
    0.0 on a zero denominator so an empty pool doesn't surface NaN.
    """
    return StratificationRow(
        label=label,
        count=count,
        count_pct=_safe_div(count, total_count),
        balance=balance,
        balance_pct=_safe_div(balance, total_balance),
    )


def _categorical_from_mapping(
    df: pl.DataFrame,
    *,
    source_col: str,
    title: str,
    label_map: dict[str, str],
    chart_type: ChartType = "pie",
) -> Stratification:
    """Generic ESMA-coded categorical stratification.

    Maps each ESMA value code via `label_map` (code -> "CODE - Description"
    full label) and aggregates counts + balances. The output has one row
    for every code in `label_map`, in `label_map` insertion order, even
    when that code has count 0 - the brief's "all spec codes always
    shown" rule, for layout stability across pools.

    Values that the source column carries but aren't in `label_map` -
    null, malformed, or a code outside the published taxonomy - route
    to a separate ``UNK`` row. That row appears only when its count is
    > 0; for valid ESMA data it stays invisible. The dual role is
    deliberate: real ``OTHR`` data goes to its own ``OTHR`` row (since
    OTHR is a valid ESMA code with a legitimate "Other" meaning), and
    UNK signals data-quality issues distinctly.
    """
    if source_col not in df.columns:
        return _empty_stratification(
            title, "categorical", chart_type,
            error=f"Column '{source_col}' not found in pool data.",
        )

    # Aggregate by the raw ESMA code; the label translation happens
    # below so codes outside `label_map` can be routed to UNK explicitly
    # rather than being silently relabeled.
    grouped = (
        df.lazy()
        .select(
            pl.col(source_col).cast(pl.Utf8, strict=False).alias("_code"),
            _balance_col(df).alias("_balance"),
        )
        .group_by("_code")
        .agg(
            pl.len().alias("_count"),
            pl.sum("_balance").alias("_sum"),
        )
        .collect()
    )

    by_code_count: dict[str | None, int] = {}
    by_code_balance: dict[str | None, float] = {}
    for code, count, balance in zip(
        grouped["_code"].to_list(),
        grouped["_count"].to_list(),
        grouped["_sum"].to_list(),
        strict=True,
    ):
        by_code_count[code] = int(count)
        by_code_balance[code] = float(balance)

    total_count = int(sum(by_code_count.values()))
    total_balance = float(sum(by_code_balance.values()))

    rows: list[StratificationRow] = []
    # Always emit every spec code, even count=0.
    for code, label in label_map.items():
        rows.append(_build_row(
            label,
            by_code_count.get(code, 0),
            by_code_balance.get(code, 0.0),
            total_count,
            total_balance,
        ))

    # UNK absorbs nulls and any code outside `label_map`. Only appended
    # when its count is > 0, so clean data leaves the table at the
    # spec-code length.
    unk_count = sum(
        by_code_count[code]
        for code in by_code_count
        if code is None or code not in label_map
    )
    unk_balance = sum(
        by_code_balance[code]
        for code in by_code_balance
        if code is None or code not in label_map
    )
    if unk_count > 0:
        rows.append(_build_row(
            UNK_LABEL, int(unk_count), float(unk_balance),
            total_count, total_balance,
        ))

    return Stratification(
        title=title,
        type="categorical",
        chart_type=chart_type,
        rows=rows,
        total=StratificationTotal(count=total_count, balance=total_balance),
        error=None,
        note=None,
    )


def _weighted_mean_pool(
    classified: pl.DataFrame,
) -> float | None:
    """Balance-weighted mean of `_value` on `classified`, skipping nulls.

    `classified` is the `_value`/`_balance` projection used in
    `_bucketed_numeric`; the WA is computed on raw values (not bucket
    midpoints, which introduce approximation error). Numerator and
    denominator both exclude null-value rows. Distinct from the
    `_weighted_mean` helper in `pipeline/exec_summary.py`, which is
    R-parity-anchored and private: this is a display-only WA where a
    plain polars expression is the right tool.
    """
    valid = classified.filter(
        pl.col("_value").is_not_null() & pl.col("_value").is_finite()
    )
    weight_sum = float(valid["_balance"].sum() or 0.0)
    if weight_sum <= 0 or valid.height == 0:
        return None
    weighted = (valid["_value"] * valid["_balance"]).sum()
    if weighted is None:
        return None
    return float(weighted) / weight_sum


def _bucketed_numeric(
    df: pl.DataFrame,
    *,
    source_col: str,
    title: str,
    breaks: tuple[float, ...],
    labels: tuple[str, ...],
    chart_type: ChartType = "bar",
    transform: Callable[[pl.Expr], pl.Expr] | None = None,
) -> Stratification:
    """Generic bucketed-numeric stratification: cast `source_col` to
    Float64, optionally `transform`, bucket via ``pl.cut`` with
    closed="right", aggregate counts and balances per bucket.

    Rows missing the numeric value (null or non-finite after transform)
    are *excluded* from the bucket rows and surfaced via a footnote
    ("note") rather than silently inflating the buckets. The Total at
    the bottom still equals the pool size; the count_pct / balance_pct
    cells use the bucketed denominator so the percentages within the
    table sum to 100% even when some rows were excluded.
    """
    if source_col not in df.columns:
        return _empty_stratification(
            title, "bucketed", chart_type,
            error=f"Column '{source_col}' not found in pool data.",
        )

    base_expr = pl.col(source_col).cast(pl.Float64, strict=False)
    numeric_expr = transform(base_expr) if transform is not None else base_expr

    classified = (
        df.lazy()
        .select(
            numeric_expr.alias("_value"),
            _balance_col(df).alias("_balance"),
        )
        .with_columns(
            pl.col("_value")
            .cut(breaks=list(breaks), labels=list(labels), left_closed=False)
            .alias("_bucket"),
        )
        .collect()
    )

    # Split missing-numeric rows out: they're counted in the pool total
    # and surfaced via `note`, but not bucketed.
    valid = classified.filter(pl.col("_value").is_not_null() & pl.col("_value").is_finite())
    missing = classified.height - valid.height

    grouped = (
        valid.lazy()
        .group_by("_bucket")
        .agg(
            pl.len().alias("_count"),
            pl.sum("_balance").alias("_sum"),
        )
        .collect()
    )
    bucket_counts = dict(
        zip(grouped["_bucket"].cast(pl.Utf8).to_list(), grouped["_count"].to_list(), strict=True)
    )
    bucket_balances = dict(
        zip(grouped["_bucket"].cast(pl.Utf8).to_list(), grouped["_sum"].to_list(), strict=True)
    )

    bucketed_total_count = int(sum(bucket_counts.values()))
    bucketed_total_balance = float(sum(bucket_balances.values()))

    rows = [
        _build_row(
            bucket_label,
            int(bucket_counts.get(bucket_label, 0)),
            float(bucket_balances.get(bucket_label, 0.0)),
            bucketed_total_count,
            bucketed_total_balance,
        )
        for bucket_label in labels
    ]

    # Pool-level total includes the rows excluded for being missing.
    pool_total_balance = float(_balance_col(df).sum() or 0.0)

    note: str | None = None
    if missing > 0:
        word = "loan" if missing == 1 else "loans"
        note = f"{missing} {word} excluded from buckets due to missing {source_col}."

    weighted_average = _weighted_mean_pool(classified)

    return Stratification(
        title=title,
        type="bucketed",
        chart_type=chart_type,
        rows=rows,
        total=StratificationTotal(count=classified.height, balance=pool_total_balance),
        error=None,
        note=note,
        weighted_average=weighted_average,
    )


# ---------------------------------------------------------------------------
# The six stratifications
# ---------------------------------------------------------------------------


def stratify_interest_rate_type(df: pl.DataFrame) -> Stratification:
    """Pool breakdown by ESMA ``interest_rate_type`` (field RREL42).

    All 13 published ESMA codes are emitted as rows, each labelled
    "CODE — Description" verbatim from the taxonomy. Codes outside the
    taxonomy and nulls route to a separate UNK row that appears only
    when its count is > 0."""
    return _categorical_from_mapping(
        df,
        source_col="interest_rate_type",
        title="Interest Rate Type",
        label_map=IR_TYPE_LABELS,
        chart_type="pie",
    )


def stratify_seasoning(df: pl.DataFrame) -> Stratification:
    """Months since origination, bucketed per the brief.

    Stage 7 carries `calc_seasoning` as float years; multiply by 12 for
    the month-based buckets. Negative seasoning (origination after pool
    cutoff) is a data-quality issue surfaced by the pipeline; here it
    lands in the first bucket since closed="right" admits any value
    <= 12.
    """
    return _bucketed_numeric(
        df,
        source_col="calc_seasoning",
        title="Seasoning (months)",
        breaks=SEASONING_BREAKS_MONTHS,
        labels=SEASONING_LABELS,
        chart_type="bar",
        transform=lambda e: e * 12.0,
    )


def stratify_current_ltv(df: pl.DataFrame) -> Stratification:
    """Current LTV bucketed at 50 / 70 / 80 / 90 / 100. Source column
    is the Stage-7 group-level `calc_current_LTV` (decimal)."""
    return _bucketed_numeric(
        df,
        source_col="calc_current_LTV",
        title="Current LTV",
        breaks=LTV_BREAKS,
        labels=LTV_LABELS,
        chart_type="bar",
    )


def stratify_geographic(df: pl.DataFrame) -> Stratification:
    """Pool breakdown by main-property region.

    `geographic_region_collateral` in `combined_flattened` is the main
    property's region (see select_main_property). Top 10 regions by
    current balance are listed; remaining regions collapse into "Other".
    Nulls are routed to "Other".
    """
    source_col = "geographic_region_collateral"
    title = "Geographic Distribution"
    chart_type: ChartType = "bar"

    if source_col not in df.columns:
        return _empty_stratification(
            title, "categorical", chart_type,
            error=f"Column '{source_col}' not found in pool data.",
        )

    grouped = (
        df.lazy()
        .select(
            pl.col(source_col)
            .cast(pl.Utf8, strict=False)
            .fill_null("Other")
            .alias("_label"),
            _balance_col(df).alias("_balance"),
        )
        .group_by("_label")
        .agg(
            pl.len().alias("_count"),
            pl.sum("_balance").alias("_sum"),
        )
        .sort("_sum", descending=True)
        .collect()
    )

    if grouped.height == 0:
        return Stratification(
            title=title, type="categorical", chart_type=chart_type,
            rows=[], total=StratificationTotal(count=0, balance=0.0),
            error=None, note=None,
        )

    # Separate "Other" out so it can be combined with any tail beyond top-N.
    # (When the source data already contains "Other" rows from nulls, we
    # don't want to double-count by listing it both in the top-N and the
    # tail bucket.)
    explicit_other = grouped.filter(pl.col("_label") == "Other")
    non_other = grouped.filter(pl.col("_label") != "Other")

    top = non_other.head(GEOGRAPHIC_TOP_N)
    tail = non_other.slice(GEOGRAPHIC_TOP_N, non_other.height)

    other_count = int(explicit_other["_count"].sum() or 0) + int(tail["_count"].sum() or 0)
    other_balance = (
        float(explicit_other["_sum"].sum() or 0.0) + float(tail["_sum"].sum() or 0.0)
    )

    total_count = int(grouped["_count"].sum() or 0)
    total_balance = float(grouped["_sum"].sum() or 0.0)

    rows: list[StratificationRow] = []
    for label, count, balance in zip(
        top["_label"].to_list(), top["_count"].to_list(), top["_sum"].to_list(), strict=True
    ):
        rows.append(
            StratificationRow(
                label=str(label),
                count=int(count),
                count_pct=_safe_div(int(count), total_count),
                balance=float(balance),
                balance_pct=_safe_div(float(balance), total_balance),
            )
        )
    if other_count > 0:
        rows.append(
            StratificationRow(
                label="Other",
                count=other_count,
                count_pct=_safe_div(other_count, total_count),
                balance=other_balance,
                balance_pct=_safe_div(other_balance, total_balance),
            )
        )

    return Stratification(
        title=title,
        type="categorical",
        chart_type=chart_type,
        rows=rows,
        total=StratificationTotal(count=total_count, balance=total_balance),
        error=None,
        note=None,
    )


def stratify_loan_purpose(df: pl.DataFrame) -> Stratification:
    """Pool breakdown by ESMA ``purpose`` (field RREL27).

    All 13 published ESMA codes are emitted as rows with their full
    "CODE — Description" labels. See `stratify_interest_rate_type` for
    the UNK fallback behaviour."""
    return _categorical_from_mapping(
        df,
        source_col="purpose",
        title="Loan Purpose",
        label_map=LOAN_PURPOSE_LABELS,
        chart_type="pie",
    )


def stratify_occupancy(df: pl.DataFrame) -> Stratification:
    """Pool breakdown by main-property ESMA ``occupancy_type`` (field RREC7).

    Reflects the main property only; see module docstring for the
    multi-property attribution convention. All 5 published ESMA codes
    emitted as rows; see `stratify_interest_rate_type` for the UNK
    fallback behaviour."""
    return _categorical_from_mapping(
        df,
        source_col="occupancy_type",
        title="Occupancy",
        label_map=OCCUPANCY_LABELS,
        chart_type="pie",
    )
