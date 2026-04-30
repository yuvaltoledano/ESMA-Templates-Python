"""Stage 10: Execution Summary helpers.

Mirrors the metric calculations and formatting helpers in the
Execution Summary block of r_reference/R/pipeline.R:625-895. The
composer itself lands in C-2; this module covers the primitives.

Design notes:

  - Format helpers (_fmt_pct / _fmt_comma / _fmt_comma_int) are
    scalar-input, str-output. Output strings are byte-equal to R's
    `scales::*` defaults at the time the synthetic fixture was
    generated. If R's `scales` package updates its defaults,
    regenerate the fixture and re-pin the helper tests.

  - Metric helpers that aggregate over multiple rows (per-borrower
    grouping, weighted means) sort their inputs deterministically
    before accumulating. Polars' group_by output order is
    implementation-defined; R's `aggregate` returns sorted-by-key.
    Sorting both first gives the same accumulation order across
    engines and eliminates a class of latent IEEE-754 drift bugs
    that would otherwise surface only on real-fixture parity.
    Cost: one sort per metric. Benefit: byte-equal totals on
    real_anonymised data without per-bug forensics.

  - _milan_as_num collapses BOTH "ND" tokens AND null / empty
    strings to null - matching R's milan_as_num(x). The asymmetric
    defaults from R-repo entry #9 (Pari Passu emits "0" on ND
    input while External Prior emits "ND") flow through here:
    "0" contributes 0 to a sum, "ND" is excluded entirely.
    Verified by test_milan_as_num_handles_asymmetric_defaults.
"""

from __future__ import annotations

import math

import polars as pl

# ---------------------------------------------------------------------------
# Sentinel for missing values in formatted output
# ---------------------------------------------------------------------------

_NOT_AVAILABLE = "<not available>"


# ---------------------------------------------------------------------------
# Format helpers (scalar -> str)
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    """Mirror R's `fmt_pct` -> `scales::percent(x, accuracy = 0.01)`.

    Returns "12.34%" for 0.1234, "<not available>" for None / NaN.
    Round-half-to-even (banker's rounding) matches R's `scales::percent`
    default, which itself wraps `format(round(x*100, 2), nsmall=2)`.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return _NOT_AVAILABLE
    return f"{x:.2%}"


def _fmt_comma(x: float | None) -> str:
    """Mirror R's `fmt_comma` -> `scales::comma(x, accuracy = 0.01)`.

    Returns "1,234,567.89" for 1234567.89, "<not available>" for
    None / NaN.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return _NOT_AVAILABLE
    return f"{x:,.2f}"


def _fmt_comma_int(x: int | None) -> str:
    """Mirror R's `fmt_comma_int` -> `scales::comma(x)`.

    Returns "1,234" for 1234, "<not available>" for None.
    Input is constrained to int so the helper can't be misused on
    floats - R's `scales::comma` without accuracy rounds non-integer
    inputs to integer, but every call site in the pipeline passes a
    distinct-count.
    """
    if x is None:
        return _NOT_AVAILABLE
    return f"{x:,}"


# ---------------------------------------------------------------------------
# _milan_as_num
# ---------------------------------------------------------------------------


def _milan_as_num(s: pl.Series) -> pl.Series:
    """Coerce a MILAN string column to Float64, collapsing "ND" -> null.

    Mirrors r_reference/R/pipeline.R:629-631 (`milan_as_num`):

        suppressWarnings(as.numeric(if_else(x == "ND", NA_character_, x)))

    Asymmetric-defaults note (R-repo entry #9): "ND" -> null (excluded
    from sums); "0" -> 0.0 (contributes 0 to sums but counts as a row
    with a value). This asymmetry is intentional in MILAN's contract -
    External Prior Ranks emits "ND" on missing-input rows, Pari Passu
    emits "0", and Total External Prior Ranks vs Total External Equal
    Ranks compute differently as a consequence.
    """
    df = pl.DataFrame({"x": s.cast(pl.Utf8, strict=False)})
    return df.select(
        pl.when((pl.col("x") == "ND") | pl.col("x").is_null())
        .then(None)
        .otherwise(pl.col("x"))
        .cast(pl.Float64, strict=False)
        .alias(s.name)
    )[s.name]


# ---------------------------------------------------------------------------
# Borrower-level aggregations (deterministic sort)
# ---------------------------------------------------------------------------


def _borrower_balances_sorted_desc(
    borrower_ids: pl.Series,
    balances: pl.Series,
) -> list[float]:
    """Group balances by borrower_id, return list sorted descending.

    Filters out null / "ND" borrower_ids and null balances (matches
    R's `borrower_mask <- !is.na(m_borrower_ids) & m_borrower_ids
    != "ND" & !is.na(m_loan_cb)`).

    Tie-breaks identical-balance borrowers by borrower_id ascending
    so the output ordering is fully deterministic AND matches R's
    `sort(decreasing=TRUE)` applied to the alphabetically-sorted
    output of R's `aggregate(...)` (R's aggregate returns rows
    sorted by group key; sort-desc on numeric values preserves
    insertion order on ties).

    Within-group sum order: Polars' `group_by("id").agg(pl.col("cb")
    .sum())` sums values in storage order within each group, which
    is the same row order R's `aggregate(..., FUN=sum)` uses. No
    intra-group sort imposed - matching R's order rather than
    layering Python-side determinism that would diverge.
    """
    df = pl.DataFrame({"id": borrower_ids, "cb": balances}).filter(
        pl.col("id").is_not_null()
        & (pl.col("id") != "ND")
        & pl.col("cb").is_not_null()
    )
    if df.height == 0:
        return []
    agg = df.group_by("id").agg(pl.col("cb").sum().alias("cb"))
    # Tie-break on id ascending: matches R's alphabetical group-key
    # ordering preserved by sort-desc on numeric values.
    agg = agg.sort(["cb", "id"], descending=[True, False])
    return agg["cb"].to_list()


def _top_20_borrower_exposure(
    borrower_ids: pl.Series,
    balances: pl.Series,
    total_cb: float,
) -> float | None:
    """sum(top 20 borrower balances) / total_cb.

    Mirrors r_reference/R/pipeline.R:670-684. Iterates the top-20
    slice in descending order - same as R's `sum(head(borrower_balances,
    20))` where `borrower_balances` is sort-desc. Returns None if
    total_cb <= 0 or no eligible borrowers.
    """
    if total_cb <= 0:
        return None
    sorted_desc = _borrower_balances_sorted_desc(borrower_ids, balances)
    if not sorted_desc:
        return None
    return sum(sorted_desc[:20]) / total_cb


def _effective_borrowers(
    borrower_ids: pl.Series,
    balances: pl.Series,
    total_cb: float,
) -> float | None:
    """Inverse Herfindahl-Hirschman: 1 / sum((share)^2).

    Mirrors r_reference/R/pipeline.R:670-684:

        effective_borrowers <- 1 / sum((borrower_balances / total_cb)^2)

    where `borrower_balances` is already sort-descending. R sums in
    descending order; this function iterates the same desc-sorted
    list directly without re-sorting. Imposing an ascending sort
    here would diverge from R's accumulation order on IEEE-754 -
    matching R is the goal, not Python-side determinism (see
    R-repo issue tracker entry #10).
    """
    if total_cb <= 0:
        return None
    sorted_desc = _borrower_balances_sorted_desc(borrower_ids, balances)
    if not sorted_desc:
        return None
    # Iterate in descending order - same as R's `sum((borrower_balances/total_cb)^2)`
    # where borrower_balances is sort-desc. Do NOT re-sort: matching R's
    # accumulation order is the goal.
    s = sum((b / total_cb) ** 2 for b in sorted_desc)
    return 1.0 / s if s > 0 else None


# ---------------------------------------------------------------------------
# Weighted mean / pct_of_cb / structure breakdown
# ---------------------------------------------------------------------------


def _weighted_mean(values: pl.Series, weights: pl.Series) -> float | None:
    """R `weighted.mean(values, weights, na.rm=TRUE)` equivalent.

    Skips index pairs where either values[i] or weights[i] is null /
    NaN. NO sort applied - R's `weighted.mean` accumulates in input
    row order, and Polars' `Series.sum()` after `filter` does the
    same in storage order. Imposing a sort here would diverge from
    R's accumulation order on IEEE-754 (see R-repo issue tracker
    entry #10: matching R is the goal, not Python-side determinism).
    """
    df = pl.DataFrame({"v": values.cast(pl.Float64, strict=False),
                       "w": weights.cast(pl.Float64, strict=False)}).filter(
        pl.col("v").is_not_null() & pl.col("w").is_not_null()
        & ~pl.col("v").is_nan() & ~pl.col("w").is_nan()
    )
    if df.height == 0:
        return None
    weight_sum = df["w"].sum()
    if weight_sum is None or weight_sum == 0:
        return None
    weighted_sum = (df["v"] * df["w"]).sum()
    return float(weighted_sum) / float(weight_sum)


def _pct_of_cb_mask(
    mask: pl.Series,
    balances: pl.Series,
    total_cb: float,
) -> float | None:
    """sum(balances[mask]) / total_cb, with zero-denominator guard.

    Mirrors r_reference/R/pipeline.R:768-771 (`pct_of_cb`). Returns
    None if total_cb <= 0; mask all False -> 0.0 / total_cb = 0.0.
    """
    if total_cb <= 0:
        return None
    df = pl.DataFrame({"m": mask, "b": balances.cast(pl.Float64, strict=False)})
    masked_sum = df.filter(pl.col("m").fill_null(False))["b"].sum()
    if masked_sum is None:
        return 0.0
    return float(masked_sum) / total_cb


def _structure_breakdown(structure_type_col: pl.Series) -> pl.DataFrame:
    """Per-structure-type loan count rows for the bottom of the summary.

    Mirrors r_reference/R/pipeline.R:805-820. Returns a 2-column
    Polars DataFrame:

        Metric                                      | Value
        Loan Count: 1: one loan -> one property     | "1"
        Loan Count: 2: one loan -> multiple props   | "1"
        ...

    Sorted by structure_type ascending (lex-sort, matches R's
    `order(structure_df$structure_type)`). Returns an empty 2-col
    frame when the input is all null / empty (so concat with the
    base 38-row frame still works).
    """
    df = pl.DataFrame({"st": structure_type_col}).filter(
        pl.col("st").is_not_null()
    )
    if df.height == 0:
        return pl.DataFrame(
            {"Metric": [], "Value": []},
            schema={"Metric": pl.Utf8, "Value": pl.Utf8},
        )
    counts = (
        df.group_by("st")
        .agg(pl.len().alias("n"))
        .sort("st")
    )
    return pl.DataFrame(
        {
            "Metric": [f"Loan Count: {st}" for st in counts["st"].to_list()],
            "Value": [_fmt_comma_int(int(n)) for n in counts["n"].to_list()],
        },
        schema={"Metric": pl.Utf8, "Value": pl.Utf8},
    )
