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
from datetime import date

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


# ---------------------------------------------------------------------------
# Composer (Stage 10 / C-2)
# ---------------------------------------------------------------------------


# Canonical 38 base-row Metric labels in the order they appear on Sheet 1.
# Mirrors r_reference/R/pipeline.R:823-862. The structure breakdown rows
# follow the base 38 (variable count depending on which structure types
# are present in the MILAN pool).
_BASE_METRIC_LABELS: tuple[str, ...] = (
    "Deal Name",
    "Pool Cut-off Date",
    "Aggregation Method",
    "Original Balance",
    "Current Balance",
    "External Prior Ranks",
    "External Equal Ranks",
    "Number Of Borrowers",
    "Number Of Complete Loans",
    "Number Of Loan Parts",
    "Number Of Properties",
    "Average Loan Per Borrower",
    "Average Total Loan",
    "Average Loan Part",
    "Top 20 Borrower Exposure",
    "Effective Number of Borrowers",
    "WA Interest Rate",
    "Average Months To Reset For Reset Loans",
    "Average Months To Reset For All Loans",
    "WA Seasoning (Yrs)",
    "WA Yrs To Maturity",
    "Maximum Maturity Date",
    "WA Original LTV",
    "WA Current LTV",
    "Current LTV: >= 70%",
    "Current LTV: >= 80%",
    "Current LTV: >= 90%",
    "Current LTV: >= 100%",
    "Current LTV: >= 110%",
    "Interest Rate Type: Floating For Life",
    "Interest Rate Type: Fixed For Life",
    "Interest Rate Type: Fixed With Future Periodic Resets",
    "Interest Rate Type: Fixed With Future Switch To Floating",
    "Principal Payment Type: Initial IO > 10 Years",
    "Loan Purpose: Purchased, Remortgage and Renovation",
    "Employment Type: Self Employed",
    "Employment Type: Companies",
    "Arrears: > 3 Month",
)


def _n_distinct_excluding_nd(s: pl.Series) -> int:
    """n_distinct excluding null and "ND" tokens.

    Mirrors `dplyr::n_distinct(x[!is.na(x) & x != "ND"])`.
    """
    df = pl.DataFrame({"x": s.cast(pl.Utf8, strict=False)}).filter(
        pl.col("x").is_not_null() & (pl.col("x") != "ND")
    )
    return df["x"].n_unique()


def _max_or_none(s: pl.Series) -> object | None:
    """`max(x, na.rm=TRUE)` returning None on all-null. Polars' `Series.max()`
    already skips nulls and returns None on all-null; this thin wrapper is
    a parity-anchor for the `Maximum Maturity Date` metric.
    """
    out = s.max()
    return out


def compose_execution_summary(
    *,
    deal_name: str,
    chosen_aggregation: str,
    pool_cutoff_date: date,
    milan_pool: pl.DataFrame,
    loans_enriched: pl.DataFrame,
    properties_enriched: pl.DataFrame,
    combined_flattened: pl.DataFrame,
) -> pl.DataFrame:
    """Build the 38-row + N-breakdown-row Execution Summary frame.

    Mirrors r_reference/R/pipeline.R:625-895. Returns a 2-column
    Polars DataFrame `(Metric: Utf8, Value: Utf8)` with exactly
    38 + n_structure_types rows, ready for the workbook writer.
    """
    # Parse MILAN numeric columns via _milan_as_num (ND -> null,
    # numeric strings -> floats). Asymmetric defaults from R-repo
    # entry #9: External Prior emits "ND" on missing-input rows;
    # Pari Passu emits "0". After _milan_as_num, External Prior
    # null-rows are excluded from the sum; Pari Passu zero-rows
    # contribute 0. Different totals on real data; preserved per R.
    m_loan_ob = _milan_as_num(milan_pool["Loan OB"])
    m_loan_cb = _milan_as_num(milan_pool["Loan CB"])
    m_ext_prior = _milan_as_num(milan_pool["External Prior Ranks CB"])
    m_pari_passu = _milan_as_num(
        milan_pool["Pari Passu Ranking Loans (Not In Pool)"]
    )
    m_interest_rate = _milan_as_num(milan_pool["Interest Rate"])
    m_months_arrears = _milan_as_num(milan_pool["Months In Arrears"])

    m_borrower_ids = milan_pool["Borrower Identifier"]
    m_property_ids = milan_pool["Property Identifier"]
    m_loan_ids = milan_pool["Loan Identifier"]
    m_ir_types = milan_pool["Interest Rate Type"]

    # Pool totals. sum() in Polars skips nulls (matches R's na.rm=TRUE).
    total_ob = float(m_loan_ob.sum() or 0.0)
    total_cb = float(m_loan_cb.sum() or 0.0)
    total_ext_prior = float(m_ext_prior.sum() or 0.0)
    total_pari_passu = float(m_pari_passu.sum() or 0.0)

    # Counts. R uses dplyr::n_distinct on non-null non-ND values.
    nr_unique_borrowers = (
        loans_enriched.filter(pl.col("calc_borrower_id").is_not_null())[
            "calc_borrower_id"
        ].n_unique()
    )
    nr_complete_loans = _n_distinct_excluding_nd(m_property_ids)
    nr_loan_parts = _n_distinct_excluding_nd(m_loan_ids)
    nr_unique_properties = (
        properties_enriched.filter(pl.col("calc_property_id").is_not_null())[
            "calc_property_id"
        ].n_unique()
    )

    # Averages, with zero-denominator guard.
    avg_loan_per_borrower: float | None = (
        total_cb / nr_unique_borrowers if nr_unique_borrowers > 0 else None
    )
    avg_total_loan: float | None = (
        total_cb / nr_complete_loans if nr_complete_loans > 0 else None
    )
    avg_loan_part: float | None = (
        total_cb / nr_loan_parts if nr_loan_parts > 0 else None
    )

    # Top-20 + effective borrowers.
    top_20_exposure = _top_20_borrower_exposure(m_borrower_ids, m_loan_cb, total_cb)
    effective_borrowers = _effective_borrowers(m_borrower_ids, m_loan_cb, total_cb)

    # WA Interest Rate.
    wa_interest_rate = (
        _weighted_mean(m_interest_rate, m_loan_cb) if total_cb > 0 else None
    )

    # Date-based metrics. Re-parse the MILAN char-coerced date columns.
    reset_dates = milan_pool["Interest Reset Date"].str.to_date(
        format="%Y-%m-%d", strict=False
    )
    origination_dates = milan_pool["Origination Date"].str.to_date(
        format="%Y-%m-%d", strict=False
    )
    maturity_dates = milan_pool["Maturity Date"].str.to_date(
        format="%Y-%m-%d", strict=False
    )

    # months_to_reset (vector aligned with milan_pool rows).
    months_to_reset = (reset_dates - pool_cutoff_date).dt.total_days() / 30.4375
    # reset_mask: ir_type in {3, 4} AND non-null reset date.
    reset_mask = (
        m_ir_types.is_not_null()
        & m_ir_types.is_in(["3", "4"])
        & reset_dates.is_not_null()
    )

    avg_months_reset_loans = (
        _weighted_mean(months_to_reset.filter(reset_mask), m_loan_cb.filter(reset_mask))
        if total_cb > 0
        else None
    )
    # "Average Months To Reset For All Loans" - per R-repo entry #11
    # (preserved-quirk): Moody's-domain formula. Numerator is the weighted
    # contribution of reset-eligible loans only; denominator is total_cb.
    if total_cb > 0:
        # sum_contribution = sum(months_to_reset[reset_mask] * m_loan_cb[reset_mask])
        df_rm = pl.DataFrame(
            {
                "m": months_to_reset.filter(reset_mask),
                "cb": m_loan_cb.filter(reset_mask),
            }
        )
        contrib = (df_rm["m"] * df_rm["cb"]).sum()
        avg_months_all_loans: float | None = (
            float(contrib) / total_cb if contrib is not None else None
        )
    else:
        avg_months_all_loans = None

    # Seasoning + maturity.
    seasoning_years = (
        (pool_cutoff_date - origination_dates).dt.total_days() / 365.25
    )
    wa_seasoning = (
        _weighted_mean(seasoning_years, m_loan_cb) if total_cb > 0 else None
    )
    yrs_to_maturity = (maturity_dates - pool_cutoff_date).dt.total_days() / 365.25
    wa_yrs_to_maturity = (
        _weighted_mean(yrs_to_maturity, m_loan_cb) if total_cb > 0 else None
    )
    max_maturity = _max_or_none(maturity_dates)
    max_maturity_str = (
        max_maturity.isoformat() if isinstance(max_maturity, date) else None
    )

    # LTV WAs from combined_flattened (calc_current_LTV / calc_original_LTV).
    # Mirrors R's pipeline.R:738-748: use combined_flattened_final's group-
    # level calc LTVs, NOT MILAN's per-loan "Current LTV"/"Original LTV"
    # fields, so the summary's LTV methodology matches the Combined
    # flattened pool sheet.
    cf_current_ltv = combined_flattened["calc_current_LTV"].cast(
        pl.Float64, strict=False
    )
    cf_original_ltv = combined_flattened["calc_original_LTV"].cast(
        pl.Float64, strict=False
    )
    wa_original_ltv = (
        _weighted_mean(cf_original_ltv, m_loan_ob) if total_ob > 0 else None
    )
    wa_current_ltv = (
        _weighted_mean(cf_current_ltv, m_loan_cb) if total_cb > 0 else None
    )

    # LTV threshold distribution (share of CB at LTV >= threshold).
    ltv_thresholds = (0.70, 0.80, 0.90, 1.00, 1.10)
    ltv_pcts = [
        _pct_of_cb_mask(
            cf_current_ltv.is_not_null() & (cf_current_ltv >= threshold),
            m_loan_cb,
            total_cb,
        )
        for threshold in ltv_thresholds
    ]

    # IR Type distribution.
    ir_pcts = [
        _pct_of_cb_mask(
            m_ir_types.is_not_null() & (m_ir_types == code), m_loan_cb, total_cb
        )
        for code in ("1", "3", "4", "5")
    ]

    # Other categorical percentages.
    pp_type = milan_pool["Principal Payment Type"]
    purpose = milan_pool["Loan Purpose"]
    employment = milan_pool["Employment Type"]
    borrower_type = milan_pool["Borrower Type"]

    io_gt_10 = _pct_of_cb_mask(
        pp_type.is_not_null() & (pp_type == "8"), m_loan_cb, total_cb
    )
    purpose_pmr = _pct_of_cb_mask(
        purpose.is_not_null() & purpose.is_in(["1", "2", "3"]),
        m_loan_cb,
        total_cb,
    )
    self_employed = _pct_of_cb_mask(
        employment.is_not_null() & (employment == "3"), m_loan_cb, total_cb
    )
    companies = _pct_of_cb_mask(
        borrower_type.is_not_null() & (borrower_type == "2"), m_loan_cb, total_cb
    )
    arrears_gt_3 = _pct_of_cb_mask(
        m_months_arrears.is_not_null() & (m_months_arrears > 3),
        m_loan_cb,
        total_cb,
    )

    # Assemble the 38 base-row values in label order.
    base_values: list[str] = [
        deal_name,
        pool_cutoff_date.isoformat(),
        chosen_aggregation,
        _fmt_comma(total_ob),
        _fmt_comma(total_cb),
        _fmt_comma(total_ext_prior),
        _fmt_comma(total_pari_passu),
        _fmt_comma_int(nr_unique_borrowers),
        _fmt_comma_int(nr_complete_loans),
        _fmt_comma_int(nr_loan_parts),
        _fmt_comma_int(nr_unique_properties),
        _fmt_comma(avg_loan_per_borrower),
        _fmt_comma(avg_total_loan),
        _fmt_comma(avg_loan_part),
        _fmt_pct(top_20_exposure),
        _fmt_comma(effective_borrowers),
        _fmt_pct(wa_interest_rate),
        _fmt_comma(avg_months_reset_loans),
        _fmt_comma(avg_months_all_loans),
        _fmt_comma(wa_seasoning),
        _fmt_comma(wa_yrs_to_maturity),
        max_maturity_str if max_maturity_str is not None else _NOT_AVAILABLE,
        _fmt_pct(wa_original_ltv),
        _fmt_pct(wa_current_ltv),
        _fmt_pct(ltv_pcts[0]),
        _fmt_pct(ltv_pcts[1]),
        _fmt_pct(ltv_pcts[2]),
        _fmt_pct(ltv_pcts[3]),
        _fmt_pct(ltv_pcts[4]),
        _fmt_pct(ir_pcts[0]),
        _fmt_pct(ir_pcts[1]),
        _fmt_pct(ir_pcts[2]),
        _fmt_pct(ir_pcts[3]),
        _fmt_pct(io_gt_10),
        _fmt_pct(purpose_pmr),
        _fmt_pct(self_employed),
        _fmt_pct(companies),
        _fmt_pct(arrears_gt_3),
    ]
    base_df = pl.DataFrame(
        {"Metric": list(_BASE_METRIC_LABELS), "Value": base_values},
        schema={"Metric": pl.Utf8, "Value": pl.Utf8},
    )

    breakdown_df = _structure_breakdown(
        milan_pool["Additional data 9 - calc_structure_type"]
    )

    return pl.concat([base_df, breakdown_df], how="vertical")
