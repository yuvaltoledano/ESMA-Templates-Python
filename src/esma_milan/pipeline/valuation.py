"""Stage 6: detect_aggregation_method (by_loan vs by_group).
Stage 7 (B-2): calculate_unique_property_values.

Stage 6 mirrors r_reference/R/utils.R::detect_aggregation_method and
auto-detects whether multi-loan groups in the input pool follow the
"by_loan" convention (full property value duplicated on every linked
loan row) or the "by_group" convention (property value pre-split
pro-rata across loans). Used by Stage 7 to know which aggregation rule
to apply when collapsing properties into one row per loan.

Stage 7 B-2 mirrors `calculate_unique_property_values()` in
r_reference/R/property_flattening_function.R:9-54 and reduces a
properties frame to one row per (collateral_group_id, calc_property_id)
with a single `property_value`, applying the chosen reduction rule.

The Stage-1/Stage-2 valuation selection is shared between detection
(amount-only) and flattening (full method/date/amount), and lives in
this module too: `select_valuation_amount` (B-1 / Stage 6) and
`select_valuation_fields` (B-1 / Stage 7).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import polars as pl
import structlog

from esma_milan.config import (
    MIN_VALID_PROPERTY_VALUE,
    VALUATION_METHODS_FULL_INSPECTION,
)

log = structlog.get_logger(__name__)

_INSPECTION_METHODS: list[str] = list(VALUATION_METHODS_FULL_INSPECTION)

DetectedMethod = Literal["by_loan", "by_group", "ambiguous"]

# Strict greater-than threshold (matches R's `n_by_loan / total > 0.9`).
# 9/10 = 0.9 -> ambiguous (not strictly greater); 10/10 -> by_loan.
DECISION_THRESHOLD: float = 0.9


@dataclass(frozen=True)
class Stage6Output:
    """Output of Stage 6.

    `detected_method` is one of "by_loan" / "by_group" / "ambiguous".
    `message` mirrors the human-readable string R returns alongside.
    """

    detected_method: DetectedMethod
    message: str


def _add_selection_columns(properties: pl.DataFrame) -> pl.DataFrame:
    """Add the four internal Stage-1/Stage-2 working columns to `properties`.

    Encapsulates the shared logic between `select_valuation_amount` and
    `select_valuation_fields` so neither helper duplicates the Stage-1
    pick / Stage-2 flip rule. Returns a new frame with:

      _initial_use_current : bool, True if current_valuation_method
                             is in VALUATION_METHODS_FULL_INSPECTION.
      _initial_amount      : the Stage-1 chosen amount.
      _is_bad_value        : True if _initial_amount is null or
                             <= MIN_VALID_PROPERTY_VALUE (= 10).
      _final_use_current   : the Stage-2 final pick after flipping if
                             the initial was bad.
    """
    return (
        properties.with_columns(
            _initial_use_current=pl.col("current_valuation_method").is_in(
                _INSPECTION_METHODS
            ),
        )
        .with_columns(
            _initial_amount=pl.when(pl.col("_initial_use_current"))
            .then(pl.col("current_valuation_amount"))
            .otherwise(pl.col("original_valuation_amount")),
        )
        .with_columns(
            _is_bad_value=(
                pl.col("_initial_amount").is_null()
                | (pl.col("_initial_amount") <= MIN_VALID_PROPERTY_VALUE)
            ),
        )
        .with_columns(
            _final_use_current=pl.when(pl.col("_is_bad_value"))
            .then(~pl.col("_initial_use_current"))
            .otherwise(pl.col("_initial_use_current")),
        )
    )


def _drop_selection_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop the four internal columns added by `_add_selection_columns`."""
    return df.drop(
        "_initial_use_current",
        "_initial_amount",
        "_is_bad_value",
        "_final_use_current",
    )


def select_valuation_amount(properties: pl.DataFrame) -> pl.DataFrame:
    """Add a `valuation_amount` column per the Stage-1/Stage-2 selection rule.

    Stage 1: prefer current valuation if `current_valuation_method` is
             in VALUATION_METHODS_FULL_INSPECTION (FIEI, FOEI); else
             original.
    Stage 2: if the chosen amount is null OR <= MIN_VALID_PROPERTY_VALUE,
             flip to the other valuation.

    Mirrors the inline mutate at r_reference/R/utils.R:473-481.
    Used by `detect_aggregation_method` (Stage 6) where only the
    amount matters; Stage 7's `flatten_loan_collateral` uses the
    sibling `select_valuation_fields` to also surface
    valuation_method and valuation_date for the MILAN Property
    Valuation Type / Date fields.
    """
    return _drop_selection_columns(
        _add_selection_columns(properties).with_columns(
            valuation_amount=pl.when(pl.col("_final_use_current"))
            .then(pl.col("current_valuation_amount"))
            .otherwise(pl.col("original_valuation_amount")),
        )
    )


def select_valuation_fields(properties: pl.DataFrame) -> pl.DataFrame:
    """Add `valuation_method`, `valuation_date`, `valuation_amount`.

    Full version of the Stage-1/Stage-2 selection rule used by Stage 7
    flattening. Mirrors `get_valuation_fields()` in
    r_reference/R/property_flattening_function.R:84-101:

        df |>
          mutate(
            initial_use_current = current_valuation_method %in%
                                  VALUATION_METHODS_FULL_INSPECTION,
            initial_amount      = if_else(initial_use_current,
                                          current_valuation_amount,
                                          original_valuation_amount),
            is_bad_value        = is.na(initial_amount) |
                                  initial_amount <= MIN_VALID_PROPERTY_VALUE,
            final_use_current   = if_else(is_bad_value,
                                          !initial_use_current,
                                          initial_use_current),
            valuation_method    = if_else(final_use_current,
                                          current_valuation_method,
                                          original_valuation_method),
            valuation_date      = if_else(final_use_current,
                                          current_valuation_date,
                                          original_valuation_date),
            valuation_amount    = if_else(final_use_current,
                                          current_valuation_amount,
                                          original_valuation_amount),
          ) |>
          select(-any_of(c("initial_use_current", "initial_amount",
                           "is_bad_value", "final_use_current")))

    The crucial property the MILAN Property Valuation Type / Date
    contract relies on: when Stage 2 flips, ALL THREE of method, date,
    and amount come from the same source (current OR original). They
    never split. This is what `final_valuation_method` /
    `final_valuation_date` propagation in Stage 7 carries forward to
    MILAN cols 73/74. Pinned by the asymmetric-flip unit test in
    `tests/unit/test_valuation.py`.

    Input columns required:
      current_valuation_method, current_valuation_date,
      current_valuation_amount, original_valuation_method,
      original_valuation_date, original_valuation_amount.

    Output: input frame + `valuation_method`, `valuation_date`,
    `valuation_amount` appended.
    """
    return _drop_selection_columns(
        _add_selection_columns(properties).with_columns(
            valuation_method=pl.when(pl.col("_final_use_current"))
            .then(pl.col("current_valuation_method"))
            .otherwise(pl.col("original_valuation_method")),
            valuation_date=pl.when(pl.col("_final_use_current"))
            .then(pl.col("current_valuation_date"))
            .otherwise(pl.col("original_valuation_date")),
            valuation_amount=pl.when(pl.col("_final_use_current"))
            .then(pl.col("current_valuation_amount"))
            .otherwise(pl.col("original_valuation_amount")),
        )
    )


def detect_aggregation_method(
    loans_enriched: pl.DataFrame,
    properties_enriched: pl.DataFrame,
) -> Stage6Output:
    """Auto-detect the aggregation convention used in the input pool.

    See module docstring for the full algorithm. Returns a Stage6Output
    with detected_method ∈ {"by_loan", "by_group", "ambiguous"} and a
    human-readable message string.
    """
    # Step 1: filter to multi-loan groups (>1 loan).
    multi_loan_groups = (
        loans_enriched.group_by("collateral_group_id")
        .agg(pl.len().alias("_n_loans"))
        .filter(pl.col("_n_loans") > 1)
        .select("collateral_group_id")
    )

    if multi_loan_groups.height == 0:
        return Stage6Output(
            detected_method="ambiguous",
            message="No multi-loan groups found. Cannot infer aggregation method.",
        )

    multi_loan_loans = (
        loans_enriched.join(multi_loan_groups, on="collateral_group_id", how="inner")
        .select("collateral_group_id", "calc_loan_id", "current_principal_balance")
    )

    # Step 2: apply Stage-1/Stage-2 valuation selection per property row.
    properties_with_vals = select_valuation_amount(properties_enriched)

    # Step 3: inner-join properties to multi-loan loans on (group, loan_id).
    # Polars renames the right-side join key with a `_right` suffix; the
    # join key from the left side stays as `underlying_exposure_identifier`.
    analysis_set = (
        properties_with_vals.join(
            multi_loan_loans,
            left_on=["collateral_group_id", "underlying_exposure_identifier"],
            right_on=["collateral_group_id", "calc_loan_id"],
            how="inner",
        )
        .select(
            "collateral_group_id",
            "calc_property_id",
            pl.col("underlying_exposure_identifier").alias("calc_loan_id"),
            "current_principal_balance",
            "valuation_amount",
        )
    )

    if analysis_set.height == 0:
        return Stage6Output(
            detected_method="ambiguous",
            message="No property-loan matches found for analysis.",
        )

    # Step 4: filter to candidate (group, property) with >=2 distinct
    # linked loans AND >=2 distinct loan balances. n_unique() over a
    # window includes nulls (matches R's dplyr::n_distinct default).
    test_candidates = (
        analysis_set.with_columns(
            _n_distinct_loans=pl.col("calc_loan_id")
            .n_unique()
            .over(["collateral_group_id", "calc_property_id"]),
            _balance_variance=pl.col("current_principal_balance")
            .n_unique()
            .over(["collateral_group_id", "calc_property_id"]),
        )
        .filter(
            (pl.col("_n_distinct_loans") > 1) & (pl.col("_balance_variance") > 1)
        )
        .drop("_n_distinct_loans", "_balance_variance")
    )

    if test_candidates.height == 0:
        return Stage6Output(
            detected_method="ambiguous",
            message=(
                "No suitable test candidates (properties linked to loans "
                "with differing balances) found."
            ),
        )

    # Step 5: per (group, property) count distinct non-NA valuations
    # and total non-NA observations. Drop properties with <2 non-NA
    # observations (single observation carries no signal). Mirrors
    # r_reference/R/utils.R:524-528.
    evaluation = (
        test_candidates.group_by(["collateral_group_id", "calc_property_id"])
        .agg(
            values_distinct_count=pl.col("valuation_amount")
            .drop_nulls()
            .n_unique(),
            non_na_observations=pl.col("valuation_amount").is_not_null().sum(),
        )
        .filter(pl.col("non_na_observations") >= 2)
        .with_columns(
            implied_method=pl.when(pl.col("values_distinct_count") == 1)
            .then(pl.lit("by_loan"))
            .otherwise(pl.lit("by_group"))
        )
    )

    n_by_loan = evaluation.filter(pl.col("implied_method") == "by_loan").height
    n_by_group = evaluation.filter(pl.col("implied_method") == "by_group").height
    total_evidence = n_by_loan + n_by_group

    log.info(
        "detect_aggregation_method",
        n_by_loan=n_by_loan,
        n_by_group=n_by_group,
        total_evidence=total_evidence,
    )

    if total_evidence == 0:
        return Stage6Output(
            detected_method="ambiguous",
            message="No evidence found to infer aggregation method.",
        )

    # Strict > 0.9 threshold (matches R's `n_by_loan / total > 0.9`).
    if n_by_loan / total_evidence > DECISION_THRESHOLD:
        return Stage6Output(
            detected_method="by_loan",
            message=(
                f"Strong evidence ({n_by_loan}/{total_evidence} "
                f"properties) suggests 'by_loan' duplication."
            ),
        )
    if n_by_group / total_evidence > DECISION_THRESHOLD:
        return Stage6Output(
            detected_method="by_group",
            message=(
                f"Strong evidence ({n_by_group}/{total_evidence} "
                f"properties) suggests 'by_group' splitting."
            ),
        )
    return Stage6Output(
        detected_method="ambiguous",
        message=(
            f"Mixed signals: {n_by_loan} properties suggest 'by_loan', "
            f"{n_by_group} suggest 'by_group'."
        ),
    )


def run_stage6(
    loans_enriched: pl.DataFrame,
    properties_enriched: pl.DataFrame,
) -> Stage6Output:
    """Stage 6 driver. Wraps detect_aggregation_method for naming
    consistency with run_stage1..run_stage5."""
    return detect_aggregation_method(loans_enriched, properties_enriched)


# ---------------------------------------------------------------------------
# Stage 7 B-2: calculate_unique_property_values
# ---------------------------------------------------------------------------


AggregationMethod = Literal["by_loan", "by_group"]


def calculate_unique_property_values(
    properties_with_vals: pl.DataFrame,
    aggregation_method: AggregationMethod,
) -> pl.DataFrame:
    """Reduce to one (`collateral_group_id`, `calc_property_id`,
    `property_value`) row per unique property, applying the chosen
    aggregation rule.

    Mirrors `calculate_unique_property_values()` in
    r_reference/R/property_flattening_function.R:9-54:

        if aggregation_method == "by_group":
            sum(valuation_amount, na.rm = TRUE) per (group, property)
        elif aggregation_method == "by_loan":
            first(valuation_amount[!is.na(valuation_amount)],
                  default = NA_real_) per (group, property)

    Determinism note for the by_loan branch:
      "First non-null" is well-defined only if the row order within
      each (group, property) bucket is stable. Polars' group_by does
      NOT guarantee within-group row ordering by default; a future
      version or parallel-execution flag could shuffle silently. The
      implementation here pins the order by:

        1. Adding a sequential `_row_idx` column reflecting the input
           frame's row order at call time.
        2. Sorting `valuation_amount` by `_row_idx` inside the per-group
           agg before applying `drop_nulls().first()`.

      Result: "first non-null" is a pure function of the input frame's
      row order at the moment of the call. The upstream pipeline
      produces input rows in a deterministic order (Stage 1's CSV read
      preserves source order; subsequent stages' joins are stable), so
      the helper's output is deterministic across runs.

      `tests/unit/test_valuation.py::test_calculate_unique_by_loan_
      result_invariant_to_row_shuffling_when_group_has_one_non_null`
      pins the determinism property: when each group has at most one
      non-null value, the result is invariant to input-row permutation
      (because there's no ambiguity about which value "first" picks).
      For groups with >=2 disagreeing non-null values, the inconsistency
      warning fires AND the chosen value is implementation-defined
      (input-order-defined); the warning surfaces the data quality
      issue rather than papering over it.

    The by_loan inconsistency warning fires when a (group, property)
    pair has >=2 distinct non-NA valuation amounts. Mirrors the
    `n_distinct(valuation_amount, na.rm = TRUE) > 1` check in R.
    Properties with one valid value and (NA, NA, ..., NA) siblings
    are NOT flagged — the analyst-walkthrough fixture's (500000, NA, NA)
    pattern is the canonical example.

    Args:
        properties_with_vals: a DataFrame with at least
            `collateral_group_id`, `calc_property_id`, and
            `valuation_amount` columns. Typically the output of
            `select_valuation_fields()`.
        aggregation_method: one of "by_loan" or "by_group".

    Returns:
        A DataFrame with columns
        (`collateral_group_id`, `calc_property_id`, `property_value`),
        sorted by (collateral_group_id ASC, calc_property_id ASC) so
        the output row order is a pure function of the input set,
        independent of any group_by emission-order nondeterminism.

    Raises:
        ValueError: if aggregation_method is not "by_loan" or "by_group".
    """
    if aggregation_method not in ("by_loan", "by_group"):
        raise ValueError(
            f"aggregation_method must be one of: by_loan, by_group "
            f"(got {aggregation_method!r})"
        )

    if aggregation_method == "by_group":
        return (
            properties_with_vals.group_by(
                ["collateral_group_id", "calc_property_id"], maintain_order=True
            )
            .agg(property_value=pl.col("valuation_amount").sum())
            .sort(["collateral_group_id", "calc_property_id"])
        )

    # by_loan branch.

    # Inconsistency warning: per (group, property), count distinct
    # NON-NULL valuations. >=2 distinct -> warn. Mirrors R's
    # `n_distinct(valuation_amount, na.rm = TRUE) > 1` filter.
    inconsistent = (
        properties_with_vals.group_by(
            ["collateral_group_id", "calc_property_id"], maintain_order=True
        )
        .agg(n_distinct=pl.col("valuation_amount").drop_nulls().n_unique())
        .filter(pl.col("n_distinct") > 1)
    )
    if inconsistent.height > 0:
        sample_ids = (
            inconsistent.sort(["collateral_group_id", "calc_property_id"])[
                "calc_property_id"
            ]
            .head(5)
            .to_list()
        )
        warnings.warn(
            f"{inconsistent.height} properties have inconsistent valuation "
            f"amounts across linked loans. Using first non-missing value. "
            f"Property IDs: {', '.join(str(p) for p in sample_ids)}...",
            UserWarning,
            stacklevel=2,
        )

    # First non-null by input row order. _row_idx pins the order at
    # call time; sort_by inside agg makes "first" deterministic
    # regardless of Polars' internal group_by ordering.
    df = properties_with_vals.with_row_index("_row_idx")
    return (
        df.group_by(
            ["collateral_group_id", "calc_property_id"], maintain_order=True
        )
        .agg(
            property_value=pl.col("valuation_amount")
            .sort_by("_row_idx")
            .drop_nulls()
            .first()
        )
        .sort(["collateral_group_id", "calc_property_id"])
    )
