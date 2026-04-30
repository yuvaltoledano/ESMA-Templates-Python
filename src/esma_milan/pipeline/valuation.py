"""Stage 6: detect_aggregation_method (by_loan vs by_group).

Mirrors r_reference/R/utils.R::detect_aggregation_method (lines 454-570).
Auto-detects whether multi-loan groups in the input pool follow the
"by_loan" convention (full property value duplicated on every linked
loan row) or the "by_group" convention (property value pre-split
pro-rata across loans). Used by Stage 7 to know which aggregation rule
to apply when collapsing properties into one row per loan.

Detection logic:

  1. Filter to multi-loan groups (>1 loan).
  2. Apply Stage-1/Stage-2 valuation selection per property row to get
     a single `valuation_amount` per row.
  3. Inner-join properties to multi-loan loans on (group, loan_id).
  4. Per (group, property): require >=2 distinct linked loans AND >=2
     distinct loan balances. Properties whose linked loans share a
     balance carry no signal (the by_loan vs by_group test depends on
     differing balances exposing differing valuations).
  5. Per surviving (group, property): count distinct non-NA valuation
     amounts AND count non-NA observations. Drop properties with <2
     non-NA observations (single observation carries no signal).
  6. Classify: 1 distinct value -> "by_loan"; >1 -> "by_group".
  7. Tally. Decide via strict-greater-than 0.9 threshold; otherwise
     "ambiguous".

The Stage-1/Stage-2 valuation selection is also used by Stage 7
flattening, so it lives here as `select_valuation_amount`. Stage 7
will add a sibling `select_valuation_fields` (returning method + date
+ amount, used for the Property Valuation Type / Date columns).
"""

from __future__ import annotations

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


def select_valuation_amount(properties: pl.DataFrame) -> pl.DataFrame:
    """Add a `valuation_amount` column per the Stage-1/Stage-2 selection rule.

    Stage 1: prefer current valuation if `current_valuation_method` is
             in VALUATION_METHODS_FULL_INSPECTION (FIEI, FOEI); else
             original.
    Stage 2: if the chosen amount is null OR <= MIN_VALID_PROPERTY_VALUE,
             flip to the other valuation.

    Mirrors the inline mutate at r_reference/R/utils.R:473-481.
    Stage 7 will land a sibling helper that also emits valuation_method
    and valuation_date for the MILAN Property Valuation Type / Date
    columns.
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
        .with_columns(
            valuation_amount=pl.when(pl.col("_final_use_current"))
            .then(pl.col("current_valuation_amount"))
            .otherwise(pl.col("original_valuation_amount")),
        )
        .drop(
            "_initial_use_current",
            "_initial_amount",
            "_is_bad_value",
            "_final_use_current",
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
