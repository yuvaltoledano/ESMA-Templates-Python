"""Stage 7: flatten_loan_collateral + Stage 8: derived fields.

This module hosts the helpers that compose the Stage-7 flattening
pipeline and the Stage-8 derived-fields work that produces the
"Combined flattened pool" sheet.

  B-1 select_valuation_fields           -> pipeline/valuation.py
  B-2 calculate_unique_property_values  -> pipeline/valuation.py
  B-3 select_main_property              -> THIS MODULE (B-3 commit)
  B-4 flatten_loan_collateral           -> THIS MODULE (B-4 commit)
  B-5 apply_derived_fields + run_stage7 -> THIS MODULE (this commit)

`flatten_loan_collateral` produces the one-row-per-loan frame with
pro-rata aggregated_property_value, main-property fields, and
final_valuation_method/date.

`apply_derived_fields` is the post-flatten step that mirrors the inline
block at r_reference/R/pipeline.R:506-592: compute calc_current_LTV /
calc_original_LTV (group-level, NaN-coalescing per R's
sum(coalesce(x,0), na.rm=TRUE) semantic), compute calc_seasoning
(per-row, days/365.25), apply the calc_ prefix rename to a fixed
allowlist of columns, and reorder columns so non-calc precede calc
(in calc_target_order). The output is byte-equal to the fixture's
"Combined flattened pool" sheet.

`run_stage7` is the stage driver that chains flatten + apply_derived_fields.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import polars as pl
import structlog

from esma_milan.config import VALUATION_METHODS_FULL_INSPECTION
from esma_milan.pipeline.valuation import (
    calculate_unique_property_values,
    select_valuation_fields,
)

log = structlog.get_logger(__name__)

_INSPECTION_METHODS: list[str] = list(VALUATION_METHODS_FULL_INSPECTION)

AggregationMethod = Literal["by_loan", "by_group"]


# Columns the R orchestrator drops from loans_enriched and
# properties_enriched at the top of flatten before re-joining group
# classification. dplyr::any_of semantics: drop if present, ignore if not.
_PROPERTY_DROP_COLS_BEFORE_FLATTEN: tuple[str, ...] = (
    "structure_type",
    "loans",
    "collaterals",
    "full_set",
    "cross_collateralized_set",
    "is_full_set",
)
_LOAN_DROP_COLS_BEFORE_FLATTEN: tuple[str, ...] = ("structure_type",)

# main_property_details drops these after the join with main_properties
# because the orchestrator output carries a single per-row valuation_amount
# computed by Stage-1/Stage-2; the source amount columns are redundant.
_MAIN_PROPERTY_DROP_COLS: tuple[str, ...] = (
    "underlying_exposure_identifier",
    "current_valuation_amount",
    "original_valuation_amount",
    "valuation_amount",
)


def select_main_property(
    properties_with_vals: pl.DataFrame,
    unique_property_values: pl.DataFrame,
) -> pl.DataFrame:
    """Pick one main property per `collateral_group_id` via the 5-key
    priority sort.

    Mirrors r_reference/R/property_flattening_function.R:114-126:

        main_properties <- properties_with_vals |>
          dplyr::left_join(
            unique_property_values,
            by = c("collateral_group_id", "calc_property_id")
          ) |>
          dplyr::group_by(collateral_group_id) |>
          dplyr::arrange(
            dplyr::desc(property_value),                           # 1
            dplyr::desc(occupancy_type == "FOWN"),                 # 2
            dplyr::desc(valuation_method %in% c("FIEI", "FOEI")),  # 3
            dplyr::desc(valuation_date),                           # 4
            calc_property_id                                       # 5 (asc)
          ) |>
          dplyr::slice(1) |>
          dplyr::ungroup() |>
          dplyr::select(collateral_group_id,
                        main_property_id = calc_property_id)

    Decision keys, in priority order:

      1. `desc(property_value)`                       — highest value wins.
      2. `desc(occupancy_type == "FOWN")`             — owner-occupied preferred.
      3. `desc(valuation_method in {FIEI, FOEI})`     — full-inspection preferred.
      4. `desc(valuation_date)`                       — most-recent valuation.
      5. `calc_property_id` ASC                       — lex tie-break (deterministic).

    NA / null handling: every descending key sets `nulls_last=True`,
    matching R's default `arrange(..., na.last = TRUE)` behaviour.
    Concretely:
      - NA `property_value`        sorts AFTER non-NA values.
      - NA `occupancy_type`        sorts AFTER both FOWN and non-FOWN.
      - NA `valuation_method`      sorts AFTER both inspection and
                                    non-inspection methods.
      - NA `valuation_date`        sorts AFTER non-NA dates.
      - NA `calc_property_id`      sorts AFTER non-NA ids (ascending tail).
        In practice all property rows have non-NA calc_property_id by
        Stage 3, so this case is defensive.

    Implementation: `sort(...)` followed by
    `unique(subset=["collateral_group_id"], keep="first", maintain_order=True)`.

    Why `unique(... keep="first")` rather than
    `group_by(...).agg(first())`:
      - `unique` is unambiguously row-level: scan, keep the first row
        encountered per subset key, drop the rest. Direct translation of
        dplyr's `arrange() %>% slice(1)`.
      - `group_by + agg(first())` semantics depend on Polars'
        `maintain_order` flag AND on whether multithreaded execution
        preserves intra-group row order. The contract is less tight in
        recent Polars versions where multithreading is the default.
      - `unique(keep="first", maintain_order=True)` always picks the
        sorted-first row regardless of Polars version.

    Args:
        properties_with_vals: properties frame post-`select_valuation_fields`.
            Must carry `collateral_group_id`, `calc_property_id`,
            `occupancy_type`, `valuation_method`, `valuation_date`.
        unique_property_values: output of `calculate_unique_property_values`.
            Provides `property_value` per (group, property) for the
            primary sort key.

    Returns:
        A frame with columns (`collateral_group_id`, `main_property_id`),
        one row per group, sorted by `collateral_group_id` ascending.
    """
    joined = properties_with_vals.join(
        unique_property_values,
        on=["collateral_group_id", "calc_property_id"],
        how="left",
    )

    sort_keys = [
        pl.col("property_value"),
        pl.col("occupancy_type") == "FOWN",
        pl.col("valuation_method").is_in(_INSPECTION_METHODS),
        pl.col("valuation_date"),
        pl.col("calc_property_id"),
    ]
    descending = [True, True, True, True, False]
    nulls_last = [True, True, True, True, True]

    sorted_df = joined.sort(
        by=sort_keys, descending=descending, nulls_last=nulls_last
    )

    main = (
        sorted_df.unique(
            subset=["collateral_group_id"], keep="first", maintain_order=True
        )
        .select(
            "collateral_group_id",
            main_property_id=pl.col("calc_property_id"),
        )
        .sort("collateral_group_id")
    )

    log.info("select_main_property", n_groups=main.height)

    return main


# ---------------------------------------------------------------------------
# B-4: flatten_loan_collateral orchestrator
# ---------------------------------------------------------------------------


def flatten_loan_collateral(
    loans_enriched: pl.DataFrame,
    properties_enriched: pl.DataFrame,
    group_classification: pl.DataFrame,
    aggregation_method: AggregationMethod = "by_loan",
) -> pl.DataFrame:
    """Collapse the loan-property graph into one row per loan with
    pro-rata-allocated `aggregated_property_value` and the main property's
    valuation fields propagated as `final_valuation_method` /
    `final_valuation_date`.

    Mirrors r_reference/R/property_flattening_function.R::flatten_loan_collateral
    (lines 73-241).

    Pipeline:
      1. Validate aggregation_method.
      2. Drop classification columns from `properties_enriched` (R's
         `select(-any_of(...))` style: drop if present, ignore if not).
      3. Apply Stage-1/Stage-2 valuation selection per property row
         (`select_valuation_fields`).
      4. Reduce per (group, property) to one `property_value`
         (`calculate_unique_property_values`, by_loan or by_group).
      5. Pick one main property per group (`select_main_property`).
      6. Hard error if any group has structure_type "unclassified".
      7. Compute group totals (sum of unique property values; sum of
         loan balances).
      8. Per loan: compute aggregated_property_value via pro-rata
         (`balance / group_loan_total * group_property_total`) OR
         equal-split fallback (`group_property_total / n_loans_in_group`)
         when group_loan_total is 0, null, OR NaN.
      9. Build main_property_details: rename `valuation_method` ->
         `final_valuation_method`, `valuation_date` -> `final_valuation_date`,
         `calc_property_id` -> `main_property_id`. Drop redundant amount
         columns. Distinct on group_id (one main property per group).
      10. Final assembly: left-join loans_enriched (minus structure_type)
          with all_aggregated_values, then with main_property_details,
          then with group_classification's structure_type.

    NaN-vs-null handling:
      R's `is.na()` catches both NA and NaN. Polars' `is_null()` catches
      only Polars null. For float columns where NaN can appear (e.g.
      `current_principal_balance`), we use `is_null() | is_nan()` in the
      equal-split trigger AND `drop_nans()` before the per-group sum to
      mirror R's `sum(..., na.rm = TRUE)` (which drops both). Pinned by
      `test_flatten_equal_split_fires_when_group_total_balance_is_nan`.

    final_valuation_* propagation:
      The MILAN cols 73/74 contract requires Property Valuation Type /
      Date / Property Value to all reference the same valuation source
      (current OR original; never split). `select_valuation_fields`
      enforces this at the per-property row level (B-1's co-vary
      invariance test). This orchestrator carries it forward by
      renaming the per-row `valuation_method` / `valuation_date` to
      `final_valuation_method` / `final_valuation_date` on the
      main_property record. The B-4 propagation test
      (`test_flatten_propagates_final_valuation_method_date_amount_
      together_after_stage2_flip`) verifies this end-to-end on a
      deliberately-asymmetric fixture.

    Pro-rata conservation:
      Across all groups, `sum(aggregated_property_value)` over the
      loans in a group equals the group's `group_property_total`. True
      for both branches: pro-rata weights sum to 1, equal-split is
      `n * (group_total / n) = group_total`. Pinned by
      `test_flatten_pro_rata_conservation_per_group` (parametrised over
      both branches).

    Returns:
        A DataFrame with one row per `calc_loan_id`. Columns:
        - all of `loans_enriched` minus `structure_type`
        - `aggregated_property_value`
        - all of `main_property_details` (occupancy_type,
          current_valuation_method, current_valuation_date,
          original_valuation_method, original_valuation_date,
          geographic_region_collateral, main_property_id,
          final_valuation_method, final_valuation_date, ...)
        - `structure_type` (from group_classification)

    Raises:
        ValueError: invalid aggregation_method, OR any group has
            structure_type == "unclassified".
    """
    if aggregation_method not in ("by_loan", "by_group"):
        raise ValueError(
            f"aggregation_method must be one of: by_group, by_loan "
            f"(got {aggregation_method!r})"
        )

    # Step 2: drop classification cols from properties so they don't
    # collide with the group_classification join later.
    properties_clean = properties_enriched.drop(
        [c for c in _PROPERTY_DROP_COLS_BEFORE_FLATTEN if c in properties_enriched.columns]
    )

    # Step 3: Stage-1/Stage-2 valuation selection.
    properties_with_vals = select_valuation_fields(properties_clean)

    # Step 4: per (group, property) reduction.
    unique_property_values = calculate_unique_property_values(
        properties_with_vals, aggregation_method
    )

    # Step 5: main property per group.
    main_properties = select_main_property(properties_with_vals, unique_property_values)

    # Step 6: unclassified hard error. R: pipeline.R:135-145.
    classification_minimal = group_classification.select(
        "collateral_group_id", "structure_type"
    )
    properties_with_structure = properties_with_vals.join(
        classification_minimal, on="collateral_group_id", how="left"
    )
    unclassified = (
        properties_with_structure.filter(pl.col("structure_type") == "unclassified")
        .select("collateral_group_id")
        .unique()
    )
    if unclassified.height > 0:
        sample = unclassified["collateral_group_id"].head(5).to_list()
        raise ValueError(
            f"ERROR: {unclassified.height} collateral groups are "
            f"'unclassified'. All groups must have a valid structure type "
            f"before flattening. Group IDs: "
            f"{', '.join(str(g) for g in sample)}..."
        )

    # Step 7: group totals.
    group_property_totals = unique_property_values.group_by(
        "collateral_group_id", maintain_order=True
    ).agg(total_property_value=pl.col("property_value").sum())

    # NaN-aware sum: drop_nans() before sum mirrors R's sum(x, na.rm=TRUE)
    # which excludes both NA and NaN. Polars' sum propagates NaN by
    # default, so without drop_nans() a single NaN balance in a group
    # would make total_loan_balance NaN -> equal-split fallback fires
    # (which is the intended behaviour, but the path differs).
    group_loan_totals = loans_enriched.group_by(
        "collateral_group_id", maintain_order=True
    ).agg(
        total_loan_balance=pl.col("current_principal_balance").drop_nans().sum()
    )

    # Warning on bad-balance groups (R: pipeline.R:166-173).
    bad_balance_count = (
        group_loan_totals.filter(
            (pl.col("total_loan_balance") == 0)
            | pl.col("total_loan_balance").is_null()
            | pl.col("total_loan_balance").is_nan()
        ).height
    )
    if bad_balance_count > 0:
        warnings.warn(
            f"WARNING: {bad_balance_count} collateral groups have zero "
            f"total balance. Property values cannot be allocated pro-rata. "
            f"Using equal allocation instead.",
            UserWarning,
            stacklevel=2,
        )

    # Step 8: per-loan allocation.
    all_aggregated_values = (
        loans_enriched.select(
            "calc_loan_id", "collateral_group_id", "current_principal_balance"
        )
        .join(group_property_totals, on="collateral_group_id", how="left")
        .join(group_loan_totals, on="collateral_group_id", how="left")
        .with_columns(n_loans_in_group=pl.len().over("collateral_group_id"))
        .with_columns(
            aggregated_property_value=pl.when(
                (pl.col("total_loan_balance") == 0)
                | pl.col("total_loan_balance").is_null()
                | pl.col("total_loan_balance").is_nan()
            )
            .then(pl.col("total_property_value") / pl.col("n_loans_in_group"))
            .otherwise(
                (pl.col("current_principal_balance") / pl.col("total_loan_balance"))
                * pl.col("total_property_value")
            ),
        )
        .select(
            "calc_loan_id",
            "collateral_group_id",
            "aggregated_property_value",
        )
    )

    # Step 9: main_property_details. Inner join properties_with_vals
    # with main_properties on (group, calc_property_id == main_property_id),
    # rename to final_*, drop redundant amount cols, distinct on group.
    main_property_details = (
        properties_with_vals.join(
            main_properties,
            left_on=["collateral_group_id", "calc_property_id"],
            right_on=["collateral_group_id", "main_property_id"],
            how="inner",
        )
        .rename(
            {
                "calc_property_id": "main_property_id",
                "valuation_method": "final_valuation_method",
                "valuation_date": "final_valuation_date",
            }
        )
    )
    main_property_details = main_property_details.drop(
        [c for c in _MAIN_PROPERTY_DROP_COLS if c in main_property_details.columns]
    ).unique(subset=["collateral_group_id"], keep="first", maintain_order=True)

    # Step 10: final assembly. Drop structure_type from loans (R does
    # this so it can come back from the classification join).
    loans_clean = loans_enriched.drop(
        [c for c in _LOAN_DROP_COLS_BEFORE_FLATTEN if c in loans_enriched.columns]
    )

    final_output = (
        loans_clean.join(
            all_aggregated_values,
            on=["calc_loan_id", "collateral_group_id"],
            how="left",
        )
        .join(main_property_details, on="collateral_group_id", how="left")
        .join(classification_minimal, on="collateral_group_id", how="left")
    )

    # Sanity check: one row per calc_loan_id (R: pipeline.R:594-598).
    n_loans = final_output["calc_loan_id"].drop_nulls().len()
    n_unique = final_output["calc_loan_id"].drop_nulls().n_unique()
    if n_loans != n_unique:
        warnings.warn(
            "Duplicate calc_loan_id found in final output",
            UserWarning,
            stacklevel=2,
        )

    log.info(
        "flatten_loan_collateral",
        n_loans=final_output.height,
        aggregation_method=aggregation_method,
    )

    return final_output


# ---------------------------------------------------------------------------
# B-5 / Stage 8: derived fields + calc_ rename + column reorder
# ---------------------------------------------------------------------------


# Column ordering R applies after the calc_ rename (r_reference/R/pipeline.R:564-577).
# Non-calc columns precede calc columns; calc columns appear in this order
# first, then any remaining calc_* in their post-rename insertion order.
# `calc_seasoning` is intentionally NOT in this list - R's pipeline.R:564-577
# leaves it in remaining_calc_cols, so it lands at the very end of the sheet
# (col 84 of 85 in the synthetic fixture).
CALC_TARGET_ORDER: tuple[str, ...] = (
    "calc_loan_id",
    "calc_borrower_id",
    "calc_collateral_group_id",
    "calc_nr_loans_in_group",
    "calc_nr_properties_in_group",
    "calc_cross_collateralized_set",
    "calc_full_set",
    "calc_aggregated_property_value",
    "calc_main_property_id",
    "calc_structure_type",
    "calc_current_LTV",
    "calc_original_LTV",
)

# Columns that get a literal `calc_` prefix added (R: pipeline.R:553-557).
_CALC_PREFIX_COLUMNS: tuple[str, ...] = (
    "main_property_id",
    "structure_type",
    "collateral_group_id",
    "cross_collateralized_set",
    "full_set",
    "aggregated_property_value",
)

# Custom rename map (R: pipeline.R:558-559).
_CALC_CUSTOM_RENAMES: dict[str, str] = {
    "loans": "calc_nr_loans_in_group",
    "collaterals": "calc_nr_properties_in_group",
}


def apply_derived_fields(flattened: pl.DataFrame) -> pl.DataFrame:
    """Compute calc_current_LTV, calc_original_LTV, calc_seasoning;
    apply calc_ rename; reorder columns.

    Mirrors r_reference/R/pipeline.R:506-592 (the post-flatten block).

    LTV computation (per group, both LTVs share the same denominator):

        denom = sum(coalesce(aggregated_property_value, 0), na.rm=TRUE)
        calc_current_LTV  = ifelse(denom > 0,
                                   sum(coalesce(current_principal_balance, 0),
                                       na.rm=TRUE) / denom,
                                   NA_real_)
        calc_original_LTV = ifelse(denom > 0,
                                   sum(coalesce(original_principal_balance, 0),
                                       na.rm=TRUE) / denom,
                                   NA_real_)

    NaN handling (per user direction): R's `sum(coalesce(x,0), na.rm=TRUE)`
    treats both null AND NaN as zero-contribution to the sum. Polars'
    `is_null()` only catches Polars null, NOT NaN; sum without explicit
    NaN handling propagates NaN. So we apply `fill_null(0).fill_nan(0)`
    before each per-group sum so:

      - one row with NaN numerator  -> fills to 0, doesn't collapse LTV
      - one row with NaN denominator -> fills to 0, doesn't collapse LTV
                                        (other rows in the group still
                                         contribute)
      - all rows NaN denominator    -> all fill to 0, sum=0, denom-gate
                                        fails -> LTV = NaN (per user's
                                        third addition: the boundary
                                        case the synthetic fixture
                                        doesn't exercise)
      - explicit zero denominator   -> denom-gate fails -> LTV = NaN
                                        (different code path, same
                                        result; pinned separately)

    Seasoning (per row):

        calc_seasoning = (pool_cutoff_date - origination_date).days / 365.25

    Per-row null behaviour (verified against r_reference/R/pipeline.R:540-546):
    if either date is null on a row, the difference is null and seasoning
    propagates as null. R does NOT warn or error on per-row null; the
    column-missing case (caught at Stage 1's validate_required_columns)
    is the only hard error. Documented and pinned by
    `test_apply_derived_fields_seasoning_null_pool_cutoff_propagates_to_null`.

    Negative seasoning (origination_date AFTER pool_cutoff_date) produces
    a negative value with no warning, matching R behaviour. Treated as
    an analyst-input data-quality issue rather than a pipeline error;
    the issue is logged in the R-repo tracker for upstream review.

    Calc rename + reorder (R: pipeline.R:553-592):

      1. main_property_id, structure_type, collateral_group_id,
         cross_collateralized_set, full_set, aggregated_property_value
         get a literal `calc_` prefix.
      2. loans -> calc_nr_loans_in_group; collaterals ->
         calc_nr_properties_in_group.
      3. Reorder: non-calc cols first (in their existing order), then
         calc cols in CALC_TARGET_ORDER, then any remaining calc_* (in
         post-rename insertion order). For the synthetic fixture this
         puts calc_seasoning last (col 84 of 85).

    Args:
        flattened: output of `flatten_loan_collateral`.

    Returns:
        The combined flattened pool frame in fixture-byte-equal column
        order, ready for the workbook writer.
    """
    df = flattened

    # -- LTVs: per-group sums with NaN-as-0 + null-as-0 + denom-gate.
    df = df.with_columns(
        _agg_sum=(
            pl.col("aggregated_property_value")
            .fill_null(0.0)
            .fill_nan(0.0)
            .sum()
            .over("collateral_group_id")
        ),
        _cpb_sum=(
            pl.col("current_principal_balance")
            .fill_null(0.0)
            .fill_nan(0.0)
            .sum()
            .over("collateral_group_id")
        ),
        _opb_sum=(
            pl.col("original_principal_balance")
            .fill_null(0.0)
            .fill_nan(0.0)
            .sum()
            .over("collateral_group_id")
        ),
    ).with_columns(
        calc_current_LTV=pl.when(pl.col("_agg_sum") > 0)
        .then(pl.col("_cpb_sum") / pl.col("_agg_sum"))
        .otherwise(None),
        calc_original_LTV=pl.when(pl.col("_agg_sum") > 0)
        .then(pl.col("_opb_sum") / pl.col("_agg_sum"))
        .otherwise(None),
    ).drop("_agg_sum", "_cpb_sum", "_opb_sum")

    # -- Seasoning: per row, (pool_cutoff_date - origination_date).days / 365.25.
    df = df.with_columns(
        calc_seasoning=(
            (pl.col("pool_cutoff_date") - pl.col("origination_date"))
            .dt.total_days()
            .cast(pl.Float64)
            / 365.25
        )
    )

    # -- Calc renames.
    rename_map: dict[str, str] = {
        col: f"calc_{col}" for col in _CALC_PREFIX_COLUMNS if col in df.columns
    }
    rename_map.update(
        {old: new for old, new in _CALC_CUSTOM_RENAMES.items() if old in df.columns}
    )
    if rename_map:
        df = df.rename(rename_map)

    # -- Reorder.
    all_cols = df.columns
    calc_cols = [c for c in all_cols if c.startswith("calc_")]
    other_cols = [c for c in all_cols if not c.startswith("calc_")]
    ordered_calc = [c for c in CALC_TARGET_ORDER if c in calc_cols]
    remaining_calc = [c for c in calc_cols if c not in ordered_calc]
    final_calc_cols = ordered_calc + remaining_calc

    return df.select(*other_cols, *final_calc_cols)


# ---------------------------------------------------------------------------
# Stage 7 driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage7Output:
    """Output of Stage 7 + Stage 8 combined.

    `combined_flattened` is the frame written to the workbook's
    "Combined flattened pool" sheet, post-flatten, with derived fields
    applied and column order matching the R fixture exactly.
    """

    combined_flattened: pl.DataFrame


def run_stage7(
    loans_enriched: pl.DataFrame,
    properties_enriched: pl.DataFrame,
    group_classification: pl.DataFrame,
    *,
    aggregation_method: AggregationMethod = "by_loan",
) -> Stage7Output:
    """Stage 7 driver. Composes flatten_loan_collateral with
    apply_derived_fields to produce the Combined flattened pool frame.

    Naming consistency with run_stage1..run_stage6. The single Stage7Output
    covers what R's pipeline.R does inline at lines 491-558 (call
    flatten_loan_collateral, then compute LTVs/seasoning, then apply the
    calc_ rename + reorder)."""
    flattened = flatten_loan_collateral(
        loans_enriched,
        properties_enriched,
        group_classification,
        aggregation_method=aggregation_method,
    )
    combined = apply_derived_fields(flattened)
    return Stage7Output(combined_flattened=combined)
