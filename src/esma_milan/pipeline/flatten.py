"""Stage 7: flatten_loan_collateral.

This module hosts the helpers that compose the Stage-7 flattening
pipeline. Helpers land one-per-commit in option-B granularity (see the
PR description for the full plan):

  B-1 select_valuation_fields           -> pipeline/valuation.py
  B-2 calculate_unique_property_values  -> pipeline/valuation.py
  B-3 select_main_property              -> THIS MODULE (this commit)
  B-4 flatten_loan_collateral           -> THIS MODULE (next commit)
  B-5 derived fields + Sheet-9 writer   -> runner / write_workbook

`select_main_property` mirrors the main-property selection block in
r_reference/R/property_flattening_function.R:114-126, which is one
slice of the larger flatten function in R. We pull it out here as a
named public helper because the 5-key sort with the lex tie-break is
the parity-fragile bit of Stage 7 - any sort-stability or
NA-sort-order drift between Polars and dplyr surfaces here, and an
isolated commit makes bisection trivial if a downstream MILAN field
later surfaces a "wrong main property picked" symptom.
"""

from __future__ import annotations

import polars as pl
import structlog

from esma_milan.config import VALUATION_METHODS_FULL_INSPECTION

log = structlog.get_logger(__name__)

_INSPECTION_METHODS: list[str] = list(VALUATION_METHODS_FULL_INSPECTION)


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
