"""Stage 2: filter active loans and residential-real-estate properties.

Mirrors r_reference/R/pipeline.R:228-257.

  Loans:       account_status in ACTIVE_LOAN_STATUSES
               AND current_principal_balance > 0
  Properties:  property_type is not null
               AND collateral_type not in EXCLUDED_COLLATERAL_TYPES
               AND underlying_exposure_identifier is in the union of the
                   original_* and new_* loan-ID columns from the
                   already-filtered loans table
               AND NOT (coalesce(original_valuation_amount, 0) <= 0 AND
                        coalesce(current_valuation_amount, 0) <= 0)

Hard error if zero properties remain - matches the R `stop("ERROR: No
properties remaining after filtering.")` at pipeline.R:257.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import structlog

from esma_milan.config import ACTIVE_LOAN_STATUSES, EXCLUDED_COLLATERAL_TYPES

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Stage2Output:
    """Output of Stage 2: filtered loans + filtered properties."""

    loans: pl.DataFrame
    properties: pl.DataFrame


def filter_active_loans(loans: pl.DataFrame) -> pl.DataFrame:
    """Keep only loans with active status AND positive current balance.

    Mirrors r_reference/R/pipeline.R:231-235:

        loans_filtered_step_1 <- loans_cleaned |>
          dplyr::filter(
            account_status %in% ACTIVE_LOAN_STATUSES,
            current_principal_balance > 0
          )
    """
    return loans.filter(
        pl.col("account_status").is_in(list(ACTIVE_LOAN_STATUSES))
        & (pl.col("current_principal_balance") > 0)
    )


def collect_active_loan_ids(loans: pl.DataFrame) -> set[str]:
    """Union of the two loan-ID columns from `loans`, with nulls dropped.

    The properties filter accepts a property whose
    `underlying_exposure_identifier` matches EITHER convention because
    `select_calc_loan_id` (Stage 3) hasn't yet picked which column wins.
    Mirrors r_reference/R/pipeline.R:241-245:

        active_loan_ids <- unique(c(
          loans_filtered_step_1$original_underlying_exposure_identifier,
          loans_filtered_step_1$new_underlying_exposure_identifier
        ))
        active_loan_ids <- active_loan_ids[!is.na(active_loan_ids)]
    """
    ids: set[str] = set()
    for col in (
        "original_underlying_exposure_identifier",
        "new_underlying_exposure_identifier",
    ):
        if col not in loans.columns:
            continue
        for v in loans[col].drop_nulls().to_list():
            if v is not None:
                ids.add(v)
    return ids


def filter_residential_properties(
    properties: pl.DataFrame,
    active_loan_ids: set[str],
) -> pl.DataFrame:
    """Keep only RRE properties that back an active loan.

    Mirrors r_reference/R/pipeline.R:247-253:

        properties_filtered <- properties_cleaned |>
          dplyr::filter(
            !is.na(property_type),
            !collateral_type %in% EXCLUDED_COLLATERAL_TYPES,
            underlying_exposure_identifier %in% active_loan_ids,
            !((dplyr::coalesce(original_valuation_amount, 0) <= 0) &
              (dplyr::coalesce(current_valuation_amount, 0) <= 0))
          )
    """
    excluded = list(EXCLUDED_COLLATERAL_TYPES)
    active_list = list(active_loan_ids)
    return properties.filter(
        pl.col("property_type").is_not_null()
        & pl.col("collateral_type").is_in(excluded).not_()
        & pl.col("underlying_exposure_identifier").is_in(active_list)
        & ~(
            (pl.col("original_valuation_amount").fill_null(0) <= 0)
            & (pl.col("current_valuation_amount").fill_null(0) <= 0)
        )
    )


def run_stage2(loans: pl.DataFrame, properties: pl.DataFrame) -> Stage2Output:
    """Execute Stage 2 and return the filtered loans + properties tables."""
    loans_filtered = filter_active_loans(loans)
    log.info(
        "stage2_loans_filtered",
        active_statuses=sorted(ACTIVE_LOAN_STATUSES),
        n_retained=loans_filtered.height,
    )

    active_ids = collect_active_loan_ids(loans_filtered)
    properties_filtered = filter_residential_properties(properties, active_ids)
    log.info(
        "stage2_properties_filtered",
        n_retained=properties_filtered.height,
        n_unique_collateral_ids=properties_filtered["new_collateral_identifier"]
        .drop_nulls()
        .n_unique()
        if "new_collateral_identifier" in properties_filtered.columns
        else 0,
    )

    if properties_filtered.height == 0:
        raise ValueError("ERROR: No properties remaining after filtering.")

    return Stage2Output(loans=loans_filtered, properties=properties_filtered)
