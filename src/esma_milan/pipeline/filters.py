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

The conditions are defined once, as named Polars expressions, one per
reason a row is dropped (`loan_drop_reasons`, `collateral_drop_reasons`).
Each reason is total (true or false on every row, never null) and a row is
kept exactly when no reason is true. The two filter functions are built
from them, and `run_stage2` returns them evaluated per input row together
with the positions of the rows it kept, so a caller can see which records
were dropped and why without re-deriving the rules. The expressions refer
only to the cleaned column names, so they can be applied to any frame that
carries those columns.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import polars as pl
import structlog

from esma_milan.config import ACTIVE_LOAN_STATUSES, EXCLUDED_COLLATERAL_TYPES
from esma_milan.errors import PipelineDataError

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Stage2Output:
    """Output of Stage 2: filtered loans + filtered properties.

    The four optional fields describe the decision per input record.
    Stage 2's input is the Stage 1 output, so every position here is a
    0-based row position in the Stage 1 frame, which is the record's
    position in the uploaded file (see `PipelineResult.loan_record_index`
    for what that means).
    """

    loans: pl.DataFrame
    properties: pl.DataFrame

    loan_rows: pl.Series | None = None
    """Positions in the Stage 1 loans frame of the loans kept, ascending
    (UInt32). `loans` row i is Stage 1 row `loan_rows[i]`."""

    property_rows: pl.Series | None = None
    """Positions in the Stage 1 properties frame of the collateral rows
    kept, ascending (UInt32)."""

    loan_drop_reasons: pl.DataFrame | None = None
    """One row per Stage 1 loan, in order, and one Boolean column per
    reason in `loan_drop_reasons()`. A loan is kept exactly when its row
    is all false; a dropped loan can have more than one reason true."""

    property_drop_reasons: pl.DataFrame | None = None
    """One row per Stage 1 collateral row, in order, and one Boolean
    column per reason in `collateral_drop_reasons()`. The
    `no_active_loan` reason is evaluated against the loans Stage 2 kept."""


def loan_drop_reasons() -> dict[str, pl.Expr]:
    """Why Stage 2 drops a loan: reason name -> Boolean expression, in
    reporting order.

      status_not_active     account_status is not one of
                            ACTIVE_LOAN_STATUSES, or is missing.
      balance_not_positive  current_principal_balance is missing, zero or
                            negative. A NaN balance is not caught: Polars
                            orders NaN above every number, so `NaN > 0` is
                            true and the loan is kept, as it was before
                            these expressions existed.
    """
    return {
        "status_not_active": ~pl.col("account_status")
        .is_in(list(ACTIVE_LOAN_STATUSES))
        .fill_null(False),
        "balance_not_positive": ~(pl.col("current_principal_balance") > 0).fill_null(
            False
        ),
    }


def collateral_drop_reasons(active_loan_ids: Collection[str]) -> dict[str, pl.Expr]:
    """Why Stage 2 drops a collateral row: reason name -> Boolean
    expression, in reporting order.

      property_type_missing     property_type is missing.
      collateral_type_missing   collateral_type is missing. A missing
                                collateral type drops the row. This is what
                                this module did before the predicate
                                existed (`is_in(...).not_()` is null for a
                                null value and `filter` drops nulls). R
                                keeps the row
                                (r_reference/R/pipeline.R:250,
                                `!collateral_type %in% ...` is TRUE for
                                NA). Reported as it is; not corrected here.
      collateral_type_excluded  collateral_type is one of
                                EXCLUDED_COLLATERAL_TYPES.
      no_active_loan            underlying_exposure_identifier is missing
                                or is not in `active_loan_ids` (either
                                identifier of a loan Stage 2 kept).
      no_positive_valuation     original_valuation_amount and
                                current_valuation_amount are both missing,
                                zero or negative. A NaN amount counts as
                                positive (`NaN <= 0` is false in Polars),
                                as it did before.
    """
    return {
        "property_type_missing": pl.col("property_type").is_null(),
        "collateral_type_missing": pl.col("collateral_type").is_null(),
        "collateral_type_excluded": pl.col("collateral_type")
        .is_in(list(EXCLUDED_COLLATERAL_TYPES))
        .fill_null(False),
        "no_active_loan": ~pl.col("underlying_exposure_identifier")
        .is_in(list(active_loan_ids))
        .fill_null(False),
        "no_positive_valuation": (pl.col("original_valuation_amount").fill_null(0) <= 0)
        & (pl.col("current_valuation_amount").fill_null(0) <= 0),
    }


def _keep(reasons: Mapping[str, pl.Expr]) -> pl.Expr:
    """True for a row that no reason drops."""
    return ~pl.any_horizontal(list(reasons.values()))


def _apply_reasons(
    frame: pl.DataFrame, reasons: Mapping[str, pl.Expr]
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame]:
    """Evaluate `reasons` on `frame` once. Returns the rows kept, their
    positions in `frame`, and the reasons frame (one row per input row)."""
    evaluated = frame.select(**reasons)
    keep = evaluated.select(_keep({name: pl.col(name) for name in reasons})).to_series()
    return frame.filter(keep), keep.arg_true(), evaluated


def filter_active_loans(loans: pl.DataFrame) -> pl.DataFrame:
    """Keep only loans with active status AND positive current balance.

    Mirrors r_reference/R/pipeline.R:231-235:

        loans_filtered_step_1 <- loans_cleaned |>
          dplyr::filter(
            account_status %in% ACTIVE_LOAN_STATUSES,
            current_principal_balance > 0
          )
    """
    return loans.filter(_keep(loan_drop_reasons()))


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
    return properties.filter(_keep(collateral_drop_reasons(active_loan_ids)))


def run_stage2(loans: pl.DataFrame, properties: pl.DataFrame) -> Stage2Output:
    """Execute Stage 2 and return the filtered loans + properties tables."""
    loans_filtered, loan_rows, loan_reasons = _apply_reasons(
        loans, loan_drop_reasons()
    )
    log.info(
        "stage2_loans_filtered",
        active_statuses=sorted(ACTIVE_LOAN_STATUSES),
        n_retained=loans_filtered.height,
    )

    active_ids = collect_active_loan_ids(loans_filtered)
    properties_filtered, property_rows, property_reasons = _apply_reasons(
        properties, collateral_drop_reasons(active_ids)
    )
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
        raise PipelineDataError(
            "ERROR: No properties remaining after filtering.",
            stage="stage2",
            reason="no_properties_after_filter",
            context={
                "n_loans": loans.height,
                "n_loans_kept": loans_filtered.height,
                "n_collateral": properties.height,
                "loan_reason_counts": _reason_counts(loan_reasons),
                "collateral_reason_counts": _reason_counts(property_reasons),
            },
        )

    return Stage2Output(
        loans=loans_filtered,
        properties=properties_filtered,
        loan_rows=loan_rows,
        property_rows=property_rows,
        loan_drop_reasons=loan_reasons,
        property_drop_reasons=property_reasons,
    )


def _reason_counts(reasons: pl.DataFrame) -> dict[str, int]:
    """Number of rows each reason is true for."""
    return {name: int(reasons[name].sum()) for name in reasons.columns}
