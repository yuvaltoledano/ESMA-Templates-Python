"""Stage 8.5: mapping-tables composer for Sheets 1-4.

Mirrors r_reference/R/pipeline.R:333-391 (the four pivot_wider blocks)
plus :452-465 (the classification-tail extension on Loans to properties).

The four sheets share the same recipe: take a (key, value) pair from the
loans-properties join, distinct it, then per-key pivot the values into
suffix-numbered columns. Only Loans to properties gets a classification
tail (collateral_group_id + cross_collateralized_set + full_set +
structure_type) appended after the pivot.

R-repo finding #6 (logged in docs/r_repo_findings.md): R wraps the
classification-tail join in tryCatch(...) at pipeline.R:458-464, silently
swallowing errors. The Python port joins unconditionally so a failure
surfaces as a real exception rather than as a sheet emitted without
classification columns.

R-repo finding #7 (logged in docs/r_repo_findings.md): R renames
calc_loan_id -> loan_id at the top of this block but keeps calc_loan_id
elsewhere in the pipeline; the join at line 460 has to bridge the two
naming conventions. Python keeps R's renaming for parity.
"""

from __future__ import annotations

import warnings

import polars as pl
import structlog

from esma_milan.config import MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD

log = structlog.get_logger(__name__)


def compose_mapping_tables(
    stage3_loans: pl.DataFrame,
    stage3_properties: pl.DataFrame,
    loans_enriched: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Build the four mapping tables written to Sheets 1-4.

    Args:
        stage3_loans: post-Stage-3 loans frame, must carry calc_loan_id
            and calc_borrower_id.
        stage3_properties: post-Stage-3 properties frame, must carry
            underlying_exposure_identifier and calc_property_id.
        loans_enriched: post-Stage-5 loans frame, must carry
            collateral_group_id, cross_collateralized_set, full_set,
            structure_type. Used only for the Loans-to-properties
            classification tail.

    Returns:
        A dict mapping each of the four sheet names
        ("Loans to properties", "Properties to loans",
        "Borrowers to loans", "Borrowers to properties") to its
        ready-to-write Polars DataFrame.

    Warns:
        UserWarning: if any loan has > MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD
            properties. Mirrors r_reference/R/pipeline.R:357-359 with
            the same message text.
    """
    loan_file_ids = stage3_loans.select(
        loan_id=pl.col("calc_loan_id"),
        borrower_id=pl.col("calc_borrower_id"),
    )
    properties_file_ids = stage3_properties.select(
        loan_id=pl.col("underlying_exposure_identifier"),
        property_id=pl.col("calc_property_id"),
    )
    all_ids = (
        loan_file_ids.join(properties_file_ids, on="loan_id", how="left")
        .unique(maintain_order=True)
    )

    # MAX_PROPERTIES_PER_LOAN warning. Mirrors pipeline.R:348-359 message
    # text exactly: "Some loans have <N> properties (threshold: 50)."
    if all_ids.height > 0:
        max_n = (
            all_ids.group_by("loan_id")
            .len()
            .select(pl.col("len").max())
            .item()
        )
        if max_n is not None and max_n > MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD:
            warnings.warn(
                f"Some loans have {max_n} properties "
                f"(threshold: {MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD}).",
                UserWarning,
                stacklevel=2,
            )

    loans_to_properties = _pivot_with_suffix(
        all_ids, index_col="loan_id", value_col="property_id", name_prefix="property_id"
    )
    properties_to_loans = _pivot_with_suffix(
        all_ids, index_col="property_id", value_col="loan_id", name_prefix="loan_id"
    )
    borrowers_to_loans = _pivot_with_suffix(
        all_ids, index_col="borrower_id", value_col="loan_id", name_prefix="loan_id"
    )
    borrowers_to_properties = _pivot_with_suffix(
        all_ids,
        index_col="borrower_id",
        value_col="property_id",
        name_prefix="property_id",
    )

    # Classification tail on loans_to_properties only. Mirrors
    # pipeline.R:452-465. R wraps this in tryCatch silently swallowing
    # errors (logged as R-repo finding #6); the Python port joins
    # unconditionally so a failure surfaces.
    classification_tail = (
        loans_enriched.select(
            "calc_loan_id",
            "collateral_group_id",
            "cross_collateralized_set",
            "full_set",
            "structure_type",
        )
        .unique(maintain_order=True)
    )
    loans_to_properties = loans_to_properties.join(
        classification_tail,
        left_on="loan_id",
        right_on="calc_loan_id",
        how="left",
    )

    log.info(
        "compose_mapping_tables",
        loans_to_properties_rows=loans_to_properties.height,
        properties_to_loans_rows=properties_to_loans.height,
        borrowers_to_loans_rows=borrowers_to_loans.height,
        borrowers_to_properties_rows=borrowers_to_properties.height,
    )

    return {
        "Loans to properties": loans_to_properties,
        "Properties to loans": properties_to_loans,
        "Borrowers to loans": borrowers_to_loans,
        "Borrowers to properties": borrowers_to_properties,
    }


def _pivot_with_suffix(
    df: pl.DataFrame,
    *,
    index_col: str,
    value_col: str,
    name_prefix: str,
) -> pl.DataFrame:
    """Distinct on (index_col, value_col), then pivot per index group with
    suffix-numbered columns name_prefix_1, name_prefix_2, ...

    Output column order is (index_col, name_prefix_1, name_prefix_2, ...)
    in numeric ascending order (so name_prefix_10 doesn't sort lexically
    between _1 and _2 if the input ever has >=10 values per group).

    Row order: index keys appear in order of first occurrence in `df`.
    Group emission via group_by(maintain_order=True) preserves first-
    occurrence; pivot preserves that order.
    """
    distinct = df.select(index_col, value_col).unique(maintain_order=True)

    # Per-group row number (1-indexed) for the suffix slot. Count over the
    # INDEX column (which is non-null by construction) rather than the
    # VALUE column, because Polars' cum_count() counts only non-null
    # values - so a defensive (loan_id, null) row would receive _n=0 and
    # land in a ghost property_id_0 column. Counting the index column
    # gives 1, 2, 3, ... per group regardless of value-column nulls.
    distinct_with_n = distinct.with_columns(
        _n=pl.col(index_col).cum_count().over(index_col).cast(pl.Int64)
    )

    pivoted = distinct_with_n.pivot(
        on="_n",
        index=index_col,
        values=value_col,
    )

    # Polars pivot stringifies the int `on` values when generating column
    # names. Sort suffix columns by integer value (so _10 doesn't sort
    # between _1 and _2), then rename.
    suffix_cols = sorted(
        (c for c in pivoted.columns if c != index_col),
        key=int,
    )
    rename_map = {c: f"{name_prefix}_{c}" for c in suffix_cols}
    new_order = [index_col] + [f"{name_prefix}_{c}" for c in suffix_cols]

    return pivoted.rename(rename_map).select(new_order)
