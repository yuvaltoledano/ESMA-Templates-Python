"""Stage 6.5: compose loans_enriched / properties_enriched.

Mirrors the join sequence in r_reference/R/pipeline.R:405-450 that
produces the frames written to the workbook's "Cleaned ESMA loans"
and "Cleaned ESMA properties" sheets:

    loans_enriched <- safe_left_join(loans_filtered_step_2, loan_groups,
                                     by = "calc_loan_id")
    loans_enriched <- safe_left_join(loans_enriched, group_classification,
                                     by = "collateral_group_id") |>
      mutate(
        cross_collateralized_set = (structure_type == "5: ..."),
        full_set                 = (structure_type == "4: Full set"),
      ) |>
      select(-any_of("is_full_set"))

The output column ordering is fully determined by the input frames'
column order plus the deterministic join semantics: stage3 cols ++
[collateral_group_id, loans, collaterals, structure_type] (from the
classification join, with is_full_set dropped) ++ [cross_collateralized_
set, full_set] (mutated in).
"""

from __future__ import annotations

import polars as pl

from esma_milan.pipeline.classification import TYPE_4, TYPE_5


def compose_loans_enriched(
    stage3_loans: pl.DataFrame,
    stage4_loan_groups: pl.DataFrame,
    stage5_classifications: pl.DataFrame,
) -> pl.DataFrame:
    """Return the loans_enriched frame written to "Cleaned ESMA loans".

    Joins on:
      stage3_loans (..., calc_loan_id, calc_borrower_id)
        LEFT JOIN stage4_loan_groups (calc_loan_id, collateral_group_id)
        LEFT JOIN stage5_classifications minus is_full_set
                  (collateral_group_id, loans, collaterals, structure_type)
        + computed cross_collateralized_set, full_set columns
    """
    classification_cols = stage5_classifications.drop("is_full_set")
    return (
        stage3_loans.join(stage4_loan_groups, on="calc_loan_id", how="left")
        .join(classification_cols, on="collateral_group_id", how="left")
        .with_columns(
            cross_collateralized_set=(pl.col("structure_type") == TYPE_5),
            full_set=(pl.col("structure_type") == TYPE_4),
        )
    )


def compose_properties_enriched(
    stage3_properties: pl.DataFrame,
    stage4_collateral_groups: pl.DataFrame,
    stage5_classifications: pl.DataFrame,
) -> pl.DataFrame:
    """Return the properties_enriched frame written to "Cleaned ESMA properties".

    Same shape as compose_loans_enriched but joins via calc_property_id."""
    classification_cols = stage5_classifications.drop("is_full_set")
    return (
        stage3_properties.join(
            stage4_collateral_groups, on="calc_property_id", how="left"
        )
        .join(classification_cols, on="collateral_group_id", how="left")
        .with_columns(
            cross_collateralized_set=(pl.col("structure_type") == TYPE_5),
            full_set=(pl.col("structure_type") == TYPE_4),
        )
    )
