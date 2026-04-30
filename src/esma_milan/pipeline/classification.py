"""Stage 5: classify collateral groups into structure types 1-5.

Mirrors r_reference/R/cross_collateralization_functions.R::
classify_collateral_groups (lines 107-137). Consumes the
edges_with_group frame produced by Stage 4 and emits one row per
collateral_group_id with the columns the workbook's "Group
classifications" sheet expects:

    (collateral_group_id, loans, collaterals, is_full_set, structure_type)

Five structure types, distinguished by the (loans, collaterals,
actual_edges, theoretical_edges) tuple per group:

    1. one loan -> one property                       (1 loan, 1 collateral)
    2. one loan -> multiple properties                (1 loan, >1 collaterals)
    3. multiple loans -> one property                 (>1 loans, 1 collateral)
    4. Full set                                       (>1 each, full matrix)
    5. Cross-collateralised set                       (>1 each, partial matrix)

Type-4 vs Type-5 split: actual_edges == loans * collaterals (full
matrix) -> type 4; otherwise type 5.

The unicode arrow in the type strings is U+2192 ("→"), pinned by
the unit tests against the R fixture's exact output.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import structlog

from esma_milan.pipeline.graph import GraphResult

log = structlog.get_logger(__name__)

# Exact strings R produces. Source: cross_collateralization_functions.R:126-131.
# Unicode arrow is U+2192. Pinned by tests/unit/test_classification.py.
TYPE_1: str = "1: one loan → one property"
TYPE_2: str = "2: one loan → multiple properties"
TYPE_3: str = "3: multiple loans → one property"
TYPE_4: str = "4: Full set"
TYPE_5: str = "5: Cross-collateralised set"
TYPE_UNCLASSIFIED: str = "unclassified"


@dataclass(frozen=True)
class Stage5Output:
    """Output of Stage 5: per-group classification frame.

    `classifications` is the table written to the "Group classifications"
    sheet of the output workbook. Columns and dtypes match the R
    fixture's Sheet 8 exactly:

        collateral_group_id : Int64
        loans               : Int64
        collaterals         : Int64
        is_full_set         : Boolean
        structure_type      : String

    Sorted by `collateral_group_id` ascending - matches R's dplyr
    `group_by(...) %>% summarise(...)` natural ordering on integer
    group keys, and is independent of input row order.
    """

    classifications: pl.DataFrame


def classify_collateral_groups(edges_with_group: pl.DataFrame) -> pl.DataFrame:
    """Per collateral_group_id: count distinct loans / collaterals /
    edges; derive is_full_set and structure_type.

    Args:
        edges_with_group: a frame with columns (loan_exposure_id,
            collateral_id, collateral_group_id). Produced by Stage 4's
            `build_loan_collateral_graph`.

    Returns:
        A DataFrame with columns
        (collateral_group_id, loans, collaterals, is_full_set,
         structure_type), sorted by collateral_group_id ascending.

    Raises:
        ValueError: if any required column is missing.
    """
    for col in ("loan_exposure_id", "collateral_id", "collateral_group_id"):
        if col not in edges_with_group.columns:
            raise ValueError(
                f"classify_collateral_groups: edges_with_group is missing "
                f"column {col!r}"
            )

    aggregated = (
        edges_with_group.group_by("collateral_group_id")
        .agg(
            pl.col("loan_exposure_id").n_unique().cast(pl.Int64).alias("loans"),
            pl.col("collateral_id").n_unique().cast(pl.Int64).alias("collaterals"),
            pl.len().cast(pl.Int64).alias("actual_edges"),
        )
        .with_columns(
            theoretical_edges=pl.col("loans") * pl.col("collaterals"),
        )
        .with_columns(
            is_full_set=(
                (pl.col("loans") > 1)
                & (pl.col("collaterals") > 1)
                & (pl.col("actual_edges") == pl.col("theoretical_edges"))
            ),
        )
        .with_columns(
            structure_type=pl.when(
                (pl.col("loans") == 1) & (pl.col("collaterals") == 1)
            )
            .then(pl.lit(TYPE_1))
            .when((pl.col("loans") == 1) & (pl.col("collaterals") > 1))
            .then(pl.lit(TYPE_2))
            .when((pl.col("loans") > 1) & (pl.col("collaterals") == 1))
            .then(pl.lit(TYPE_3))
            .when(
                (pl.col("loans") > 1)
                & (pl.col("collaterals") > 1)
                & pl.col("is_full_set")
            )
            .then(pl.lit(TYPE_4))
            .when(
                (pl.col("loans") > 1)
                & (pl.col("collaterals") > 1)
                & ~pl.col("is_full_set")
            )
            .then(pl.lit(TYPE_5))
            .otherwise(pl.lit(TYPE_UNCLASSIFIED))
        )
        .select(
            "collateral_group_id",
            "loans",
            "collaterals",
            "is_full_set",
            "structure_type",
        )
        .sort("collateral_group_id")
    )

    log.info(
        "classify_collateral_groups",
        n_groups=aggregated.height,
        type_counts={
            row[0]: row[1]
            for row in (
                aggregated.group_by("structure_type")
                .agg(pl.len().alias("n"))
                .sort("structure_type")
                .iter_rows()
            )
        },
    )

    return aggregated


def run_stage5(graph: GraphResult) -> Stage5Output:
    """Stage 5 driver. Wraps classify_collateral_groups for naming
    consistency with run_stage1..run_stage4."""
    return Stage5Output(
        classifications=classify_collateral_groups(graph.edges_with_group)
    )
