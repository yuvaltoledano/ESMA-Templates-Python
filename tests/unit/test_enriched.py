"""Unit tests for compose_loans_enriched / compose_properties_enriched.

The composition is responsible for the column ORDER and the computed
cross_collateralized_set / full_set columns. Parity vs the R fixture is
covered end-to-end by tests/integration/test_stage6_5_writer.py; this
module exercises the helpers in isolation.
"""

from __future__ import annotations

import polars as pl

from esma_milan.pipeline.classification import (
    TYPE_1,
    TYPE_4,
    TYPE_5,
)
from esma_milan.pipeline.enriched import (
    compose_loans_enriched,
    compose_properties_enriched,
)


def _classifications(rows: list[tuple[int, int, int, bool, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "collateral_group_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "loans": pl.Series([r[1] for r in rows], dtype=pl.Int64),
            "collaterals": pl.Series([r[2] for r in rows], dtype=pl.Int64),
            "is_full_set": pl.Series([r[3] for r in rows], dtype=pl.Boolean),
            "structure_type": pl.Series([r[4] for r in rows], dtype=pl.String),
        }
    )


# ---------------------------------------------------------------------------
# compose_loans_enriched
# ---------------------------------------------------------------------------


def test_compose_loans_enriched_appends_columns_in_canonical_order() -> None:
    """Loans-side enrichment ends with collateral_group_id, loans,
    collaterals, structure_type, cross_collateralized_set, full_set
    (matching the synthetic fixture's column tail)."""
    stage3_loans = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
            "calc_borrower_id": pl.Series(["B1"], dtype=pl.String),
        }
    )
    stage4_loan_groups = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
            "collateral_group_id": pl.Series([1], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications([(1, 1, 1, False, TYPE_1)])

    result = compose_loans_enriched(
        stage3_loans, stage4_loan_groups, stage5_classifications
    )
    assert result.columns == [
        "calc_loan_id",
        "calc_borrower_id",
        "collateral_group_id",
        "loans",
        "collaterals",
        "structure_type",
        "cross_collateralized_set",
        "full_set",
    ]


def test_compose_loans_enriched_drops_is_full_set_column() -> None:
    """is_full_set is dropped during the join (R's `select(-any_of(
    "is_full_set"))`); cross_collateralized_set / full_set are computed
    from structure_type instead."""
    stage3_loans = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
        }
    )
    stage4_loan_groups = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
            "collateral_group_id": pl.Series([1], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications([(1, 1, 1, True, TYPE_4)])

    result = compose_loans_enriched(
        stage3_loans, stage4_loan_groups, stage5_classifications
    )
    assert "is_full_set" not in result.columns


def test_compose_loans_enriched_computes_full_set_from_type_4() -> None:
    """structure_type == "4: Full set" -> full_set=True;
    cross_collateralized_set=False."""
    stage3_loans = pl.DataFrame(
        {"calc_loan_id": pl.Series(["L1"], dtype=pl.String)}
    )
    stage4_loan_groups = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
            "collateral_group_id": pl.Series([1], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications([(1, 2, 2, True, TYPE_4)])

    result = compose_loans_enriched(
        stage3_loans, stage4_loan_groups, stage5_classifications
    )
    assert result["full_set"].to_list() == [True]
    assert result["cross_collateralized_set"].to_list() == [False]


def test_compose_loans_enriched_computes_cross_from_type_5() -> None:
    stage3_loans = pl.DataFrame(
        {"calc_loan_id": pl.Series(["L1"], dtype=pl.String)}
    )
    stage4_loan_groups = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
            "collateral_group_id": pl.Series([1], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications([(1, 2, 2, False, TYPE_5)])

    result = compose_loans_enriched(
        stage3_loans, stage4_loan_groups, stage5_classifications
    )
    assert result["full_set"].to_list() == [False]
    assert result["cross_collateralized_set"].to_list() == [True]


def test_compose_loans_enriched_neither_full_nor_cross_for_types_1_2_3() -> None:
    """Types 1, 2, 3 -> both bool columns False."""
    stage3_loans = pl.DataFrame(
        {"calc_loan_id": pl.Series(["L1"], dtype=pl.String)}
    )
    stage4_loan_groups = pl.DataFrame(
        {
            "calc_loan_id": pl.Series(["L1"], dtype=pl.String),
            "collateral_group_id": pl.Series([1], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications([(1, 1, 1, False, TYPE_1)])

    result = compose_loans_enriched(
        stage3_loans, stage4_loan_groups, stage5_classifications
    )
    assert result["full_set"].to_list() == [False]
    assert result["cross_collateralized_set"].to_list() == [False]


# ---------------------------------------------------------------------------
# compose_properties_enriched
# ---------------------------------------------------------------------------


def test_compose_properties_enriched_appends_columns_in_canonical_order() -> None:
    """Properties-side enrichment column tail mirrors loans-side."""
    stage3_props = pl.DataFrame(
        {
            "underlying_exposure_identifier": pl.Series(["L1"], dtype=pl.String),
            "calc_property_id": pl.Series(["P1"], dtype=pl.String),
        }
    )
    stage4_collateral_groups = pl.DataFrame(
        {
            "calc_property_id": pl.Series(["P1"], dtype=pl.String),
            "collateral_group_id": pl.Series([1], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications([(1, 1, 1, False, TYPE_1)])

    result = compose_properties_enriched(
        stage3_props, stage4_collateral_groups, stage5_classifications
    )
    assert result.columns == [
        "underlying_exposure_identifier",
        "calc_property_id",
        "collateral_group_id",
        "loans",
        "collaterals",
        "structure_type",
        "cross_collateralized_set",
        "full_set",
    ]


def test_compose_properties_enriched_handles_multiple_groups() -> None:
    """Multiple groups + multiple property rows: full_set / cross_set
    computed per row from each row's structure_type."""
    stage3_props = pl.DataFrame(
        {
            "calc_property_id": pl.Series(
                ["P1", "P2", "P3", "P4"], dtype=pl.String
            ),
        }
    )
    stage4_collateral_groups = pl.DataFrame(
        {
            "calc_property_id": pl.Series(
                ["P1", "P2", "P3", "P4"], dtype=pl.String
            ),
            "collateral_group_id": pl.Series([1, 2, 3, 4], dtype=pl.Int64),
        }
    )
    stage5_classifications = _classifications(
        [
            (1, 1, 1, False, TYPE_1),
            (2, 2, 2, True, TYPE_4),
            (3, 2, 2, False, TYPE_5),
            (4, 1, 1, False, TYPE_1),
        ]
    )

    result = compose_properties_enriched(
        stage3_props, stage4_collateral_groups, stage5_classifications
    )
    assert result["full_set"].to_list() == [False, True, False, False]
    assert result["cross_collateralized_set"].to_list() == [False, False, True, False]
