"""Tests for the Stage 4 graph builder.

Mirrors r_reference/tests/testthat/test-graph.R one-for-one (the
build_loan_collateral_graph block) plus dedicated determinism tests
that the R reference doesn't have but we need because networkx's
connected_components has no guaranteed order.
"""

from __future__ import annotations

import polars as pl
import pytest

from esma_milan.pipeline.graph import (
    GraphResult,
    build_loan_collateral_graph,
)


def _loans(loan_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"calc_loan_id": loan_ids}, schema={"calc_loan_id": pl.String})


def _properties(rows: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "underlying_exposure_identifier": [r[0] for r in rows],
            "calc_property_id": [r[1] for r in rows],
        },
        schema={
            "underlying_exposure_identifier": pl.String,
            "calc_property_id": pl.String,
        },
    )


# ---------------------------------------------------------------------------
# Mirrors r_reference/tests/testthat/test-graph.R::"build_loan_collateral_graph
# correctly identifies connected components"
# ---------------------------------------------------------------------------


def test_complex_scenario_yields_five_components_and_twelve_edges() -> None:
    """8 loans, 12 edges, 5 connected components - one for each
    structure type. Mirrors the R test exactly."""
    loans = _loans(["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"])
    properties = _properties([
        ("L1", "P1"),
        ("L2", "P2"),
        ("L2", "P3"),
        ("L3", "P4"),
        ("L4", "P4"),
        ("L5", "P5"),
        ("L5", "P6"),
        ("L6", "P5"),
        ("L6", "P6"),
        ("L7", "P7"),
        ("L7", "P8"),
        ("L8", "P8"),
    ])

    result = build_loan_collateral_graph(loans, properties)

    n_groups = result.loan_groups["collateral_group_id"].n_unique()
    assert n_groups == 5
    assert result.edges_with_group.height == 12


def test_returns_graph_result_with_three_named_frames() -> None:
    loans = _loans(["L1"])
    properties = _properties([("L1", "P1")])
    result = build_loan_collateral_graph(loans, properties)
    assert isinstance(result, GraphResult)
    assert "calc_loan_id" in result.loan_groups.columns
    assert "collateral_group_id" in result.loan_groups.columns
    assert "calc_property_id" in result.collateral_groups.columns
    assert "collateral_group_id" in result.collateral_groups.columns
    assert "loan_exposure_id" in result.edges_with_group.columns
    assert "collateral_id" in result.edges_with_group.columns
    assert "collateral_group_id" in result.edges_with_group.columns


# ---------------------------------------------------------------------------
# Determinism guarantees (the part the R reference doesn't test)
# ---------------------------------------------------------------------------


def test_group_ids_are_deterministic_across_input_row_orders() -> None:
    """Shuffling the input rows must NOT change which (loan_id, group_id)
    pairs appear in the output. networkx's connected_components has no
    guaranteed order; this test pins our explicit canonicalisation.
    """
    loans_a = _loans(["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"])
    loans_b = _loans(["L8", "L7", "L6", "L5", "L4", "L3", "L2", "L1"])

    edges = [
        ("L1", "P1"),
        ("L2", "P2"),
        ("L2", "P3"),
        ("L3", "P4"),
        ("L4", "P4"),
        ("L5", "P5"),
        ("L5", "P6"),
        ("L6", "P5"),
        ("L6", "P6"),
        ("L7", "P7"),
        ("L7", "P8"),
        ("L8", "P8"),
    ]
    properties_a = _properties(edges)
    properties_b = _properties(list(reversed(edges)))

    result_a = build_loan_collateral_graph(loans_a, properties_a)
    result_b = build_loan_collateral_graph(loans_b, properties_b)

    # Sort both sides by calc_loan_id so the per-loan group_id assignment
    # is directly comparable.
    a_loan_pairs = sorted(
        zip(
            result_a.loan_groups["calc_loan_id"].to_list(),
            result_a.loan_groups["collateral_group_id"].to_list(),
            strict=True,
        )
    )
    b_loan_pairs = sorted(
        zip(
            result_b.loan_groups["calc_loan_id"].to_list(),
            result_b.loan_groups["collateral_group_id"].to_list(),
            strict=True,
        )
    )
    assert a_loan_pairs == b_loan_pairs


def test_group_ids_are_assigned_in_lex_order_of_smallest_member() -> None:
    """Group containing L1 gets id=1, group containing L2 gets id=2,
    etc., because L1 < L2 < L3 lex-wise. Pins the canonical labelling."""
    loans = _loans(["L1", "L2", "L3"])
    properties = _properties([("L1", "P1"), ("L2", "P2"), ("L3", "P3")])
    result = build_loan_collateral_graph(loans, properties)

    pairs = dict(
        zip(
            result.loan_groups["calc_loan_id"].to_list(),
            result.loan_groups["collateral_group_id"].to_list(),
            strict=True,
        )
    )
    assert pairs == {"L1": 1, "L2": 2, "L3": 3}


def test_group_ids_are_one_indexed_contiguous() -> None:
    """For a graph with N components, group IDs span exactly [1..N]."""
    loans = _loans(["L1", "L2", "L3", "L4"])
    properties = _properties(
        [("L1", "P1"), ("L2", "P2"), ("L3", "P3"), ("L4", "P4")]
    )
    result = build_loan_collateral_graph(loans, properties)
    gids = sorted(set(result.loan_groups["collateral_group_id"].to_list()))
    assert gids == [1, 2, 3, 4]


def test_loan_groups_sorted_by_calc_loan_id_ascending() -> None:
    """Output frame is sorted - identical inputs always produce
    identical row order."""
    loans = _loans(["L3", "L1", "L2"])
    properties = _properties([("L1", "P1"), ("L2", "P2"), ("L3", "P3")])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups["calc_loan_id"].to_list() == ["L1", "L2", "L3"]


# ---------------------------------------------------------------------------
# Edge cases (empty / disjoint / single-edge)
# ---------------------------------------------------------------------------


def test_empty_loans_and_properties_yields_empty_graph() -> None:
    loans = _loans([])
    properties = _properties([])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups.height == 0
    assert result.collateral_groups.height == 0
    assert result.edges_with_group.height == 0


def test_loans_with_no_matching_properties_yields_empty_edges() -> None:
    """Loans without any property reference produce no edges and no
    nodes (graph only carries nodes that participate in at least one
    edge - matches R's igraph behaviour)."""
    loans = _loans(["L1", "L2"])
    properties = _properties([])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups.height == 0
    assert result.collateral_groups.height == 0
    assert result.edges_with_group.height == 0


def test_single_edge_yields_one_two_node_component() -> None:
    loans = _loans(["L1"])
    properties = _properties([("L1", "P1")])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups.height == 1
    assert result.collateral_groups.height == 1
    assert result.edges_with_group.height == 1
    assert result.loan_groups["collateral_group_id"].to_list() == [1]
    assert result.collateral_groups["collateral_group_id"].to_list() == [1]


def test_two_disjoint_edges_yield_two_components() -> None:
    loans = _loans(["L1", "L2"])
    properties = _properties([("L1", "P1"), ("L2", "P2")])
    result = build_loan_collateral_graph(loans, properties)
    assert sorted(set(result.loan_groups["collateral_group_id"].to_list())) == [1, 2]


def test_duplicate_edges_collapsed() -> None:
    """Mirrors R's `dplyr::distinct(loan_exposure_id, collateral_id)` -
    a (L1, P1) edge listed three times in the input still produces one
    edge in the output."""
    loans = _loans(["L1"])
    properties = _properties([("L1", "P1"), ("L1", "P1"), ("L1", "P1")])
    result = build_loan_collateral_graph(loans, properties)
    assert result.edges_with_group.height == 1


# ---------------------------------------------------------------------------
# Synthetic-fixture-shaped scenarios for the five structure types
# ---------------------------------------------------------------------------


def test_type_1_one_loan_one_property() -> None:
    loans = _loans(["L1"])
    properties = _properties([("L1", "P1")])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups.height == 1
    assert result.collateral_groups.height == 1


def test_type_2_one_loan_multiple_properties() -> None:
    loans = _loans(["L1"])
    properties = _properties([("L1", "P1"), ("L1", "P2")])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups.height == 1
    assert result.collateral_groups.height == 2
    # Single component - both properties get the same group_id.
    assert result.collateral_groups["collateral_group_id"].n_unique() == 1


def test_type_3_multiple_loans_one_property() -> None:
    loans = _loans(["L1", "L2"])
    properties = _properties([("L1", "P1"), ("L2", "P1")])
    result = build_loan_collateral_graph(loans, properties)
    assert result.loan_groups.height == 2
    assert result.collateral_groups.height == 1
    assert result.loan_groups["collateral_group_id"].n_unique() == 1


def test_type_4_full_set_two_loans_two_properties_four_edges() -> None:
    loans = _loans(["L1", "L2"])
    properties = _properties(
        [("L1", "P1"), ("L1", "P2"), ("L2", "P1"), ("L2", "P2")]
    )
    result = build_loan_collateral_graph(loans, properties)
    # All four nodes in one component.
    assert result.loan_groups["collateral_group_id"].n_unique() == 1


def test_type_5_cross_collat_three_edges_missing_one() -> None:
    loans = _loans(["L1", "L2"])
    properties = _properties([("L1", "P1"), ("L2", "P1"), ("L2", "P2")])
    result = build_loan_collateral_graph(loans, properties)
    # Still one component (L1-P1-L2-P2 connected via L2), but only 3 edges.
    assert result.loan_groups["collateral_group_id"].n_unique() == 1
    assert result.edges_with_group.height == 3


# ---------------------------------------------------------------------------
# Required-column validation
# ---------------------------------------------------------------------------


def test_missing_calc_loan_id_raises() -> None:
    loans = pl.DataFrame({"other_col": ["L1"]})
    properties = _properties([("L1", "P1")])
    with pytest.raises(ValueError, match="calc_loan_id"):
        build_loan_collateral_graph(loans, properties)


def test_missing_underlying_exposure_identifier_raises() -> None:
    loans = _loans(["L1"])
    properties = pl.DataFrame({"calc_property_id": ["P1"]})
    with pytest.raises(ValueError, match="underlying_exposure_identifier"):
        build_loan_collateral_graph(loans, properties)


def test_missing_calc_property_id_raises() -> None:
    loans = _loans(["L1"])
    properties = pl.DataFrame({"underlying_exposure_identifier": ["L1"]})
    with pytest.raises(ValueError, match="calc_property_id"):
        build_loan_collateral_graph(loans, properties)
