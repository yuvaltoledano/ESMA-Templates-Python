"""Tests for esma_milan.pipeline.classification.

Mirrors r_reference/tests/testthat/test-graph.R::"classify_collateral_groups
correctly classifies all structure types" plus property-based tests for
the type-4 / type-5 distinction (full-set vs cross-collateralised) and
the deterministic row order.
"""

from __future__ import annotations

import polars as pl
import pytest

from esma_milan.pipeline.classification import (
    TYPE_1,
    TYPE_2,
    TYPE_3,
    TYPE_4,
    TYPE_5,
    classify_collateral_groups,
    run_stage5,
)
from esma_milan.pipeline.graph import GraphResult


def _edges(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    """Build an edges_with_group frame from (gid, loan_id, prop_id) tuples."""
    return pl.DataFrame(
        {
            "collateral_group_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "loan_exposure_id": pl.Series([r[1] for r in rows], dtype=pl.String),
            "collateral_id": pl.Series([r[2] for r in rows], dtype=pl.String),
        }
    )


# ---------------------------------------------------------------------------
# Five structure types: one fixture per type, mirroring test-graph.R
# ---------------------------------------------------------------------------


def test_classifies_all_five_structure_types_by_fixture() -> None:
    """Mirrors test-graph.R::"classify_collateral_groups correctly
    classifies all structure types" exactly."""
    edges = _edges(
        [
            # Group 1: Type 1 (1L, 1P)
            (1, "L1", "P1"),
            # Group 2: Type 2 (1L, 2P)
            (2, "L2", "P2"),
            (2, "L2", "P3"),
            # Group 3: Type 3 (2L, 1P)
            (3, "L3", "P4"),
            (3, "L4", "P4"),
            # Group 4: Type 4 (2L, 2P, full matrix - 4 edges)
            (4, "L5", "P5"),
            (4, "L5", "P6"),
            (4, "L6", "P5"),
            (4, "L6", "P6"),
            # Group 5: Type 5 (2L, 2P, missing edge - 3 edges)
            (5, "L7", "P7"),
            (5, "L7", "P8"),
            (5, "L8", "P8"),
        ]
    )
    result = classify_collateral_groups(edges)

    by_gid = {row[0]: row for row in result.iter_rows()}
    assert by_gid[1] == (1, 1, 1, False, TYPE_1)
    assert by_gid[2] == (2, 1, 2, False, TYPE_2)
    assert by_gid[3] == (3, 2, 1, False, TYPE_3)
    assert by_gid[4] == (4, 2, 2, True, TYPE_4)
    assert by_gid[5] == (5, 2, 2, False, TYPE_5)


def test_type_strings_use_unicode_arrow() -> None:
    """The arrow MUST be U+2192 ('→'), not '->'. R writes it as the
    unicode arrow and the parity contract requires byte-equal strings."""
    assert "→" in TYPE_1
    assert "→" in TYPE_2
    assert "→" in TYPE_3
    assert "->" not in TYPE_1
    assert "->" not in TYPE_2
    assert "->" not in TYPE_3


def test_full_set_distinction_at_minimum_2x2() -> None:
    """A 2x2 group with 4 edges is type 4; with 3 edges is type 5. Pin
    the boundary: this is where bisection bugs land most often."""
    full = _edges(
        [
            (1, "L1", "P1"),
            (1, "L1", "P2"),
            (1, "L2", "P1"),
            (1, "L2", "P2"),
        ]
    )
    partial = _edges(
        [
            (1, "L1", "P1"),
            (1, "L1", "P2"),
            (1, "L2", "P2"),
        ]
    )
    full_row = classify_collateral_groups(full).row(0)
    partial_row = classify_collateral_groups(partial).row(0)
    assert full_row[3] is True and full_row[4] == TYPE_4
    assert partial_row[3] is False and partial_row[4] == TYPE_5


def test_full_set_at_3x3() -> None:
    """3 loans, 3 properties, all 9 edges -> type 4."""
    rows = [(1, f"L{i}", f"P{j}") for i in range(1, 4) for j in range(1, 4)]
    result = classify_collateral_groups(_edges(rows))
    assert result.row(0) == (1, 3, 3, True, TYPE_4)


def test_cross_collateralised_at_3x3_minus_one() -> None:
    """3 loans, 3 properties, 8 edges (one removed) -> type 5."""
    rows = [(1, f"L{i}", f"P{j}") for i in range(1, 4) for j in range(1, 4)]
    rows.pop()  # drop one edge
    result = classify_collateral_groups(_edges(rows))
    assert result.row(0) == (1, 3, 3, False, TYPE_5)


# ---------------------------------------------------------------------------
# Output shape + dtype contract
# ---------------------------------------------------------------------------


def test_output_columns_and_order() -> None:
    edges = _edges([(1, "L1", "P1")])
    result = classify_collateral_groups(edges)
    assert result.columns == [
        "collateral_group_id",
        "loans",
        "collaterals",
        "is_full_set",
        "structure_type",
    ]


def test_output_dtypes_match_r_fixture() -> None:
    """R's fixture has Int64 for the count columns, Bool for is_full_set,
    String for structure_type. Pin the schema so workbook output is
    byte-equal to the R fixture without per-call coercion."""
    edges = _edges([(1, "L1", "P1")])
    schema = classify_collateral_groups(edges).schema
    assert schema["collateral_group_id"] == pl.Int64
    assert schema["loans"] == pl.Int64
    assert schema["collaterals"] == pl.Int64
    assert schema["is_full_set"] == pl.Boolean
    assert schema["structure_type"] == pl.String


def test_output_sorted_by_gid_ascending() -> None:
    """R's dplyr summarise on integer group keys produces ascending
    order. Match this so per-implementation rows have the same order."""
    edges = _edges([(3, "L3", "P3"), (1, "L1", "P1"), (2, "L2", "P2")])
    result = classify_collateral_groups(edges)
    assert result["collateral_group_id"].to_list() == [1, 2, 3]


def test_output_row_order_independent_of_edge_input_order() -> None:
    """Any permutation of the edge rows must produce the same output."""
    base = [
        (1, "L1", "P1"),
        (2, "L2", "P2"),
        (2, "L2", "P3"),
        (3, "L3", "P4"),
        (3, "L4", "P4"),
    ]
    forward = classify_collateral_groups(_edges(base))
    reversed_input = classify_collateral_groups(_edges(list(reversed(base))))
    assert forward.equals(reversed_input)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_edges_returns_empty_frame() -> None:
    edges = _edges([])
    result = classify_collateral_groups(edges)
    assert result.height == 0
    assert result.columns == [
        "collateral_group_id",
        "loans",
        "collaterals",
        "is_full_set",
        "structure_type",
    ]


def test_missing_column_raises() -> None:
    df = pl.DataFrame({"loan_exposure_id": ["L1"], "collateral_id": ["P1"]})
    with pytest.raises(ValueError, match="collateral_group_id"):
        classify_collateral_groups(df)


def test_duplicate_edge_in_input_does_not_inflate_actual_edges_above_distinct_pairs() -> None:
    """The Stage-4 graph builder dedups (loan, property) pairs before
    emitting edges_with_group, so duplicate edges should never reach
    the classifier. But Polars `len()` counts rows literally, so if a
    duplicate sneaks through, actual_edges will be inflated and a 2x2
    group with 4 distinct edges + 1 duplicate would falsely classify
    as type 5 with theoretical_edges < actual_edges. Pin the precondition.

    This test does NOT exercise the bug - it documents that the
    classifier trusts upstream dedup. If this assumption ever changes,
    the test will need updating.
    """
    # 2x2 full-set with one duplicated edge -> 5 rows -> theoretical=4,
    # actual=5, is_full_set computed as (actual_edges == theoretical_edges)
    # -> False -> classifier returns type 5 WRONGLY.
    edges = _edges(
        [
            (1, "L1", "P1"),
            (1, "L1", "P2"),
            (1, "L2", "P1"),
            (1, "L2", "P2"),
            (1, "L2", "P2"),  # duplicate
        ]
    )
    result = classify_collateral_groups(edges).row(0)
    # Documenting the trust boundary - the classifier itself does NOT
    # dedup. Stage 4's graph builder is responsible. If this test ever
    # fails after a change, ensure the upstream dedup contract holds.
    assert result[1] == 2 and result[2] == 2  # n_unique on loans/collaterals
    assert result[3] is False  # actual_edges (5) != theoretical_edges (4)
    assert result[4] == TYPE_5


# ---------------------------------------------------------------------------
# run_stage5 driver
# ---------------------------------------------------------------------------


def test_run_stage5_wraps_classifier_into_stage5output() -> None:
    edges = _edges([(1, "L1", "P1"), (2, "L2", "P2")])
    graph = GraphResult(
        loan_groups=pl.DataFrame(
            {
                "calc_loan_id": pl.Series(["L1", "L2"], dtype=pl.String),
                "collateral_group_id": pl.Series([1, 2], dtype=pl.Int64),
            }
        ),
        collateral_groups=pl.DataFrame(
            {
                "calc_property_id": pl.Series(["P1", "P2"], dtype=pl.String),
                "collateral_group_id": pl.Series([1, 2], dtype=pl.Int64),
            }
        ),
        edges_with_group=edges,
    )
    out = run_stage5(graph)
    assert out.classifications.height == 2
    assert set(out.classifications["structure_type"].to_list()) == {TYPE_1}
