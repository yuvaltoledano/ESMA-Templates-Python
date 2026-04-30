"""Tests for esma_milan.pipeline.mapping_tables.

Mirrors r_reference/R/pipeline.R:333-391 + :452-465. The four mapping
tables share the same recipe (distinct (key, value) pairs pivoted into
suffix-numbered columns); only "Loans to properties" gets the
classification tail.

Test coverage:
  - Per-sheet column composition + dtypes (4 sheets).
  - Cross-cutting: shared-borrower dedup; shared-property dedup;
    loan-with-no-matching-property defensive case;
    MAX_PROPERTIES_PER_LOAN warning text + threshold boundary.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import polars as pl
import pytest

from esma_milan.config import MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD
from esma_milan.pipeline.mapping_tables import compose_mapping_tables


def _stage3_loans(
    loan_ids: Sequence[str],
    borrower_ids: Sequence[str],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "calc_loan_id": pl.Series(list(loan_ids), dtype=pl.String),
            "calc_borrower_id": pl.Series(list(borrower_ids), dtype=pl.String),
        }
    )


def _stage3_properties(
    loan_refs: Sequence[str],
    property_ids: Sequence[str],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "underlying_exposure_identifier": pl.Series(
                list(loan_refs), dtype=pl.String
            ),
            "calc_property_id": pl.Series(list(property_ids), dtype=pl.String),
        }
    )


def _loans_enriched(
    loan_ids: Sequence[str],
    group_ids: Sequence[int],
    cross_collat: Sequence[bool],
    full_set: Sequence[bool],
    structure_types: Sequence[str],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "calc_loan_id": pl.Series(list(loan_ids), dtype=pl.String),
            "collateral_group_id": pl.Series(list(group_ids), dtype=pl.Int64),
            "cross_collateralized_set": pl.Series(list(cross_collat), dtype=pl.Boolean),
            "full_set": pl.Series(list(full_set), dtype=pl.Boolean),
            "structure_type": pl.Series(list(structure_types), dtype=pl.String),
        }
    )


# ---------------------------------------------------------------------------
# Per-sheet column composition + dtypes
# ---------------------------------------------------------------------------


def test_loans_to_properties_columns_and_dtypes() -> None:
    loans = _stage3_loans(["L1", "L2"], ["B1", "B2"])
    props = _stage3_properties(["L1", "L2"], ["P1", "P2"])
    le = _loans_enriched(
        loan_ids=["L1", "L2"],
        group_ids=[1, 2],
        cross_collat=[False, False],
        full_set=[False, False],
        structure_types=["1: one loan → one property"] * 2,
    )
    out = compose_mapping_tables(loans, props, le)["Loans to properties"]
    assert out.columns == [
        "loan_id",
        "property_id_1",
        "collateral_group_id",
        "cross_collateralized_set",
        "full_set",
        "structure_type",
    ]
    assert out.schema["loan_id"] == pl.String
    assert out.schema["property_id_1"] == pl.String
    assert out.schema["collateral_group_id"] == pl.Int64
    assert out.schema["cross_collateralized_set"] == pl.Boolean
    assert out.schema["full_set"] == pl.Boolean
    assert out.schema["structure_type"] == pl.String


def test_properties_to_loans_columns_and_dtypes() -> None:
    loans = _stage3_loans(["L1", "L2"], ["B1", "B2"])
    props = _stage3_properties(["L1", "L2"], ["P1", "P2"])
    le = _loans_enriched(["L1", "L2"], [1, 2], [False] * 2, [False] * 2, ["t"] * 2)
    out = compose_mapping_tables(loans, props, le)["Properties to loans"]
    assert out.columns == ["property_id", "loan_id_1"]
    assert out.schema["property_id"] == pl.String
    assert out.schema["loan_id_1"] == pl.String


def test_borrowers_to_loans_columns_and_dtypes() -> None:
    loans = _stage3_loans(["L1", "L2"], ["B1", "B2"])
    props = _stage3_properties(["L1", "L2"], ["P1", "P2"])
    le = _loans_enriched(["L1", "L2"], [1, 2], [False] * 2, [False] * 2, ["t"] * 2)
    out = compose_mapping_tables(loans, props, le)["Borrowers to loans"]
    assert out.columns == ["borrower_id", "loan_id_1"]
    assert out.schema["borrower_id"] == pl.String
    assert out.schema["loan_id_1"] == pl.String


def test_borrowers_to_properties_columns_and_dtypes() -> None:
    loans = _stage3_loans(["L1", "L2"], ["B1", "B2"])
    props = _stage3_properties(["L1", "L2"], ["P1", "P2"])
    le = _loans_enriched(["L1", "L2"], [1, 2], [False] * 2, [False] * 2, ["t"] * 2)
    out = compose_mapping_tables(loans, props, le)["Borrowers to properties"]
    assert out.columns == ["borrower_id", "property_id_1"]
    assert out.schema["borrower_id"] == pl.String
    assert out.schema["property_id_1"] == pl.String


# ---------------------------------------------------------------------------
# Per-sheet row composition + ordering (input-first-occurrence semantics)
# ---------------------------------------------------------------------------


def test_loans_to_properties_one_row_per_loan_in_input_order() -> None:
    loans = _stage3_loans(["L_C", "L_A", "L_B"], ["B1", "B2", "B3"])
    props = _stage3_properties(
        ["L_C", "L_A", "L_B"],
        ["P1", "P2", "P3"],
    )
    le = _loans_enriched(
        ["L_C", "L_A", "L_B"], [1, 2, 3], [False] * 3, [False] * 3, ["t"] * 3
    )
    out = compose_mapping_tables(loans, props, le)["Loans to properties"]
    # Input order preserved (NOT sorted alphabetically).
    assert out["loan_id"].to_list() == ["L_C", "L_A", "L_B"]


def test_properties_to_loans_one_row_per_property_in_input_order() -> None:
    loans = _stage3_loans(["L1"], ["B1"])
    props = _stage3_properties(["L1", "L1"], ["P_C", "P_A"])
    le = _loans_enriched(["L1"], [1], [False], [False], ["t"])
    out = compose_mapping_tables(loans, props, le)["Properties to loans"]
    assert out["property_id"].to_list() == ["P_C", "P_A"]


# ---------------------------------------------------------------------------
# Multi-value pivoting (>1 trailing slot)
# ---------------------------------------------------------------------------


def test_loans_to_properties_multi_property_loan_uses_property_id_1_and_2() -> None:
    """LOAN_002-style: one loan with two properties -> property_id_1
    and property_id_2 populated."""
    loans = _stage3_loans(["L1"], ["B1"])
    props = _stage3_properties(["L1", "L1"], ["P_A", "P_B"])
    le = _loans_enriched(["L1"], [1], [False], [False], ["t"])
    out = compose_mapping_tables(loans, props, le)["Loans to properties"]
    assert out.row(0)[:3] == ("L1", "P_A", "P_B")


def test_properties_to_loans_shared_property_uses_loan_id_1_and_2() -> None:
    """PROP_004-style: property shared by two loans -> loan_id_1 and
    loan_id_2 populated."""
    loans = _stage3_loans(["L_A", "L_B"], ["B1", "B2"])
    props = _stage3_properties(["L_A", "L_B"], ["P1", "P1"])
    le = _loans_enriched(
        ["L_A", "L_B"], [1, 1], [False] * 2, [False] * 2, ["t"] * 2
    )
    out = compose_mapping_tables(loans, props, le)["Properties to loans"]
    assert out.row(0) == ("P1", "L_A", "L_B")


# ---------------------------------------------------------------------------
# Shared-borrower dedup (the BORIG_03 pattern)
# ---------------------------------------------------------------------------


def test_borrowers_to_loans_shared_borrower_emits_loan_id_1_and_2() -> None:
    """One borrower owns two loans -> single row with both loans in
    suffix slots. Mirrors BORIG_03 sharing LOAN_003+LOAN_004 in the
    synthetic fixture."""
    loans = _stage3_loans(["L_A", "L_B"], ["B_SHARED", "B_SHARED"])
    props = _stage3_properties(["L_A", "L_B"], ["P_A", "P_B"])
    le = _loans_enriched(
        ["L_A", "L_B"], [1, 2], [False] * 2, [False] * 2, ["t"] * 2
    )
    out = compose_mapping_tables(loans, props, le)["Borrowers to loans"]
    assert out.height == 1
    assert out.row(0) == ("B_SHARED", "L_A", "L_B")


def test_borrowers_to_properties_shared_borrower_dedups_properties() -> None:
    """Borrower's two loans share a property -> distinct() collapses
    to a single property_id_1 (no duplicate). Mirrors BORIG_03 whose
    two loans both reference PROP_004."""
    loans = _stage3_loans(["L_A", "L_B"], ["B_SHARED", "B_SHARED"])
    props = _stage3_properties(["L_A", "L_B"], ["P_SHARED", "P_SHARED"])
    le = _loans_enriched(
        ["L_A", "L_B"], [1, 1], [False] * 2, [False] * 2, ["t"] * 2
    )
    out = compose_mapping_tables(loans, props, le)["Borrowers to properties"]
    assert out.height == 1
    # Only one distinct property -> property_id_1 = P_SHARED, no _2 column.
    assert out.columns == ["borrower_id", "property_id_1"]
    assert out.row(0) == ("B_SHARED", "P_SHARED")


# ---------------------------------------------------------------------------
# Loan with no matching property (defensive; documents R's behaviour)
# ---------------------------------------------------------------------------


def test_loan_with_no_matching_property_has_null_property_id_1() -> None:
    """Stage 3's intersection filter prevents this on the synthetic
    fixture, but on a malformed real fixture it could happen. R's
    left_join keeps the loan row with property_id=NA; pivot_wider
    emits property_id_1=NA. Mirror exactly."""
    loans = _stage3_loans(["L_A", "L_B"], ["B1", "B2"])
    # L_B has no property in props
    props = _stage3_properties(["L_A"], ["P_A"])
    le = _loans_enriched(
        ["L_A", "L_B"], [1, 2], [False] * 2, [False] * 2, ["t"] * 2
    )
    out = compose_mapping_tables(loans, props, le)["Loans to properties"]
    by_loan = {row[0]: row for row in out.iter_rows()}
    assert by_loan["L_A"][1] == "P_A"
    assert by_loan["L_B"][1] is None


# ---------------------------------------------------------------------------
# MAX_PROPERTIES_PER_LOAN warning (text + threshold boundary)
# ---------------------------------------------------------------------------


def test_max_properties_per_loan_warning_does_not_fire_at_threshold() -> None:
    """Threshold is `> 50`, not `>= 50`. A loan with exactly 50
    properties must NOT trigger the warning."""
    n = MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD  # 50
    loans = _stage3_loans(["L1"], ["B1"])
    props = _stage3_properties(["L1"] * n, [f"P{i:03d}" for i in range(n)])
    le = _loans_enriched(["L1"], [1], [False], [False], ["t"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        compose_mapping_tables(loans, props, le)


def test_max_properties_per_loan_warning_fires_above_threshold_with_exact_text() -> None:
    """Loan with > 50 properties triggers the warning. The text must
    match R's `warning("Some loans have ", N, " properties (threshold:
    ", THRESH, ").")` exactly. Pinning text catches a future R-repo
    update that changes the threshold or message format."""
    n = MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD + 1  # 51
    loans = _stage3_loans(["L1"], ["B1"])
    props = _stage3_properties(["L1"] * n, [f"P{i:03d}" for i in range(n)])
    le = _loans_enriched(["L1"], [1], [False], [False], ["t"])
    expected = (
        f"Some loans have {n} properties "
        f"(threshold: {MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD})."
    )
    with pytest.warns(UserWarning) as caught:
        compose_mapping_tables(loans, props, le)
    messages = [str(w.message) for w in caught]
    assert expected in messages, (
        f"Expected exact text {expected!r}, got messages: {messages}"
    )


# ---------------------------------------------------------------------------
# Empty-input edge case
# ---------------------------------------------------------------------------


def test_empty_inputs_produce_empty_frames_with_minimal_schema() -> None:
    """Empty stage3 frames -> empty output with only the index column
    (no suffix slots, since pivot_wider has no values to widen)."""
    loans = _stage3_loans([], [])
    props = _stage3_properties([], [])
    le = _loans_enriched([], [], [], [], [])
    out = compose_mapping_tables(loans, props, le)
    for sheet_name in (
        "Loans to properties",
        "Properties to loans",
        "Borrowers to loans",
        "Borrowers to properties",
    ):
        assert out[sheet_name].height == 0
