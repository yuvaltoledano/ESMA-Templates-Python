"""Tests for esma_milan.pipeline.flatten.

B-3 covers `select_main_property`: 5-key priority sort with the
mandatory deliberate-tie fixtures (lex-smallest calc_property_id wins
when keys 1-4 all tie) plus determinism anchors (reversed property_id
order; shuffled row order) and NA-sort-key behaviour for each of the
four primary keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from esma_milan.pipeline.flatten import select_main_property


def _props(
    *,
    group_ids: Sequence[int],
    property_ids: Sequence[str],
    occupancy_types: Sequence[str | None],
    valuation_methods: Sequence[str | None],
    valuation_dates: Sequence[date | None],
) -> pl.DataFrame:
    """Properties frame post-select_valuation_fields, minimal columns."""
    return pl.DataFrame(
        {
            "collateral_group_id": pl.Series(list(group_ids), dtype=pl.Int64),
            "calc_property_id": pl.Series(list(property_ids), dtype=pl.String),
            "occupancy_type": pl.Series(list(occupancy_types), dtype=pl.String),
            "valuation_method": pl.Series(list(valuation_methods), dtype=pl.String),
            "valuation_date": pl.Series(list(valuation_dates), dtype=pl.Date),
        }
    )


def _unique_vals(
    group_ids: Sequence[int],
    property_ids: Sequence[str],
    values: Sequence[float | None],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "collateral_group_id": pl.Series(list(group_ids), dtype=pl.Int64),
            "calc_property_id": pl.Series(list(property_ids), dtype=pl.String),
            "property_value": pl.Series(list(values), dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Key 1: highest property_value wins
# ---------------------------------------------------------------------------


def test_highest_property_value_wins() -> None:
    """3 properties in one group, distinct values -> highest wins."""
    props = _props(
        group_ids=[1, 1, 1],
        property_ids=["P_A", "P_B", "P_C"],
        occupancy_types=["FOWN", "FOWN", "FOWN"],
        valuation_methods=["FIEI", "FIEI", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 3,
    )
    vals = _unique_vals([1, 1, 1], ["P_A", "P_B", "P_C"], [200000.0, 500000.0, 300000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


# ---------------------------------------------------------------------------
# Key 2: owner-occupied (FOWN) wins on value tie
# ---------------------------------------------------------------------------


def test_value_tie_breaks_to_owner_occupied() -> None:
    """Same value -> FOWN preferred over non-FOWN."""
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["TLET", "FOWN"],
        valuation_methods=["FIEI", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 2,
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


# ---------------------------------------------------------------------------
# Key 3: full-inspection method (FIEI/FOEI) wins on (value, FOWN) tie
# ---------------------------------------------------------------------------


def test_value_and_fown_tie_breaks_to_full_inspection() -> None:
    """Same value, both FOWN -> full-inspection method wins."""
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["FOWN", "FOWN"],
        valuation_methods=["DRVB", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 2,
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


def test_full_inspection_includes_foei_alongside_fiei() -> None:
    """FOEI is also full-inspection per VALUATION_METHODS_FULL_INSPECTION."""
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["FOWN", "FOWN"],
        valuation_methods=["DRVB", "FOEI"],
        valuation_dates=[date(2024, 1, 1)] * 2,
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


# ---------------------------------------------------------------------------
# Key 4: most-recent valuation_date wins on (value, FOWN, method) tie
# ---------------------------------------------------------------------------


def test_value_fown_method_tie_breaks_to_most_recent_date() -> None:
    """Same value/FOWN/FIEI -> most-recent date wins."""
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["FOWN", "FOWN"],
        valuation_methods=["FIEI", "FIEI"],
        valuation_dates=[date(2020, 1, 1), date(2024, 1, 1)],
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


# ---------------------------------------------------------------------------
# Key 5: lex tie-break (THE MANDATORY DELIBERATE-TIE FIXTURE)
# ---------------------------------------------------------------------------


def _deliberate_tie_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Three properties tied on every key 1-4 - only calc_property_id
    distinguishes them. Lex-smallest must win."""
    props = _props(
        group_ids=[1, 1, 1],
        property_ids=["P_C", "P_A", "P_B"],  # deliberately scrambled
        occupancy_types=["FOWN", "FOWN", "FOWN"],
        valuation_methods=["FIEI", "FIEI", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 3,
    )
    vals = _unique_vals(
        [1, 1, 1], ["P_C", "P_A", "P_B"], [500000.0, 500000.0, 500000.0]
    )
    return props, vals


def test_lex_tie_break_picks_smallest_calc_property_id() -> None:
    """All four primary keys tie -> lex-smallest calc_property_id wins."""
    props, vals = _deliberate_tie_fixture()
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_A")


def test_lex_tie_break_invariant_to_reversed_property_id_input_order() -> None:
    """Reverse the row order of the deliberate-tie fixture - lex-smallest
    must still win. Catches "lex-smallest implemented as lex-largest"
    where the implementation might accidentally pick the LAST sorted row."""
    props, vals = _deliberate_tie_fixture()
    out_normal = select_main_property(props, vals)
    out_reversed = select_main_property(props.reverse(), vals.reverse())
    assert out_normal.row(0) == (1, "P_A")
    assert out_reversed.row(0) == (1, "P_A")
    assert out_normal.equals(out_reversed)


def test_lex_tie_break_invariant_to_arbitrary_input_row_shuffle() -> None:
    """Shuffle input rows in a non-trivial permutation. Mirrors B-2's
    determinism anchor: the answer must be identical regardless of
    input row order, since the sort is supposed to be a pure function
    of row content. Catches sort-stability or hidden-ordering leakage
    that the reversed-input test alone would miss."""
    props, vals = _deliberate_tie_fixture()

    # An arbitrary non-trivial permutation: rotate by 1 (move first row
    # to the end). Use Polars row index + sort to materialise the
    # permutation deterministically.
    rotated_props = pl.concat([props.slice(1, 2), props.slice(0, 1)])
    rotated_vals = pl.concat([vals.slice(1, 2), vals.slice(0, 1)])

    out_normal = select_main_property(props, vals)
    out_rotated = select_main_property(rotated_props, rotated_vals)
    assert out_normal.row(0) == (1, "P_A")
    assert out_rotated.row(0) == (1, "P_A")
    assert out_normal.equals(out_rotated)


# ---------------------------------------------------------------------------
# NA-sort-key behaviour (Polars vs R sort-order parity)
# ---------------------------------------------------------------------------


def test_null_property_value_sorts_after_non_null() -> None:
    """NA property_value -> sorts after non-NA. With nulls_last=True
    descending, the null-value row loses despite no other distinguishing
    features. Catches a future Polars change where descending sort
    moves nulls first."""
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["FOWN", "FOWN"],
        valuation_methods=["FIEI", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 2,
    )
    # P_A null value, P_B has 100k. Even though P_A's lex is smaller,
    # the null sorts last on key 1 -> P_B wins.
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [None, 100000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


def test_null_occupancy_type_sorts_after_fown_and_non_fown() -> None:
    """Per user direction: occupancy_type == "FOWN" produces null when
    the input is null, and Polars vs R may sort that null differently
    under desc(). With nulls_last=True the null sorts AFTER both True
    (FOWN) and False (non-FOWN) values.

    Fixture: two rows, all keys except occupancy_type tied. Row B has
    occupancy_type=TLET (False), Row A has occupancy_type=None (Null).
    P_A is lex-smaller, so if lex tie-break decided, P_A would win. But
    the boolean key 2 fires first: False > Null -> P_B wins.
    """
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=[None, "TLET"],
        valuation_methods=["FIEI", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 2,
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


def test_null_valuation_method_sorts_after_inspection_and_non_inspection() -> None:
    """Same shape as above for key 3 (`valuation_method ∈ {FIEI, FOEI}`).

    Fixture: two rows tied on keys 1-2, differ on key 3. Row B has
    valuation_method=DRVB (False on inspection check), Row A has
    valuation_method=None (Null). P_A is lex-smaller. Boolean key 3
    fires: False > Null -> P_B wins.
    """
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["FOWN", "FOWN"],
        valuation_methods=[None, "DRVB"],
        valuation_dates=[date(2024, 1, 1)] * 2,
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


def test_null_valuation_date_sorts_after_non_null_dates() -> None:
    """Key 4 (valuation_date) - null sorts after non-null with nulls_last=True
    descending. Non-null date wins."""
    props = _props(
        group_ids=[1, 1],
        property_ids=["P_A", "P_B"],
        occupancy_types=["FOWN", "FOWN"],
        valuation_methods=["FIEI", "FIEI"],
        valuation_dates=[None, date(2020, 1, 1)],
    )
    vals = _unique_vals([1, 1], ["P_A", "P_B"], [500000.0, 500000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_B")


# ---------------------------------------------------------------------------
# Multi-group + edge cases
# ---------------------------------------------------------------------------


def test_multi_group_each_independently() -> None:
    """Multiple groups in one input - each group's main property is
    selected independently."""
    props = _props(
        group_ids=[1, 1, 2, 2, 3],
        property_ids=["P_A", "P_B", "P_X", "P_Y", "P_Z"],
        occupancy_types=["FOWN", "FOWN", "FOWN", "FOWN", "FOWN"],
        valuation_methods=["FIEI", "FIEI", "FIEI", "FIEI", "FIEI"],
        valuation_dates=[date(2024, 1, 1)] * 5,
    )
    vals = _unique_vals(
        [1, 1, 2, 2, 3],
        ["P_A", "P_B", "P_X", "P_Y", "P_Z"],
        [200000.0, 500000.0, 300000.0, 100000.0, 400000.0],
    )
    out = select_main_property(props, vals)
    by_gid = {row[0]: row[1] for row in out.iter_rows()}
    assert by_gid == {1: "P_B", 2: "P_X", 3: "P_Z"}


def test_single_property_per_group_picks_that_property() -> None:
    """Group with one property -> that one wins trivially."""
    props = _props(
        group_ids=[1],
        property_ids=["P_ONLY"],
        occupancy_types=["FOWN"],
        valuation_methods=["FIEI"],
        valuation_dates=[date(2024, 1, 1)],
    )
    vals = _unique_vals([1], ["P_ONLY"], [100000.0])
    out = select_main_property(props, vals)
    assert out.row(0) == (1, "P_ONLY")


def test_output_one_row_per_group_sorted_by_group_id() -> None:
    """Output is exactly one row per collateral_group_id, sorted by
    group_id ascending."""
    props = _props(
        group_ids=[3, 1, 2, 3, 1],
        property_ids=["P_3a", "P_1a", "P_2a", "P_3b", "P_1b"],
        occupancy_types=["FOWN"] * 5,
        valuation_methods=["FIEI"] * 5,
        valuation_dates=[date(2024, 1, 1)] * 5,
    )
    vals = _unique_vals(
        [3, 1, 2, 3, 1],
        ["P_3a", "P_1a", "P_2a", "P_3b", "P_1b"],
        [100.0, 200.0, 300.0, 400.0, 500.0],
    )
    out = select_main_property(props, vals)
    assert out["collateral_group_id"].to_list() == [1, 2, 3]


def test_output_schema() -> None:
    """Schema check: collateral_group_id Int64, main_property_id String."""
    props = _props(
        group_ids=[1],
        property_ids=["P_A"],
        occupancy_types=["FOWN"],
        valuation_methods=["FIEI"],
        valuation_dates=[date(2024, 1, 1)],
    )
    vals = _unique_vals([1], ["P_A"], [100000.0])
    out = select_main_property(props, vals)
    assert out.schema["collateral_group_id"] == pl.Int64
    assert out.schema["main_property_id"] == pl.String
    assert out.columns == ["collateral_group_id", "main_property_id"]
