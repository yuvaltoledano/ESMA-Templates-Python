"""Unit tests for the parity diff library itself.

Builds tiny Excel workbooks on the fly and verifies that diff_workbooks
catches the diffs we care about (cell mismatch, column-order mismatch,
row-count mismatch, sheet-missing, group-id canonicalisation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

from esma_milan.parity import StringFloatTolerance, diff_workbooks

# Type alias kept loose because the literal dicts in these tests carry
# heterogeneous row types that mypy widens to list[object]; openpyxl's
# ws.append() accepts anything iterable.
SheetData = dict[str, Any]


def _write_xlsx(path: Path, sheets: SheetData) -> None:
    wb = openpyxl.Workbook(write_only=True)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(r)
    wb.save(path)


def test_identical_workbooks_pass(tmp_path: Path) -> None:
    sheets = {"S": [["a", "b"], [1, 2.5], [3, 4.5]]}
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, sheets)
    _write_xlsx(b, sheets)
    rep = diff_workbooks(a, b)
    assert rep.passed
    assert all(s.passed for s in rep.sheet_diffs)


def test_value_mismatch_caught(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["x"], ["foo"]]})
    _write_xlsx(b, {"S": [["x"], ["bar"]]})
    rep = diff_workbooks(a, b)
    assert not rep.passed
    sd = next(s for s in rep.sheet_diffs if s.sheet == "S")
    assert len(sd.cell_diffs) == 1
    assert sd.cell_diffs[0].column_name == "x"


def test_numeric_within_tolerance_passes(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["v"], [1.0 + 1e-12]]})
    _write_xlsx(b, {"S": [["v"], [1.0]]})
    rep = diff_workbooks(a, b)
    assert rep.passed


def test_numeric_outside_tolerance_fails(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["v"], [1.0001]]})
    _write_xlsx(b, {"S": [["v"], [1.0]]})
    rep = diff_workbooks(a, b)
    assert not rep.passed


def test_column_order_mismatch_caught(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["a", "b"], [1, 2]]})
    _write_xlsx(b, {"S": [["b", "a"], [2, 1]]})
    rep = diff_workbooks(a, b)
    sd = next(s for s in rep.sheet_diffs if s.sheet == "S")
    assert not sd.column_order_match
    # When column order doesn't match we don't bother diffing cells.
    assert not sd.cell_diffs


def test_sheet_order_mismatch_caught(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"X": [["v"], [1]], "Y": [["v"], [2]]})
    _write_xlsx(b, {"Y": [["v"], [2]], "X": [["v"], [1]]})
    rep = diff_workbooks(a, b)
    assert rep.actual_sheet_order != rep.expected_sheet_order
    assert not rep.passed


def test_row_count_mismatch_noted(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["v"], [1]]})
    _write_xlsx(b, {"S": [["v"], [1], [2]]})
    rep = diff_workbooks(a, b)
    sd = next(s for s in rep.sheet_diffs if s.sheet == "S")
    assert any("row-count mismatch" in n for n in sd.notes)


def test_missing_sheet_noted(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["v"], [1]]})
    _write_xlsx(b, {"S": [["v"], [1]], "T": [["w"], [2]]})
    rep = diff_workbooks(a, b)
    t = next(s for s in rep.sheet_diffs if s.sheet == "T")
    assert not t.actual_present


def test_group_id_canonicalisation_works_on_property_sheets(tmp_path: Path) -> None:
    """The Cleaned ESMA properties sheet has no calc_loan_id column; the
    canonicalisation falls through the priority list to calc_property_id
    so group-id relabelling still works for that sheet's group IDs."""
    actual = {
        "Cleaned ESMA properties": [
            ["calc_property_id", "collateral_group_id"],
            ["P1", 5],
            ["P2", 5],
            ["P3", 9],
        ]
    }
    expected = {
        "Cleaned ESMA properties": [
            ["calc_property_id", "collateral_group_id"],
            ["P1", 1],
            ["P2", 1],
            ["P3", 2],
        ]
    }
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, actual)
    _write_xlsx(b, expected)
    rep = diff_workbooks(a, b)
    assert rep.passed, "property-sheet canonicalisation should mask label permutation"


def test_group_id_canonicalisation_handles_relabelling(tmp_path: Path) -> None:
    """Same partition, different integer labels -> diff passes."""
    # Loans L1, L2 in group X; loan L3 alone in group Y. R might label
    # X=1, Y=2; Python might label X=2, Y=1. Both must diff as equal.
    actual = {
        "Cleaned ESMA loans": [
            ["calc_loan_id", "collateral_group_id"],
            ["L1", 2],
            ["L2", 2],
            ["L3", 1],
        ]
    }
    expected = {
        "Cleaned ESMA loans": [
            ["calc_loan_id", "collateral_group_id"],
            ["L1", 1],
            ["L2", 1],
            ["L3", 2],
        ]
    }
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, actual)
    _write_xlsx(b, expected)
    rep = diff_workbooks(a, b)
    assert rep.passed, "group-id canonicalisation should mask label permutation"


def test_group_id_canonicalisation_catches_real_partition_diff(tmp_path: Path) -> None:
    """Different partition -> diff still fails after canonicalisation."""
    actual = {
        "Cleaned ESMA loans": [
            ["calc_loan_id", "collateral_group_id"],
            ["L1", 1],  # Python: L1 alone
            ["L2", 2],
            ["L3", 2],
        ]
    }
    expected = {
        "Cleaned ESMA loans": [
            ["calc_loan_id", "collateral_group_id"],
            ["L1", 1],  # R: L1 + L2 together
            ["L2", 1],
            ["L3", 2],
        ]
    }
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, actual)
    _write_xlsx(b, expected)
    rep = diff_workbooks(a, b)
    assert not rep.passed


def test_numeric_vs_iso_string_date_is_a_diff(tmp_path: Path) -> None:
    """An Excel-serial cell (43539) is not the same as the ISO string
    '2019-03-15'; the diff must catch this. Confirms the contract that
    each sheet stores dates in its own format."""
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["d"], [43539]]})
    _write_xlsx(b, {"S": [["d"], ["2019-03-15"]]})
    rep = diff_workbooks(a, b)
    assert not rep.passed


def test_empty_and_none_treated_as_equal(tmp_path: Path) -> None:
    a = tmp_path / "a.xlsx"
    b = tmp_path / "b.xlsx"
    _write_xlsx(a, {"S": [["v"], [None]]})
    _write_xlsx(b, {"S": [["v"], [""]]})
    rep = diff_workbooks(a, b)
    assert rep.passed


# ---------------------------------------------------------------------------
# Sheet-8 cross-sheet alignment via WorkbookContext
# ---------------------------------------------------------------------------


def _aligned_workbook(
    path: Path,
    *,
    loans_rows: list[tuple[str, int]],  # (calc_loan_id, collateral_group_id)
    sheet8_rows: list[
        tuple[int, int, int, bool, str]
    ],  # (gid, loans, collaterals, is_full_set, structure_type)
    extra_sheets: SheetData | None = None,
) -> None:
    """Two-sheet workbook with the schemas expected by the alignment logic."""
    sheets: SheetData = {
        "Cleaned ESMA loans": [
            ["calc_loan_id", "collateral_group_id"],
            *([list(r) for r in loans_rows]),
        ],
        "Group classifications": [
            [
                "collateral_group_id",
                "loans",
                "collaterals",
                "is_full_set",
                "structure_type",
            ],
            *([list(r) for r in sheet8_rows]),
        ],
    }
    if extra_sheets:
        sheets.update(extra_sheets)
    _write_xlsx(path, sheets)


def test_sheet_8_aligns_rows_when_gid_labels_differ_between_sides(tmp_path: Path) -> None:
    """Same partition, divergent gid labels, divergent Sheet-8 row orders.

    Without alignment, R's Sheet-8 row 0 (gid=4 = {L3} singleton, "T1")
    would compare against Python's Sheet-8 row 0 (gid=1 = {L1,L2}, "T3")
    and falsely flag a structure_type mismatch. Alignment via the
    Loans-derived min-loan-id key collapses this to a clean pass.

    Partition: {L1,L2} (T3), {L3} (T1), {L4,L5} (T3).
    Actual labels (Python-style, lex order):  1,2,3.
    Expected labels (R-style, hash order):    7,4,9.
    """
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _aligned_workbook(
        a,
        loans_rows=[("L1", 1), ("L2", 1), ("L3", 2), ("L4", 3), ("L5", 3)],
        sheet8_rows=[
            (1, 2, 1, False, "T3"),
            (2, 1, 1, False, "T1"),
            (3, 2, 1, False, "T3"),
        ],
    )
    _aligned_workbook(
        e,
        loans_rows=[("L1", 7), ("L2", 7), ("L3", 4), ("L4", 9), ("L5", 9)],
        # R emits Sheet 8 sorted by gid asc -> 4, 7, 9 -> the {L3} group
        # comes FIRST. Without alignment, positional row matching would
        # diff its row-0 against actual's row-0 ({L1,L2}) -> false fail.
        sheet8_rows=[
            (4, 1, 1, False, "T1"),
            (7, 2, 1, False, "T3"),
            (9, 2, 1, False, "T3"),
        ],
    )
    rep = diff_workbooks(a, e)
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    assert s8.passed, f"unexpected diffs:\n{[d.short() for d in s8.cell_diffs]}"
    # The alignment should be visible in the notes (helps debugging).
    assert any("rows aligned" in n for n in s8.notes)


def test_sheet_8_canonicalises_gid_value_via_workbook_context(tmp_path: Path) -> None:
    """Sheet 8's gid column must be relabelled via the cross-sheet
    context (the in-sheet relabel returns {} for Sheet 8 since it has
    no calc_loan_id column of its own)."""
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _aligned_workbook(
        a,
        loans_rows=[("L1", 5)],
        sheet8_rows=[(5, 1, 1, False, "T1")],
    )
    _aligned_workbook(
        e,
        loans_rows=[("L1", 99)],
        sheet8_rows=[(99, 1, 1, False, "T1")],
    )
    rep = diff_workbooks(a, e)
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    assert s8.passed, "value canonicalisation should mask gid label permutation"


def test_sheet_8_alignment_does_not_mask_partition_differences(tmp_path: Path) -> None:
    """Different PARTITION (not just different labels) -> diff still fails.

    Actual: one group of 3 loans on 1 collateral (Type 3).
    Expected: two separate groups - {L1,L2} and {L3} - on the same collateral.
    Alignment can't repair this; diff must catch the loans-count and
    row-count differences.
    """
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _aligned_workbook(
        a,
        loans_rows=[("L1", 1), ("L2", 1), ("L3", 1)],
        sheet8_rows=[(1, 3, 1, False, "T3")],
    )
    _aligned_workbook(
        e,
        loans_rows=[("L1", 1), ("L2", 1), ("L3", 2)],
        sheet8_rows=[
            (1, 2, 1, False, "T3"),
            (2, 1, 1, False, "T1"),
        ],
    )
    rep = diff_workbooks(a, e)
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    assert not s8.passed
    # Row count should be flagged.
    assert any("row-count mismatch" in n for n in s8.notes)
    # And the first row's `loans` should diff (3 vs 2).
    assert any(
        d.column_name == "loans" and d.actual == 3 and d.expected == 2
        for d in s8.cell_diffs
    )


def test_sheet_8_alignment_skipped_when_actual_loans_sheet_empty(tmp_path: Path) -> None:
    """If one side has no loans-sheet context, alignment is a no-op and
    the diff falls back to positional row matching. Pinned so hand-built
    or partial workbooks without a Cleaned ESMA loans sheet still diff."""
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _aligned_workbook(
        a,
        loans_rows=[],  # empty loans
        sheet8_rows=[(1, 1, 1, False, "T1")],
    )
    _aligned_workbook(
        e,
        loans_rows=[("L1", 7)],
        sheet8_rows=[(7, 1, 1, False, "T1")],
    )
    rep = diff_workbooks(a, e)
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    # Without alignment, positional cell match: gid 1 vs gid 7 -> diff.
    assert not s8.passed
    assert not any("rows aligned" in n for n in s8.notes)


def test_sheet_8_alignment_skipped_when_loans_sheet_missing_entirely(tmp_path: Path) -> None:
    """Workbooks that don't even have a Cleaned ESMA loans sheet still
    diff cleanly via positional matching."""
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _write_xlsx(
        a,
        {
            "Group classifications": [
                ["collateral_group_id", "loans", "collaterals", "is_full_set", "structure_type"],
                [1, 1, 1, False, "T1"],
            ],
        },
    )
    _write_xlsx(
        e,
        {
            "Group classifications": [
                ["collateral_group_id", "loans", "collaterals", "is_full_set", "structure_type"],
                [1, 1, 1, False, "T1"],
            ],
        },
    )
    rep = diff_workbooks(a, e)
    # Sheet 8 itself matches positionally; the diff is against the
    # missing "Cleaned ESMA loans" sheet which appears in expected only
    # because it's listed in actual_sheet_order. Confirm Sheet 8 itself
    # still passes.
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    assert s8.passed
    assert not any("rows aligned" in n for n in s8.notes)


def test_sheet_8_row_with_unknown_gid_sorts_to_end_stably(tmp_path: Path) -> None:
    """A Sheet-8 row whose gid is not present in the loans-sheet context
    (corruption / orphan group) sorts to the end deterministically."""
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _aligned_workbook(
        a,
        loans_rows=[("L1", 1)],
        sheet8_rows=[
            (99, 0, 0, False, "ORPHAN"),  # gid=99 not in loans -> tier 1, end
            (1, 1, 1, False, "T1"),
        ],
    )
    _aligned_workbook(
        e,
        loans_rows=[("L1", 7)],
        sheet8_rows=[
            (99, 0, 0, False, "ORPHAN"),  # same shape on both sides
            (7, 1, 1, False, "T1"),
        ],
    )
    rep = diff_workbooks(a, e)
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    # After alignment: known gids first (sorted by L1), then orphan (gid=99
    # for both sides -> identical str(gid) -> stable). Passes.
    assert s8.passed, f"unexpected diffs:\n{[d.short() for d in s8.cell_diffs]}"


def test_sheet_8_alignment_no_op_when_partition_matches_with_same_labels(
    tmp_path: Path,
) -> None:
    """Synthetic-fixture case: both sides happen to use identical gid
    labels for the same partition (lex-min-loan-id order). Alignment
    is a no-op, diff still passes, behaviour matches the no-alignment
    code path."""
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    rows: list[tuple[int, int, int, bool, str]] = [
        (1, 1, 1, False, "T1"),
        (2, 2, 1, False, "T3"),
        (3, 2, 1, False, "T3"),
    ]
    _aligned_workbook(
        a,
        loans_rows=[("L1", 1), ("L2", 2), ("L3", 2), ("L4", 3), ("L5", 3)],
        sheet8_rows=rows,
    )
    _aligned_workbook(
        e,
        loans_rows=[("L1", 1), ("L2", 2), ("L3", 2), ("L4", 3), ("L5", 3)],
        sheet8_rows=rows,
    )
    rep = diff_workbooks(a, e)
    s8 = next(s for s in rep.sheet_diffs if s.sheet == "Group classifications")
    assert s8.passed


# ---------------------------------------------------------------------------
# WorkbookContext direct unit tests
# ---------------------------------------------------------------------------


def test_workbook_context_built_from_loans_sheet(tmp_path: Path) -> None:
    from esma_milan.parity.diff import _build_workbook_context, _load

    p = tmp_path / "a.xlsx"
    _aligned_workbook(
        p,
        loans_rows=[("L3", 1), ("L1", 2), ("L2", 1), ("L5", 3), ("L4", 3)],
        sheet8_rows=[(1, 2, 1, False, "T3")],  # not used here
    )
    wb = _load(p)
    ctx = _build_workbook_context(wb)
    # gid=1 contains {L3, L2} -> min "L2"; gid=2 contains {L1} -> min "L1";
    # gid=3 contains {L5, L4} -> min "L4".
    assert ctx.gid_to_min_loan_id == {1: "L2", 2: "L1", 3: "L4"}
    # Sorted by min loan: L1, L2, L4 -> gids 2, 1, 3 -> canonical 1, 2, 3.
    assert ctx.gid_to_canonical == {2: 1, 1: 2, 3: 3}
    assert ctx.is_populated


def test_workbook_context_empty_when_loans_sheet_missing(tmp_path: Path) -> None:
    from esma_milan.parity.diff import _build_workbook_context, _load

    p = tmp_path / "a.xlsx"
    _write_xlsx(p, {"Some Other Sheet": [["x"], [1]]})
    wb = _load(p)
    ctx = _build_workbook_context(wb)
    assert not ctx.is_populated
    assert ctx.gid_to_min_loan_id == {}
    assert ctx.gid_to_canonical == {}


def test_workbook_context_empty_when_loans_sheet_lacks_required_columns(
    tmp_path: Path,
) -> None:
    from esma_milan.parity.diff import _build_workbook_context, _load

    p = tmp_path / "a.xlsx"
    _write_xlsx(
        p,
        {
            "Cleaned ESMA loans": [
                ["calc_loan_id"],  # no collateral_group_id column
                ["L1"],
            ],
        },
    )
    wb = _load(p)
    ctx = _build_workbook_context(wb)
    assert not ctx.is_populated


def test_workbook_context_empty_when_loans_sheet_has_no_rows(tmp_path: Path) -> None:
    from esma_milan.parity.diff import _build_workbook_context, _load

    p = tmp_path / "a.xlsx"
    _write_xlsx(
        p,
        {
            "Cleaned ESMA loans": [
                ["calc_loan_id", "collateral_group_id"],
            ],
        },
    )
    wb = _load(p)
    ctx = _build_workbook_context(wb)
    assert not ctx.is_populated


# ---------------------------------------------------------------------------
# Every cell is compared; the cap limits kept examples only
# ---------------------------------------------------------------------------


def _one_column_pair(
    tmp_path: Path,
    actual: list[Any],
    expected: list[Any],
    *,
    sheet: str = "S",
    column: str = "v",
) -> tuple[Path, Path]:
    a = tmp_path / "a.xlsx"
    e = tmp_path / "e.xlsx"
    _write_xlsx(a, {sheet: [[column], *[[v] for v in actual]]})
    _write_xlsx(e, {sheet: [[column], *[[v] for v in expected]]})
    return a, e


# Two 15-significant-digit strings that differ in the last digit only
# (relative difference about 1e-15).
_DRIFT_A = "88567.200601784"
_DRIFT_E = "88567.2006017839"


def _tol(*columns: str, rel_tol: float = 1e-13) -> StringFloatTolerance:
    return StringFloatTolerance(columns=frozenset(columns), rel_tol=rel_tol)


def test_material_diff_after_more_than_cap_tolerated_pairs_fails(tmp_path: Path) -> None:
    """The failure the old gate had: 60 tolerated pairs filled the cap and
    a material difference behind them was never assessed."""
    a, e = _one_column_pair(
        tmp_path, [_DRIFT_A] * 60 + ["1.5"], [_DRIFT_E] * 60 + ["2.5"]
    )
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("v")})
    sd = rep.sheet_diffs[0]
    assert not rep.passed and not sd.passed
    assert sd.n_tolerated == 60
    assert sd.n_cell_diffs == 1
    assert [d.row for d in sd.cell_diffs] == [62]


def test_material_diff_after_more_than_cap_material_diffs_is_counted(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, ["x"] * 61, ["y"] * 61)
    rep = diff_workbooks(a, e)
    sd = rep.sheet_diffs[0]
    assert sd.n_cell_diffs == 61
    assert len(sd.cell_diffs) == 50
    assert "showing 50 of 61 cell diffs" in sd.notes
    assert not sd.passed


def test_cap_zero_keeps_no_examples_and_still_fails_a_differing_sheet(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, ["x", "same"], ["y", "same"])
    for cap in (0, -5):
        sd = diff_workbooks(a, e, max_cell_diffs_per_sheet=cap).sheet_diffs[0]
        assert sd.cell_diffs == []
        assert sd.n_cell_diffs == 1
        assert not sd.passed


def test_cap_zero_passes_an_identical_sheet(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, ["x", 1.5], ["x", 1.5])
    rep = diff_workbooks(a, e, max_cell_diffs_per_sheet=0)
    assert rep.passed
    assert rep.sheet_diffs[0].n_cell_diffs == 0


def test_passed_and_total_do_not_depend_on_the_cap(tmp_path: Path) -> None:
    differing = _one_column_pair(tmp_path, ["x"] * 70 + ["k"], ["y"] * 70 + ["k"])
    (tmp_path / "same").mkdir()
    identical = _one_column_pair(tmp_path / "same", ["k"] * 5, ["k"] * 5)
    for (a, e), expected_passed, expected_total in (
        (differing, False, 70),
        (identical, True, 0),
    ):
        for cap in (0, 1, 50, 10_000):
            rep = diff_workbooks(a, e, max_cell_diffs_per_sheet=cap)
            sd = rep.sheet_diffs[0]
            assert rep.passed is expected_passed, cap
            assert sd.n_cell_diffs == expected_total, cap
            assert len(sd.cell_diffs) == min(cap, expected_total), cap


# ---------------------------------------------------------------------------
# StringFloatTolerance: scope
# ---------------------------------------------------------------------------


def test_tolerated_pair_in_listed_column_passes_and_is_counted(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, [_DRIFT_A], [_DRIFT_E])
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("v")})
    assert rep.passed
    assert rep.sheet_diffs[0].n_tolerated == 1
    assert rep.sheet_diffs[0].n_cell_diffs == 0


def test_same_pair_without_tolerance_argument_fails(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, [_DRIFT_A], [_DRIFT_E])
    rep = diff_workbooks(a, e)
    assert not rep.passed
    assert rep.sheet_diffs[0].n_tolerated == 0


def test_within_tolerance_pair_in_unlisted_column_fails(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, [_DRIFT_A], [_DRIFT_E])
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("other")})
    assert not rep.passed


def test_within_tolerance_pair_in_listed_column_of_another_sheet_fails(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, [_DRIFT_A], [_DRIFT_E], sheet="T")
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("v")})
    assert not rep.passed
    assert rep.sheet_diffs[0].n_tolerated == 0


def test_listed_column_pair_outside_tolerance_fails(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, ["88567.200601784"], ["88567.200601"])
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("v")})
    assert not rep.passed
    assert rep.sheet_diffs[0].n_cell_diffs == 1


def test_tolerance_requires_two_strings(tmp_path: Path) -> None:
    """A number against a numeric string stays a type mismatch, listed
    column or not."""
    a, e = _one_column_pair(tmp_path, [1.5], ["1.5"])
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("v")})
    assert not rep.passed
    assert "type mismatch" in rep.sheet_diffs[0].cell_diffs[0].reason


def test_unparseable_and_nan_strings_are_never_tolerated(tmp_path: Path) -> None:
    a, e = _one_column_pair(tmp_path, ["abc", "nan", "ND"], ["abd", "NaN", "1.0"])
    rep = diff_workbooks(a, e, string_float_tolerances={"S": _tol("v")})
    sd = rep.sheet_diffs[0]
    assert sd.n_tolerated == 0
    assert sd.n_cell_diffs == 3


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_summary_and_report_state_total_kept_and_tolerated(tmp_path: Path) -> None:
    from esma_milan.parity import format_report

    a, e = _one_column_pair(
        tmp_path, [_DRIFT_A, *["x"] * 7], [_DRIFT_E, *["y"] * 7]
    )
    rep = diff_workbooks(
        a, e, max_cell_diffs_per_sheet=5, string_float_tolerances={"S": _tol("v")}
    )
    sheet = rep.summary()["sheets"][0]
    assert sheet["cell_diffs"] == 7
    assert sheet["cell_diffs_kept"] == 5
    assert sheet["tolerated"] == 1
    text = format_report(rep, max_diffs_shown=2)
    assert "=== Parity check: FAIL ===" in text
    assert "7 cell diff(s), 5 kept as examples, 2 shown:" in text
    assert "1 string pair(s) within tolerance" in text
    assert "note: showing 5 of 7 cell diffs" in text


def test_parity_script_exit_code_is_independent_of_both_limits(tmp_path: Path) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_parity_check.py"
    spec = importlib.util.spec_from_file_location("run_parity_check", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    a, e = _one_column_pair(tmp_path, ["x"], ["y"])
    assert module.main([str(a), str(e)]) == 1
    assert module.main([str(a), str(e), "--max-diffs-per-sheet", "0"]) == 1
    assert module.main([str(a), str(e), "--max-shown", "0"]) == 1
    assert module.main([str(a), str(a), "--max-diffs-per-sheet", "0"]) == 0
