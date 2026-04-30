"""Unit tests for the parity diff library itself.

Builds tiny Excel workbooks on the fly and verifies that diff_workbooks
catches the diffs we care about (cell mismatch, column-order mismatch,
row-count mismatch, sheet-missing, group-id canonicalisation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

from esma_milan.parity import diff_workbooks

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
# Sheet-8 cross-sheet alignment via WorkbookContext (Stage 5 addition)
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
    the diff falls back to positional row matching. Pinned so partial
    pipeline outputs (Sheet 8 written but Sheet 6 not yet) keep working."""
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
