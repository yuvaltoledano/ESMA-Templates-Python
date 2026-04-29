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
