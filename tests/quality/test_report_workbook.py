"""The report workbook: its sheets, text cells, the overflow and the cut."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from openpyxl import load_workbook

from esma_milan.config import DQ_XLSX_MAX_DETAIL_ROWS
from esma_milan.quality import assess_files, complete_report
from esma_milan.quality.catalogue import RULES
from esma_milan.quality.detail_csv import DETAIL_COLUMNS
from esma_milan.quality.report_workbook import (
    CELL_MAX_CHARS,
    SHEETS,
    detail_quotas,
    write_report_workbook,
)
from esma_milan.quality.waterfall import DROPPED_SCHEMA
from tests.quality.builders import Pair
from tests.quality.test_report import STARTED, summary_dict
from tests.quality.test_waterfall import constructed_pool

DIGITS_40 = "1234567890" * 4


def _write(pair: Pair, folder: Path, **limits: int) -> tuple[dict[str, Any], dict[str, object], Any]:
    """Assess the pair (the pipeline is not run), write the workbook, load it."""
    loans, collaterals = pair.write(folder)
    report = complete_report(
        assess_files(loans, collaterals, started_at=STARTED), pipeline=None, error=RuntimeError("x")
    )
    summary = summary_dict(report)
    assert report.waterfall is not None
    path = folder / "report.xlsx"
    cut = write_report_workbook(
        summary, report.assessment, report.waterfall.dropped, path, detail_name="the detail.csv", **limits
    )
    assert not list(folder.glob("*.part"))
    return summary, cut, load_workbook(path)


def _rows(sheet: Any) -> list[dict[str, Any]]:
    rows = list(sheet.iter_rows(values_only=True))
    return [dict(zip(rows[0], row, strict=True)) for row in rows[1:]]


def _odd_pair() -> Pair:
    pair = constructed_pool()
    pair.loans[0].update({"RREL2": "=1+1", "RREL3": "=1+1", "RREL5": "0012", "RREL22": "=sum(a1)"})
    pair.collateral[0].update({"RREC2": "=1+1", "RREC4": DIGITS_40, "RREC13": "-5"})
    pair.loans[1].update({"RREL3": "00002", "RREL22": "a\x1fb", "RREL79": "x" * 40_000})
    return pair


def test_the_sheets_their_headers_and_their_form(tmp_path: Path) -> None:
    summary, cut, book = _write(_odd_pair(), tmp_path)
    assert tuple(book.sheetnames) == SHEETS == tuple(cut["sheets"])  # type: ignore[arg-type]
    headers = {name: [cell.value for cell in book[name][1]] for name in SHEETS}
    assert headers["DQ Summary"] == ["section", "item", "value", "balance", "basis"]
    assert headers["DQ Rows"] == list(DETAIL_COLUMNS)
    assert headers["DQ Dropped loans"] == list(DROPPED_SCHEMA)
    assert headers["DQ Fields"][:3] == ["file", "field", "name"]
    assert headers["DQ Columns"][:3] == ["file", "position", "header"]
    assert headers["DQ Checks"][:5] == ["rule_id", "title", "basis", "severity", "status"]
    assert headers["DQ Findings"][:2] == ["finding_id", "sentence"]
    for name in SHEETS:
        sheet = book[name]
        assert all(cell.font.bold for cell in sheet[1]), name
        assert sheet.freeze_panes == "A2", name
        assert sheet.auto_filter.ref == f"A1:{sheet.cell(1, sheet.max_column).column_letter}{sheet.max_row}"
    # One row a field, a column, a rule and a finding group; every finding row; every dropped loan.
    assert book["DQ Fields"].max_row - 1 == len(summary["expected_fields"]) == 107
    assert book["DQ Columns"].max_row - 1 == sum(len(v) for v in summary["actual_columns"].values())
    assert [r["rule_id"] for r in _rows(book["DQ Checks"])] == list(RULES)
    assert book["DQ Findings"].max_row - 1 == len(summary["findings"])
    assert book["DQ Rows"].max_row - 1 == cut["rows_total"] == cut["rows_included"]
    assert book["DQ Dropped loans"].max_row - 1 == summary["dropped_loans"]["loans"] == 10
    assert cut["cut"] is False and DQ_XLSX_MAX_DETAIL_ROWS == 500_000


def test_values_of_the_files_are_text_cells_and_nothing_is_a_formula(tmp_path: Path) -> None:
    _, _, book = _write(_odd_pair(), tmp_path)
    for sheet in book.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                assert cell.data_type != "f", (sheet.title, cell.coordinate)  # no formula anywhere
    rows = _rows(book["DQ Rows"])
    first = [r for r in rows if r["file"] == "loans" and r["record"] == 1]
    assert {r["loan_identifier_original"] for r in first} == {"=1+1"}
    assert {r["obligor_identifier_new"] for r in first} == {"0012"}  # text, leading zeros kept
    assert "=sum(a1)" in {r["raw_value"] for r in first}
    assert {r["collateral_identifier_new"] for r in rows if r["file"] == "collateral" and r["record"] == 1} == {
        DIGITS_40
    }
    second = [r for r in rows if r["file"] == "loans" and r["record"] == 2]
    assert {r["loan_identifier_new"] for r in second} == {"00002"}
    # A control character is shown escaped, as in the detail CSV.
    assert "a\\x1Fb" in {r["raw_value"] for r in second}
    # Text beyond a cell's limit is cut, with its length and where the whole value is.
    long = next(r["raw_value"] for r in second if r["field"] == "RREL79" and r["raw_value"])
    assert len(long) == CELL_MAX_CHARS
    assert long.endswith("... (40000 characters; the full value is in the detail.csv)")
    # Records and positions are numbers; counts on the other sheets are numbers.
    assert isinstance(first[0]["record"], int) and isinstance(first[0]["physical_line"], int)
    assert all(isinstance(r["rows"], int) for r in _rows(book["DQ Findings"]))
    dropped = _rows(book["DQ Dropped loans"])
    assert dropped[0]["loan_identifier_new"] == "00002" and isinstance(dropped[0]["record"], int)
    # The sample values of the column inventory are cells of the file too.
    samples = " ".join(str(r["sample_values"]) for r in _rows(book["DQ Columns"]))
    assert "=1+1" in samples and "a\\x1Fb" in samples


def test_the_summary_sheet(tmp_path: Path) -> None:
    summary, cut, book = _write(_odd_pair(), tmp_path)
    rows = _rows(book["DQ Summary"])
    by_item = {(r["section"], r["item"]): r for r in rows}
    assert by_item[("Run", "pipeline completed")]["value"] == "False"
    waterfall = [r for r in rows if r["section"] == "Waterfall" and str(r["item"])[:2] in
                 ("1.", "2.", "3.", "4.", "5.", "6.", "7.")]
    assert [r["value"] for r in waterfall][:6] == [14, 2, 12, 4, 8, 6]
    assert waterfall[0]["balance"] == "EUR: 995.0 (13 record(s))" and waterfall[0]["basis"] == "observed"
    assert waterfall[1]["basis"] == "predicted"
    assert by_item[("Dropped loans", "loans")]["value"] == 10
    assert by_item[("This workbook", "finding rows on DQ Rows")]["value"] == cut["rows_total"]
    assert by_item[("This workbook", "statement")]["value"] == (
        "DQ Rows holds every finding row; the detail.csv holds the same rows."
    )
    assert by_item[("Provenance", "rule set version")]["value"] == summary["provenance"]["rule_set_version"]
    assert by_item[("Later releases", "policy outcome")]["value"] == "not implemented in this release"


def test_detail_rows_continue_on_another_sheet(tmp_path: Path) -> None:
    _, cut, book = _write(_odd_pair(), tmp_path, sheet_rows=40)
    total = cut["rows_total"]
    assert isinstance(total, int) and total > 80
    row_sheets = [name for name in book.sheetnames if name.startswith("DQ Rows")]
    assert row_sheets == ["DQ Rows", *[f"DQ Rows {n}" for n in range(2, len(row_sheets) + 1)]]
    assert len(row_sheets) == -(-total // 40)
    assert [book[name].max_row - 1 for name in row_sheets[:-1]] == [40] * (len(row_sheets) - 1)
    assert sum(book[name].max_row - 1 for name in row_sheets) == total
    assert book.sheetnames[-1] == "DQ Dropped loans"  # still the last sheet
    assert [cell.value for cell in book["DQ Rows 2"][1]] == list(DETAIL_COLUMNS)
    # The rows continue in the detail file's order.
    order = [(r["file"], r["record"] or 0) for name in row_sheets for r in _rows(book[name])]
    position = {"loans": 0, "collateral": 1, "pool": 2}
    assert order == sorted(order, key=lambda key: (position[key[0]], key[1]))


def test_the_cut_keeps_every_finding_group_and_says_what_it_shows(tmp_path: Path) -> None:
    summary, cut, book = _write(_odd_pair(), tmp_path, max_detail_rows=30)
    total = sum(f["rows"] for f in summary["findings"])
    assert cut["cut"] is True and cut["rows_total"] == total > 30
    rows = [r for name in book.sheetnames if name.startswith("DQ Rows") for r in _rows(book[name])]
    assert len(rows) == cut["rows_included"] < total
    shown = pl.DataFrame({"finding_id": [r["finding_id"] for r in rows]}).group_by("finding_id").len()
    held = {f["finding_id"]: f["rows"] for f in summary["findings"]}
    # Every finding group keeps at least one row, and its share of the limit.
    assert set(shown["finding_id"].to_list()) == set(held)
    for group, n in shown.iter_rows():
        assert n == max(1, held[group] * 30 // total), group
    by_item = {(r["section"], r["item"]): r["value"] for r in _rows(book["DQ Summary"])}
    assert by_item[("This workbook", "finding rows on DQ Rows")] == len(rows)
    assert by_item[("This workbook", "finding rows in the detail CSV")] == total
    assert by_item[("This workbook", "statement")] == (
        f"DQ Rows holds {len(rows)} of {total} finding rows: the first rows of each finding group, in "
        "proportion to the group's size and at least one of each. Every row is in the detail.csv."
    )
    # The rows shown of a group are its first rows in the file's order.
    largest = max(held, key=lambda g: held[g])
    records = [r["record"] for r in rows if r["finding_id"] == largest]
    assert records == sorted(records)
    # Every other sheet is complete.
    assert book["DQ Findings"].max_row - 1 == len(summary["findings"])
    assert book["DQ Dropped loans"].max_row - 1 == 10


def test_quotas_are_none_when_everything_fits() -> None:
    counts = pl.DataFrame({
        "rule_id": ["P5", "P5", "P9"], "result_kind": ["violation", "violation", "context"],
        "file": ["loans", "loans", "loans"], "field": ["RREL30", "RREL30", None], "rows": [60, 30, 10],
    })
    assert detail_quotas(counts, 100) is None
    assert detail_quotas(counts, 10) == {"P5:violation:loans:RREL30": 9, "P9:context:loans:None": 1}
    assert detail_quotas(counts, 1) == {"P5:violation:loans:RREL30": 1, "P9:context:loans:None": 1}
