"""The detail CSV and the dropped-loans CSV: the escape rules, the order,
and that every finding row is there exactly once."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

from esma_milan.quality import assess_files, complete_report
from esma_milan.quality.assess import FileAssessment
from esma_milan.quality.detail_csv import (
    DETAIL_COLUMNS,
    DETAIL_SCHEMA_LINE,
    DROPPED_SCHEMA_LINE,
    detail_chunks,
    escape,
    unescape,
    write_detail_csv,
    write_dropped_loans_csv,
)
from esma_milan.quality.store import ResultStore
from esma_milan.quality.waterfall import DROPPED_SCHEMA
from tests.quality.builders import Pair
from tests.quality.test_report import FIXTURES, STARTED
from tests.quality.test_waterfall import constructed_pool

DIGITS_40 = "1234567890" * 4


def _read(path: Path) -> tuple[str, list[dict[str, str]]]:
    """The schema line and the rows of a detail file, cells as written."""
    with path.open(encoding="utf-8", newline="") as handle:
        schema = handle.readline().rstrip("\n")
        return schema, list(csv.DictReader(handle))


def _odd_pair() -> Pair:
    pair = Pair()
    pair.set("RREL22", "A\x1fB", record=1)  # a control character
    pair.set("RREL79", "A\\x1FB", record=1)  # the literal text of an escape
    pair.set("RREL82", "C:\\dir", record=1)  # a backslash
    pair.set("RREL78", "x\ny", record=1)  # a line break inside a quoted cell
    pair.set("RREL2", "=1+1", record=1).set("RREL3", "=1+1", record=1).set("RREC2", "=1+1", record=1)
    pair.set("RREL5", "0012", record=1)  # leading zeros
    pair.set("RREC4", DIGITS_40, record=1)  # a 40-digit identifier
    pair.set("RREL40", "abc", record=2).set("RREC13", "-5", record=1)
    pair.set("RREC1", "ANOTHER IDENTIFIER")  # a finding about the pool (A11)
    return pair


def _write(pair: Pair, folder: Path) -> tuple[FileAssessment, Path]:
    loans, collaterals = pair.write(folder)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    path = folder / "detail.csv"
    write_detail_csv(assessment, path)
    return assessment, path


@pytest.mark.parametrize(
    ("text", "written"),
    [
        ("A\x1fB", "A\\x1FB"), ("A\\x1FB", "A\\\\x1FB"), ("C:\\dir", "C:\\\\dir"), ("x\ny", "x\\x0Ay"),
        ("tab\there\r\n", "tab\\x09here\\x0D\\x0A"), ("\x7f", "\\x7F"), ("\\\\", "\\\\\\\\"),
        ("=1+1", "=1+1"), ("0012", "0012"), ("é €", "é €"), ("", ""),
    ],
)
def test_escape_1_round_trips(text: str, written: str) -> None:
    assert escape(text) == written
    assert unescape(written) == text
    assert "\n" not in written and "\r" not in written


def test_the_schema_line_the_header_and_one_line_per_row(tmp_path: Path) -> None:
    assessment, path = _write(_odd_pair(), tmp_path)
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == DETAIL_SCHEMA_LINE == "# esma-milan data quality detail; schema 1; escape 1"
    assert lines[1] == ",".join(DETAIL_COLUMNS)
    assert lines[-1] == ""  # the file ends with a line end
    # A line break in a cell is escaped, so every CSV record is one line.
    assert len(lines) - 3 == assessment.results.height
    assert not list(tmp_path.glob("*.part"))


def test_values_of_the_file_are_written_exactly_and_read_back(tmp_path: Path) -> None:
    _, path = _write(_odd_pair(), tmp_path)
    _, rows = _read(path)
    by_field: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["file"] == "loans" and row["record"] == "1" and row["raw_value"]:
            by_field.setdefault(row["field"], row)
    expected = {
        "RREL22": ("A\\x1FB", "escaped", "A\x1fB"),
        "RREL79": ("A\\\\x1FB", "escaped", "A\\x1FB"),
        "RREL82": ("C:\\\\dir", "escaped", "C:\\dir"),
        "RREL78": ("x\\x0Ay", "escaped", "x\ny"),
    }
    for field, (written, encoding, original) in expected.items():
        row = by_field[field]
        assert (row["raw_value"], row["raw_encoding"]) == (written, encoding), field
        assert unescape(row["raw_value"]) == original
    # Identifiers are text as written: a leading "=", leading zeros, 40 digits.
    loan_row = by_field["RREL22"]
    assert (loan_row["loan_identifier_original"], loan_row["loan_identifier_new"]) == ("=1+1", "=1+1")
    assert loan_row["obligor_identifier_new"] == "0012"
    assert loan_row["collateral_identifier_new"] == "" and loan_row["population"] == "Active, retained"
    collateral = [r for r in rows if r["file"] == "collateral" and r["record"]]
    assert {r["collateral_identifier_new"] for r in collateral if r["record"] == "1"} == {DIGITS_40}
    assert {r["loan_identifier_new"] for r in collateral if r["record"] == "1"} == {"=1+1"}
    # A collateral record shows the population of the loan it links to.
    assert {r["population"] for r in collateral if r["record"] == "1"} == {"Active, retained"}
    plain = next(r for r in rows if r["field"] == "RREL40" and r["rule_id"] == "P5")
    assert (plain["raw_value"], plain["raw_encoding"], plain["basis"], plain["severity"]) == (
        "abc", "plain", "E", "Warn",
    )
    assert plain["finding_id"] == "P5:violation:loans:RREL40"


def test_every_finding_row_is_there_exactly_once_and_in_order(tmp_path: Path) -> None:
    assessment, path = _write(_odd_pair(), tmp_path)
    _, rows = _read(path)

    def key(record: object, position: object, *rest: object) -> tuple[str, ...]:
        return tuple("" if v is None else str(v) for v in (record, position, *rest))

    written = Counter(
        key(r["record"], r["column_position"], r["file"], r["rule_id"], r["result_kind"], r["reason"],
            unescape(r["raw_value"]))
        for r in rows
    )
    held = Counter(
        key(r["record"], r["column_position"], r["file"], r["rule_id"], r["result_kind"], r["reason"],
            r["raw_value"])
        for r in assessment.results.to_dicts()
    )
    assert written == held and max(written.values()) == 1
    # Ordered by file, then record (rows about a column or a file first), column and rule.
    order = {"loans": 0, "collateral": 1, "pool": 2}
    keys = [
        (order[r["file"]], int(r["record"] or 0), int(r["column_position"] or 0), r["rule_id"])
        for r in rows
    ]
    assert keys == sorted(keys)
    assert [r["file"] for r in rows if r["file"] == "pool"]  # pool rows are in the file too


def test_counts_is_the_catalogue_flag_on_a_critical_field(tmp_path: Path) -> None:
    pair = Pair().set("RREL30", "", record=1).set("RREL40", "", record=1)
    _, path = _write(pair, tmp_path)
    _, rows = _read(path)
    flags = {(r["field"], r["rule_id"]): r["counts"] for r in rows if r["rule_id"] == "P3c"}
    assert flags == {("RREL30", "P3c"): "yes", ("RREL40", "P3c"): "no"}  # RREL40 is not critical


def test_a_repaired_cell_shows_the_repaired_text_and_its_original_bytes(tmp_path: Path) -> None:
    loans, collaterals = Pair().write(tmp_path)
    loans.write_bytes(loans.read_bytes().replace(b"EXAMPLE BANK NV", b"EXAMPLE BANK N\xe9", 1))
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    path = tmp_path / "detail.csv"
    write_detail_csv(assessment, path)
    _, rows = _read(path)
    repaired = [r for r in rows if r["rule_id"] == "A6"]
    assert [(r["record"], r["field"], r["raw_value"], r["original_bytes_hex"]) for r in repaired] == [
        ("1", "RREL82", "EXAMPLE BANK N\u00e9", b"EXAMPLE BANK N\xe9".hex()),
    ]
    assert all(not r["original_bytes_hex"] for r in rows if r["rule_id"] != "A6" and r["field"] != "RREL82")


def test_the_file_from_a_store_equals_the_file_from_memory_in_small_windows(tmp_path: Path) -> None:
    folder = FIXTURES / "synthetic"
    in_memory = assess_files(folder / "loans.csv", folder / "collaterals.csv", started_at=STARTED)
    store = ResultStore.create(tmp_path / "work")
    try:
        stored = assess_files(
            folder / "loans.csv", folder / "collaterals.csv", started_at=STARTED, store=store
        )
        write_detail_csv(in_memory, tmp_path / "memory.csv")
        write_detail_csv(stored, tmp_path / "store.csv")
        assert (tmp_path / "store.csv").read_bytes() == (tmp_path / "memory.csv").read_bytes()
        # Windows of three records give the same rows in the same order.
        whole = pl.concat(list(detail_chunks(stored)))
        windowed = pl.concat(list(detail_chunks(stored, window=3)))
        assert windowed.equals(whole) and whole.height == in_memory.results.height
    finally:
        store.remove()


def test_the_dropped_loans_file(tmp_path: Path) -> None:
    pair = constructed_pool()
    pair.loans[1]["RREL3"] = "L0\x1f2"  # record 2 is lost at line 4: its identifier is escaped
    pair.loans[2]["RREL5"] = "=B03"
    loans, collaterals = pair.write(tmp_path)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    report = complete_report(assessment, pipeline=None, error=RuntimeError("x"))
    assert report.waterfall is not None
    path = tmp_path / "dropped.csv"
    assert write_dropped_loans_csv(report.waterfall.dropped, path) == 10
    schema, rows = _read(path)
    assert schema == DROPPED_SCHEMA_LINE == "# esma-milan dropped loans; schema 1; escape 1"
    assert list(rows[0]) == list(DROPPED_SCHEMA)
    assert [r["record"] for r in rows] == ["2", "3", "4", "5", "8", "9", "10", "11", "12", "14"]
    assert rows[0]["loan_identifier_new"] == "L0\\x1F2" and unescape(rows[0]["loan_identifier_new"]) == "L0\x1f2"
    assert rows[1]["obligor_identifier_new"] == "=B03" and rows[1]["balance"] == "-5"
    assert rows[8]["collateral_records"] == "5;6"
    assert {r["observed_or_predicted"] for r in rows} == {"predicted"}


def test_an_empty_dropped_loans_file_has_its_two_lines(tmp_path: Path) -> None:
    path = tmp_path / "dropped.csv"
    assert write_dropped_loans_csv(pl.DataFrame(schema=DROPPED_SCHEMA), path) == 0
    assert path.read_text(encoding="utf-8").split("\n") == [
        DROPPED_SCHEMA_LINE, ",".join(DROPPED_SCHEMA), "",
    ]
