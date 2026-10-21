"""Differential tests for the two input-robustness fixes in Stage 1.

There is no R here, so none of these can have an R-generated expected
workbook. Each builds a variant of the synthetic fixture in `tmp_path`
that would have crashed `read_and_clean` before the fix, runs the
pipeline on it, and checks either the cleaned Stage-1 frame (padded
pools, whose data differs from the fixture) or a cell-by-cell
`diff_workbooks` comparison against the run on the clean original
(encoding variants, whose data is identical once repaired).

1. Late decimal past row 10,000: a balance column that is whole numbers
   for >10,000 rows and then a .50 value. Sample-based inference typed it
   Int64 and the first decimal raised `ComputeError`.
2. Numeric field ND for >10,000 rows then populated. readr guesses
   `logical` from the sample and nulls the later value; the Python port
   reads it (documented deviation from R).
3. Windows-1252 input: `Müller` in `originator_name` (RREL82,
   {ALPHANUM-100}; MILAN column 1 "Originator Identifier" and a
   "Cleaned ESMA loans" column) re-encoded as cp1252, plus the same
   with a Latin-1-only byte. Polars raised `invalid utf-8 sequence`.

No fixture files are added; variants live only in `tmp_path`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
import polars as pl
import pytest
import structlog

from esma_milan.parity import diff_workbooks, format_report
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"

# Suffixing these per padding block keeps every loan, borrower and
# property unique while preserving each block's loan/property structure.
_LOAN_ID_COLS = (
    "original_underlying_exposure_identifier",
    "new_underlying_exposure_identifier",
    "original_obligor_identifier",
    "new_obligor_identifier",
)
_PROPERTY_ID_COLS = (
    "underlying_exposure_identifier",
    "original_collateral_identifier",
    "new_collateral_identifier",
)

# 1,251 blocks x 8 loans = 10,008 rows, past readr's / the old Polars
# sample of 10,000.
_BLOCKS_PAST_SAMPLE = 1_251
_MIN_ROWS = 10_001


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


def _write_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _padded_fixture(
    out_dir: Path,
    *,
    edit_loans: dict[str, str] | None = None,
    edit_last_loan: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Repeat the synthetic loans and collaterals with unique IDs per block.

    `edit_loans` sets columns on every loan row; `edit_last_loan` sets
    columns on the very last loan row only.
    """
    header_l, loans = _read_rows(SYNTHETIC / "loans.csv")
    header_c, props = _read_rows(SYNTHETIC / "collaterals.csv")
    padded_loans: list[dict[str, str]] = []
    padded_props: list[dict[str, str]] = []
    for k in range(_BLOCKS_PAST_SAMPLE):
        suffix = f"_B{k:04d}"
        for row in loans:
            r = dict(row)
            for c in _LOAN_ID_COLS:
                r[c] = r[c] + suffix
            if edit_loans:
                r.update(edit_loans)
            padded_loans.append(r)
        for row in props:
            r = dict(row)
            for c in _PROPERTY_ID_COLS:
                r[c] = r[c] + suffix
            padded_props.append(r)
    if edit_last_loan:
        padded_loans[-1].update(edit_last_loan)
    assert len(padded_loans) >= _MIN_ROWS
    loans_path = out_dir / "loans.csv"
    props_path = out_dir / "collaterals.csv"
    _write_rows(loans_path, header_l, padded_loans)
    _write_rows(props_path, header_c, padded_props)
    return loans_path, props_path


def _run(loans: Path, collaterals: Path, out_dir: Path | None, deal: str = "SYN") -> Path | None:
    result = run_pipeline(
        loans_file_path=loans,
        collaterals_file_path=collaterals,
        deal_name=deal,
        output_dir=out_dir,
        dry_run=out_dir is None,
        verbose=False,
    )
    return result.output_path


# ---------------------------------------------------------------------------
# 1 + 2: type inference past the 10,000-row sample
# ---------------------------------------------------------------------------


def test_late_decimal_past_row_10000_no_longer_crashes(tmp_path: Path) -> None:
    """`current_principal_balance` is whole numbers for 10,007 rows and
    98000.50 on the last. Before the fix: `ComputeError: could not parse
    '98000.50' as dtype i64`. After: Float64 throughout, value intact."""
    loans, props = _padded_fixture(
        tmp_path, edit_last_loan={"current_principal_balance": "98000.50"}
    )
    result = run_pipeline(
        loans_file_path=loans,
        collaterals_file_path=props,
        deal_name="PADDED",
        output_dir=tmp_path / "out",
        dry_run=True,
        verbose=False,
    )
    assert result.stage1 is not None
    cleaned = result.stage1.loans
    assert cleaned.height >= _MIN_ROWS
    assert cleaned.schema["current_principal_balance"] == pl.Float64
    assert cleaned["current_principal_balance"][-1] == 98000.5
    assert cleaned["current_principal_balance"][0] == 180000.0
    # The late-decimal loan is active and survives to the flattened pool.
    assert result.stage7 is not None
    last_id = cleaned["new_underlying_exposure_identifier"][-1]
    flat = result.stage7.combined_flattened.filter(pl.col("calc_loan_id") == last_id)
    assert flat.height == 1
    assert flat["current_principal_balance"][0] == 98000.5


def test_numeric_field_nd_for_10000_rows_then_populated(tmp_path: Path) -> None:
    """`prior_principal_balances` ({MONETARY}, not one of the four
    structural columns) is ND on every row except the last. Before the
    fix Polars inferred String from the sample and the cast was never
    attempted; readr (R) guesses `logical` from its 10,000-row sample and
    silently nulls the populated value. Here the column is Float64 and
    the last row's value survives: a documented deviation from R."""
    loans, props = _padded_fixture(
        tmp_path,
        edit_loans={"prior_principal_balances": "ND"},
        edit_last_loan={"prior_principal_balances": "12345.67"},
    )
    result = run_pipeline(
        loans_file_path=loans,
        collaterals_file_path=props,
        deal_name="PADDED",
        output_dir=tmp_path / "out",
        dry_run=True,
        verbose=False,
    )
    assert result.stage1 is not None
    col = result.stage1.loans["prior_principal_balances"]
    assert col.dtype == pl.Float64
    assert col.null_count() == col.len() - 1
    assert col[-1] == 12345.67


# ---------------------------------------------------------------------------
# 3: non-numeric value in a numeric field is a hard error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "code"),
    [
        ("current_principal_balance", "RREL30"),  # structural
        ("payment_due", "RREL39"),  # non-structural {MONETARY}
    ],
)
def test_non_numeric_value_in_numeric_field_is_hard_error(
    tmp_path: Path, column: str, code: str
) -> None:
    header, rows = _read_rows(SYNTHETIC / "loans.csv")
    rows[2][column] = "1.234,56"  # data row 3
    loans = tmp_path / "loans.csv"
    _write_rows(loans, header, rows)
    with pytest.raises(ValueError) as excinfo:
        _run(loans, SYNTHETIC / "collaterals.csv", None)
    msg = str(excinfo.value)
    assert "loans.csv" in msg
    assert code in msg and column in msg
    assert "data row 3" in msg
    assert "'1.234,56'" in msg


# ---------------------------------------------------------------------------
# 4: non-UTF-8 input, differential against the UTF-8 original (the
# header-repair case, 5, is a unit test in tests/unit/test_read_csv.py)
# ---------------------------------------------------------------------------

_ORIGINATOR = "Example Dutch Bank N.V."


def _loans_with_originator(name: str) -> str:
    text = (SYNTHETIC / "loans.csv").read_text(encoding="utf-8")
    assert _ORIGINATOR in text
    return text.replace(_ORIGINATOR, name)


def _assert_identical_workbooks(actual: Path, expected: Path) -> None:
    report = diff_workbooks(actual, expected)
    assert report.passed, f"\n{format_report(report)}\n"


def _milan_column(path: Path, column: str) -> list[object]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb["MILAN template pool"]
        ws.reset_dimensions()
        rows = ws.iter_rows(values_only=True)
        header = list(next(rows))
        idx = header.index(column)
        return [r[idx] for r in rows]
    finally:
        wb.close()


def test_windows_1252_loans_file_matches_utf8_run(tmp_path: Path) -> None:
    """The cp1252-encoded file produces a workbook cell-identical to the
    UTF-8 file carrying the same `Müller` value, and one repair warning."""
    text = _loans_with_originator("Müller Bank N.V.")
    utf8 = tmp_path / "utf8" / "loans.csv"
    cp1252 = tmp_path / "cp1252" / "loans.csv"
    utf8.parent.mkdir()
    cp1252.parent.mkdir()
    utf8.write_bytes(text.encode("utf-8"))
    cp1252.write_bytes(text.encode("cp1252"))
    assert utf8.read_bytes() != cp1252.read_bytes()

    expected = _run(utf8, SYNTHETIC / "collaterals.csv", tmp_path / "out_utf8")
    with structlog.testing.capture_logs() as logs:
        actual = _run(cp1252, SYNTHETIC / "collaterals.csv", tmp_path / "out_cp1252")
    assert expected is not None and actual is not None

    repaired = [e for e in logs if e["event"] == "csv_encoding_repaired"]
    assert len(repaired) == 1
    assert repaired[0]["file"] == "loans.csv"
    assert repaired[0]["n_fields_repaired"] == 8  # one originator_name cell per loan

    _assert_identical_workbooks(actual, expected)
    # The repaired value really reaches the output: MILAN column 1 and Sheet 6.
    assert set(_milan_column(actual, "Originator Identifier")) == {"Müller Bank N.V."}


def test_latin1_only_byte_matches_utf8_run_with_preserved_code_point(tmp_path: Path) -> None:
    """0x81 is undefined in Windows-1252, so the cascade's Latin-1 pass
    keeps it as U+0081 (not stripped), as `test_sanitize_utf8.py` pins
    and as R's iconv(latin1) does. The workbook must equal a UTF-8 run
    whose originator name carries that same code point."""
    name = "Bank \u0081 N.V."
    text = _loans_with_originator(name)
    utf8 = tmp_path / "utf8" / "loans.csv"
    latin1 = tmp_path / "latin1" / "loans.csv"
    utf8.parent.mkdir()
    latin1.parent.mkdir()
    utf8.write_bytes(text.encode("utf-8"))
    latin1.write_bytes(text.encode("latin-1"))
    assert b"\x81" in latin1.read_bytes()

    expected = _run(utf8, SYNTHETIC / "collaterals.csv", tmp_path / "out_utf8")
    actual = _run(latin1, SYNTHETIC / "collaterals.csv", tmp_path / "out_latin1")
    assert expected is not None and actual is not None
    _assert_identical_workbooks(actual, expected)
    assert set(_milan_column(actual, "Originator Identifier")) == {name}

