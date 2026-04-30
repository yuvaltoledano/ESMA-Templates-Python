"""Stage 6.5 integration test: Sheet 6 + Sheet 7 cell-by-cell parity.

Layered:

  1. Date encoding contract: read the Python output back via openpyxl
     and assert that date-typed columns surface as `int` (Excel serials),
     not `datetime.date` objects. R's openxlsx writes dates as serial
     ints; openpyxl's native date support would produce `datetime.date`,
     which would fail parity. This test makes the encoding choice
     explicit at integration level.

  2. Column-order pinning: the column sequence in the written workbook
     must match what the R fixture produces, byte-for-byte.

  3. Full parity: the diff library's report for Sheets 6 and 7 must
     come back PASS (which the parity-test harness already runs as part
     of `test_synthetic_sheet_parity[Cleaned ESMA loans/properties]`,
     but this module's anchor pins the bytes one level deeper to keep
     the failure mode self-explanatory if anything regresses).
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from esma_milan.parity import diff_workbooks
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("stage6_5")
    result = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=out_dir,
        verbose=False,
    )
    assert result.output_path is not None
    return result.output_path


def _read_sheet(path: Path, sheet_name: str) -> tuple[tuple[object, ...], ...]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    ws.reset_dimensions()
    return tuple(ws.iter_rows(values_only=True))


# ---------------------------------------------------------------------------
# Date-encoding contract
# ---------------------------------------------------------------------------


def test_cleaned_loans_origination_date_round_trips_as_int(
    synthetic_output_path: Path,
) -> None:
    """ORIG_001 has origination_date=2019-03-15 in the source CSV. The
    R fixture encodes that as Excel serial 43539 read back by openpyxl
    as `int`. The Python writer must produce the same int, NOT a
    datetime.date object."""
    rows = _read_sheet(synthetic_output_path, "Cleaned ESMA loans")
    header = rows[0]
    orig_date_col_idx = header.index("origination_date")
    loan_id_col_idx = header.index("original_underlying_exposure_identifier")
    row_for_orig_001 = next(
        r for r in rows[1:] if r[loan_id_col_idx] == "ORIG_001"
    )
    value = row_for_orig_001[orig_date_col_idx]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"origination_date is {type(value).__name__} (value={value!r}) - "
        f"expected `int` Excel serial. R's openxlsx writes dates as bare "
        f"numerics (no number format), so openpyxl reads them as ints. "
        f"If the writer was switched to openpyxl's native date support, "
        f"this would surface as datetime.date and break parity."
    )
    assert value == 43539


def test_cleaned_loans_null_date_round_trips_as_missing(
    synthetic_output_path: Path,
) -> None:
    """ORIG_001's date_last_in_arrears is NA in the source CSV. The
    written cell must be empty / missing on read (None or absent),
    matching the R fixture's behaviour."""
    rows = _read_sheet(synthetic_output_path, "Cleaned ESMA loans")
    header = rows[0]
    date_col_idx = header.index("date_last_in_arrears")
    loan_id_col_idx = header.index("original_underlying_exposure_identifier")
    row_for_orig_001 = next(
        r for r in rows[1:] if r[loan_id_col_idx] == "ORIG_001"
    )
    # openpyxl returns None for embedded empty cells. Trailing empty cells
    # may produce a shorter tuple; in either case the value at this index
    # must read as None when bounds-checked.
    value = (
        row_for_orig_001[date_col_idx]
        if date_col_idx < len(row_for_orig_001)
        else None
    )
    assert value is None


def test_cleaned_properties_valuation_dates_are_ints(
    synthetic_output_path: Path,
) -> None:
    """Same encoding applies to Cleaned ESMA properties: current/original
    valuation dates round-trip as ints."""
    rows = _read_sheet(synthetic_output_path, "Cleaned ESMA properties")
    header = rows[0]
    cur_idx = header.index("current_valuation_date")
    orig_idx = header.index("original_valuation_date")
    pid_idx = header.index("calc_property_id")
    row_p1 = next(r for r in rows[1:] if r[pid_idx] == "PORIG_01")
    assert isinstance(row_p1[cur_idx], int) and not isinstance(row_p1[cur_idx], bool)
    assert isinstance(row_p1[orig_idx], int) and not isinstance(row_p1[orig_idx], bool)


# ---------------------------------------------------------------------------
# Column-order pinning
# ---------------------------------------------------------------------------


def test_cleaned_loans_column_order_matches_r_fixture(
    synthetic_output_path: Path,
) -> None:
    actual_header = _read_sheet(synthetic_output_path, "Cleaned ESMA loans")[0]
    expected_header = _read_sheet(
        SYNTHETIC / "expected_r_output.xlsx", "Cleaned ESMA loans"
    )[0]
    assert actual_header == expected_header


def test_cleaned_properties_column_order_matches_r_fixture(
    synthetic_output_path: Path,
) -> None:
    actual_header = _read_sheet(
        synthetic_output_path, "Cleaned ESMA properties"
    )[0]
    expected_header = _read_sheet(
        SYNTHETIC / "expected_r_output.xlsx", "Cleaned ESMA properties"
    )[0]
    assert actual_header == expected_header


# ---------------------------------------------------------------------------
# Full cell-by-cell parity (anchors what the SHEET_STATUS gate already
# checks but with explicit error messages tied to Stage 6.5)
# ---------------------------------------------------------------------------


def test_cleaned_loans_parity_against_r_fixture(synthetic_output_path: Path) -> None:
    rep = diff_workbooks(
        synthetic_output_path, SYNTHETIC / "expected_r_output.xlsx"
    )
    sheet_diff = next(
        s for s in rep.sheet_diffs if s.sheet == "Cleaned ESMA loans"
    )
    assert sheet_diff.passed, (
        "Cleaned ESMA loans parity failed:\n"
        + "\n".join(d.short() for d in sheet_diff.cell_diffs[:10])
        + (f"\n... ({len(sheet_diff.cell_diffs) - 10} more)" if len(sheet_diff.cell_diffs) > 10 else "")
    )


def test_cleaned_properties_parity_against_r_fixture(
    synthetic_output_path: Path,
) -> None:
    rep = diff_workbooks(
        synthetic_output_path, SYNTHETIC / "expected_r_output.xlsx"
    )
    sheet_diff = next(
        s for s in rep.sheet_diffs if s.sheet == "Cleaned ESMA properties"
    )
    assert sheet_diff.passed, (
        "Cleaned ESMA properties parity failed:\n"
        + "\n".join(d.short() for d in sheet_diff.cell_diffs[:10])
    )
