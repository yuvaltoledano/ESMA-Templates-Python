"""Layer-1 parity test: Python output vs staged R reference, synthetic pool.

Runs the pipeline once per session and diffs the resulting workbook against
tests/fixtures/synthetic/expected_r_output.xlsx, sheet by sheet.

While stages are still being ported, every sheet that has not yet been
implemented is xfailed via the SHEET_STATUS table below. As stages land we
flip entries from "pending" to "passing"; CI-level parity is the union of
all sheets being "passing".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from esma_milan.config import OUTPUT_SHEET_ORDER
from esma_milan.parity import diff_workbooks, format_report
from esma_milan.runner import run_pipeline

from .conftest import ParityFixture

SheetStatus = Literal["pending", "passing"]

# As stages port over, flip a sheet's status from "pending" -> "passing".
# A "pending" sheet is xfailed (we expect a diff against the no-op stub);
# a "passing" sheet must match cell-for-cell or the test fails.
SHEET_STATUS: dict[str, SheetStatus] = {
    "Execution Summary": "pending",
    "Loans to properties": "pending",
    "Properties to loans": "pending",
    "Borrowers to loans": "pending",
    "Borrowers to properties": "pending",
    "Cleaned ESMA loans": "pending",
    "Cleaned ESMA properties": "pending",
    "Group classifications": "pending",
    "Combined flattened pool": "pending",
    "MILAN template pool": "pending",
}


@pytest.fixture(scope="module")
def synthetic_run_output(synthetic_fixture: ParityFixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("synthetic_run")
    result = run_pipeline(
        loans_file_path=synthetic_fixture.loans,
        collaterals_file_path=synthetic_fixture.collaterals,
        taxonomy_file_path=synthetic_fixture.taxonomy,
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=out_dir,
        verbose=False,
    )
    assert result.output_path is not None, "run_pipeline returned None outside dry_run"
    return result.output_path


@pytest.mark.parity
@pytest.mark.parametrize("sheet_name", OUTPUT_SHEET_ORDER)
def test_synthetic_sheet_parity(
    sheet_name: str,
    synthetic_run_output: Path,
    synthetic_fixture: ParityFixture,
) -> None:
    """Diff one sheet of the Python output against the R expected output."""
    report = diff_workbooks(synthetic_run_output, synthetic_fixture.expected_output)
    sheet_diff = next((s for s in report.sheet_diffs if s.sheet == sheet_name), None)
    assert sheet_diff is not None, f"sheet {sheet_name!r} not present in diff report"

    if SHEET_STATUS[sheet_name] == "pending":
        if sheet_diff.passed:
            pytest.fail(
                f"Sheet {sheet_name!r} is marked 'pending' but parity passes! "
                "Flip SHEET_STATUS to 'passing' in this file."
            )
        pytest.xfail(f"sheet {sheet_name!r} not yet implemented (status=pending)")

    # status == "passing": must match exactly.
    assert sheet_diff.passed, f"\n{format_report(report)}\n"


@pytest.mark.parity
def test_synthetic_workbook_structure(
    synthetic_run_output: Path,
    synthetic_fixture: ParityFixture,
) -> None:
    """Sheet names and order must match exactly, regardless of content."""
    report = diff_workbooks(synthetic_run_output, synthetic_fixture.expected_output)
    assert report.actual_sheet_order == report.expected_sheet_order, (
        f"sheet order mismatch:\n  actual:   {list(report.actual_sheet_order)}\n"
        f"  expected: {list(report.expected_sheet_order)}"
    )
