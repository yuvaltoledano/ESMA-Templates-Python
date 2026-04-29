"""Layer-1 parity test against the anonymised real RMBS pool.

Same shape as test_parity_synthetic, but runs against the larger fixture
(1,127 loans / 1,665 properties / 820 collateral groups). Skipped at the
module level until Stage 1 is implemented; flip SKIP_REAL_FIXTURE to False
when the synthetic pool is fully green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esma_milan.config import OUTPUT_SHEET_ORDER
from esma_milan.parity import diff_workbooks, format_report
from esma_milan.runner import run_pipeline

from .conftest import ParityFixture

# Flip to False once synthetic parity is fully green.
SKIP_REAL_FIXTURE: bool = True
SKIP_REASON: str = "real-fixture parity blocked until synthetic parity is green"


@pytest.fixture(scope="module")
def real_run_output(real_fixture: ParityFixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("real_run")
    result = run_pipeline(
        loans_file_path=real_fixture.loans,
        collaterals_file_path=real_fixture.collaterals,
        taxonomy_file_path=real_fixture.taxonomy,
        deal_name="REAL_ANONYMISED",
        output_dir=out_dir,
        verbose=False,
    )
    assert result.output_path is not None
    return result.output_path


@pytest.mark.parity
@pytest.mark.skipif(SKIP_REAL_FIXTURE, reason=SKIP_REASON)
@pytest.mark.parametrize("sheet_name", OUTPUT_SHEET_ORDER)
def test_real_sheet_parity(
    sheet_name: str,
    real_run_output: Path,
    real_fixture: ParityFixture,
) -> None:
    report = diff_workbooks(real_run_output, real_fixture.expected_output)
    sheet_diff = next((s for s in report.sheet_diffs if s.sheet == sheet_name), None)
    assert sheet_diff is not None
    assert sheet_diff.passed, f"\n{format_report(report)}\n"
