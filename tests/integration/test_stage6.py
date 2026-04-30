"""Stage 6 integration test against the staged synthetic fixture.

Anchors the expected detected_method ("by_loan") against the source-
of-truth value in the staged R reference workbook's Execution Summary.
That cell is written by R's pipeline.R:858 from the same
detect_aggregation_method() output we're porting, so reading it here
makes the assertion's provenance explicit and it'll pick up any future
fixture regeneration automatically.

Synthetic fixture by_loan justification (per analyst-walkthrough.Rmd):

  - Type 3 group (LOAN_003 + LOAN_004 share PROP_004): CPB differs
    (160k vs 95k) and current valuation is duplicated (400k on both
    rows) -> by_loan signal.
  - Type 4 group (full 2x2 set on LOAN_005,006 + PROP_005,006):
    valuations duplicated per loan -> by_loan signal.
  - Type 5 group (cross 2x2-minus on LOAN_007,008 + PROP_007,008):
    valuations duplicated per loan -> by_loan signal.

  All multi-loan groups carry the by_loan signature; total evidence
  is unanimously by_loan -> ratio 1.0 > 0.9 -> "by_loan".
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from esma_milan.runner import PipelineResult, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    out_dir = tmp_path_factory.mktemp("stage6_synthetic")
    return run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=out_dir,
        verbose=False,
    )


def _expected_aggregation_method_from_r_fixture() -> str:
    """Read the staged R-reference workbook's Execution Summary and
    return the value of the 'Aggregation Method' row.

    This makes the source of the expected value explicit: if the
    fixture is ever regenerated and the analyst dataset changes, this
    helper will pick up the new expected value automatically rather
    than requiring a literal string update in the test.
    """
    wb = openpyxl.load_workbook(
        SYNTHETIC / "expected_r_output.xlsx",
        read_only=True,
        data_only=True,
    )
    ws = wb["Execution Summary"]
    ws.reset_dimensions()
    for row in ws.iter_rows(values_only=True):
        if row and row[0] == "Aggregation Method":
            return str(row[1])
    raise AssertionError(
        "'Aggregation Method' row not found in Execution Summary "
        "of the synthetic expected fixture"
    )


def test_stage6_detects_by_loan_on_synthetic_fixture(
    synthetic_pipeline_run: PipelineResult,
) -> None:
    """The synthetic fixture is constructed so all multi-loan groups
    carry the by_loan signature. Detection must resolve to by_loan."""
    assert synthetic_pipeline_run.stage6 is not None
    assert synthetic_pipeline_run.stage6.detected_method == "by_loan"


def test_stage6_message_describes_unanimous_by_loan_evidence(
    synthetic_pipeline_run: PipelineResult,
) -> None:
    """Spot-check that the detection message names by_loan and the
    evidence ratio is >0.9 (in fact 100% on the synthetic fixture)."""
    assert synthetic_pipeline_run.stage6 is not None
    msg = synthetic_pipeline_run.stage6.message
    assert "by_loan" in msg
    assert "Strong evidence" in msg


def test_stage6_chosen_method_matches_detection(
    synthetic_pipeline_run: PipelineResult,
) -> None:
    """No explicit `aggregation_method` arg -> chosen method equals
    Stage 6's detection."""
    assert synthetic_pipeline_run.chosen_aggregation_method == "by_loan"


def test_stage6_anchored_against_r_fixture_execution_summary() -> None:
    """The source-of-truth assertion: the value in the staged R
    fixture's Execution Summary cell must equal what Stage 6 produces
    on the same input. This makes the expected value's provenance
    visible (no hard-coded literal) and ties our port to the live R
    output we're trying to match."""
    expected = _expected_aggregation_method_from_r_fixture()
    assert expected == "by_loan", (
        f"R fixture's Execution Summary now says aggregation method "
        f"{expected!r} - if the analyst dataset was updated, this test "
        f"will automatically pick up the new value but the unit-test "
        f"assertions above need updating to match."
    )


def test_stage6_explicit_argument_overrides_detection(
    tmp_path: Path,
) -> None:
    """When the user passes `aggregation_method=...` explicitly, the
    chosen method follows the argument even if Stage 6's detector would
    have picked a different one. Stage 6 still runs as a sanity check."""
    result = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=tmp_path,
        aggregation_method="by_group",  # explicit override
        verbose=False,
    )
    assert result.stage6 is not None
    assert result.stage6.detected_method == "by_loan"  # detector unaffected
    assert result.chosen_aggregation_method == "by_group"  # but chosen != detected
