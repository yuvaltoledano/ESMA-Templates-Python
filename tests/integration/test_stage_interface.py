"""The read-only stage interface on both fixtures.

The stages report their own decisions as data (which records were kept
and why the others were dropped, which identifier column was chosen, which
valuation was used). These tests check, on the synthetic and the real
fixture, that the reported data agrees with what the stages produced and
with the figures the stages already log.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from esma_milan.pipeline.filters import (
    _keep,
    collateral_drop_reasons,
    collect_active_loan_ids,
    loan_drop_reasons,
)
from esma_milan.runner import PipelineResult, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SYNTHETIC = FIXTURES / "synthetic"
REAL = FIXTURES / "real_anonymised"

FIXTURE_PARAMS = [
    pytest.param(SYNTHETIC, id="synthetic"),
    pytest.param(
        REAL,
        id="real_anonymised",
        marks=pytest.mark.skipif(
            not (REAL / "loans.csv").exists(),
            reason="Real-anonymised fixture not present in this checkout",
        ),
    ),
]

_RESULTS: dict[Path, PipelineResult] = {}


def _run(fixture_dir: Path) -> PipelineResult:
    """One dry run per fixture, shared by every test in the module."""
    if fixture_dir not in _RESULTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _RESULTS[fixture_dir] = run_pipeline(
                loans_file_path=fixture_dir / "loans.csv",
                collaterals_file_path=fixture_dir / "collaterals.csv",
                deal_name="INTERFACE",
                dry_run=True,
                verbose=True,
            )
    return _RESULTS[fixture_dir]


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage2_expressions_keep_exactly_the_stage2_output(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage1 is not None and result.stage2 is not None
    stage1, stage2 = result.stage1, result.stage2

    loans_kept = stage1.loans.filter(_keep(loan_drop_reasons()))
    assert loans_kept.equals(stage2.loans)
    active = collect_active_loan_ids(loans_kept)
    properties_kept = stage1.properties.filter(_keep(collateral_drop_reasons(active)))
    assert properties_kept.equals(stage2.properties)


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage2_rows_index_the_stage1_frames(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage1 is not None and result.stage2 is not None
    stage1, stage2 = result.stage1, result.stage2
    assert stage2.loan_rows is not None and stage2.property_rows is not None
    assert stage2.loan_drop_reasons is not None
    assert stage2.property_drop_reasons is not None

    assert stage1.loans[stage2.loan_rows].equals(stage2.loans)
    assert stage1.properties[stage2.property_rows].equals(stage2.properties)
    assert stage2.loan_rows.is_sorted() and stage2.property_rows.is_sorted()

    # One reasons row per Stage 1 record; kept exactly when none is true.
    assert stage2.loan_drop_reasons.height == stage1.loans.height
    assert stage2.property_drop_reasons.height == stage1.properties.height
    for reasons, rows in (
        (stage2.loan_drop_reasons, stage2.loan_rows),
        (stage2.property_drop_reasons, stage2.property_rows),
    ):
        no_reason = reasons.select(~pl.any_horizontal(pl.all())).to_series()
        assert no_reason.arg_true().to_list() == rows.to_list()


def test_real_fixture_drop_reason_counts() -> None:
    """The real fixture's Stage 2 losses: 169 redeemed loans with a zero
    balance, and the 288 collateral rows that reference them."""
    if not (REAL / "loans.csv").exists():
        pytest.skip("Real-anonymised fixture not present in this checkout")
    stage2 = _run(REAL).stage2
    assert stage2 is not None
    assert stage2.loan_drop_reasons is not None
    assert stage2.property_drop_reasons is not None
    assert {c: int(stage2.loan_drop_reasons[c].sum()) for c in stage2.loan_drop_reasons.columns} == {
        "status_not_active": 169,
        "balance_not_positive": 169,
    }
    assert {
        c: int(stage2.property_drop_reasons[c].sum())
        for c in stage2.property_drop_reasons.columns
    } == {
        "property_type_missing": 0,
        "collateral_type_missing": 0,
        "collateral_type_excluded": 0,
        "no_active_loan": 288,
        "no_positive_valuation": 0,
    }
