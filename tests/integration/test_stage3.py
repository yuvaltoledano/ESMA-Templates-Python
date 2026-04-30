"""Stage 3 integration test against the staged synthetic fixture.

Pins the calc_loan_id / calc_borrower_id / calc_property_id values that
fall out of Stage 3 against the staged synthetic fixture.

Synthetic-fixture inputs:
  loans:        original_underlying_exposure_identifier = ORIG_001..008
                new_underlying_exposure_identifier      = NEW_001..008
  collaterals:  underlying_exposure_identifier          = NEW_001..008
                                                          (NEW_002, 005,
                                                           006, 007, 008
                                                           appear twice)

So:
  - select_calc_loan_id Gate 2: ORIG_*  coverage = 0/8 = 0%,
                                NEW_*   coverage = 8/8 = 100%.
                                Gate 3: NEW wins (full coverage).
  - calc_loan_id = NEW_001..008
  - generate_id_column on (BORIG_01..06 vs BNEW_01..06): tied at 6
    unique values, defaults to original. calc_borrower_id = BORIG_*.
  - generate_id_column on (PORIG_01..08 vs PNEW_01..08): tied at 8
    unique values, defaults to original. calc_property_id = PORIG_*.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from esma_milan.pipeline.filters import Stage2Output, run_stage2
from esma_milan.pipeline.identifiers import Stage3Output, run_stage3
from esma_milan.pipeline.stage1 import Stage1Output, run_stage1

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def stage1_synthetic() -> Stage1Output:
    return run_stage1(
        loans_path=SYNTHETIC / "loans.csv",
        collaterals_path=SYNTHETIC / "collaterals.csv",
        taxonomy_path=SYNTHETIC / "taxonomy.xlsx",
    )


@pytest.fixture(scope="module")
def stage2_synthetic(stage1_synthetic: Stage1Output) -> Stage2Output:
    return run_stage2(stage1_synthetic.loans, stage1_synthetic.properties)


@pytest.fixture(scope="module")
def stage3_synthetic(stage2_synthetic: Stage2Output) -> Stage3Output:
    # NEW_* has full coverage; the original column has zero coverage,
    # so select_calc_loan_id picks NEW with no warning under default
    # min_coverage = 0.85.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return run_stage3(stage2_synthetic.loans, stage2_synthetic.properties)


def test_stage3_calc_loan_id_picks_new_column(
    stage3_synthetic: Stage3Output,
) -> None:
    """NEW_* covers 100% of property loan refs; ORIG_* covers 0%.
    select_calc_loan_id must pick NEW."""
    assert sorted(stage3_synthetic.loans["calc_loan_id"].to_list()) == [
        f"NEW_00{i}" for i in range(1, 9)
    ]


def test_stage3_calc_borrower_id_defaults_to_original_on_tied_unique_count(
    stage3_synthetic: Stage3Output,
) -> None:
    """6 unique values in each obligor column -> generate_id_column
    tie-break picks original. The synthetic CSV deliberately gives
    LOAN_003+LOAN_004 a shared borrower (BORIG_03) and LOAN_005+LOAN_006
    a shared borrower (BORIG_04), so we expect duplicates in the output."""
    expected = ["BORIG_01", "BORIG_02", "BORIG_03", "BORIG_03", "BORIG_04", "BORIG_04", "BORIG_05", "BORIG_06"]
    assert stage3_synthetic.loans["calc_borrower_id"].to_list() == expected


def test_stage3_calc_property_id_defaults_to_original_on_tied_unique_count(
    stage3_synthetic: Stage3Output,
) -> None:
    """8 unique values in each collateral-id column -> generate_id_column
    tie-break picks original."""
    actual = sorted(set(stage3_synthetic.properties["calc_property_id"].to_list()))
    assert actual == [f"PORIG_0{i}" for i in range(1, 9)]


def test_stage3_loan_count_unchanged_for_synthetic(
    stage3_synthetic: Stage3Output,
) -> None:
    """All 8 active loans pre-Stage-3 reference NEW_* IDs that exist in
    the property table, so the intersection filter is a no-op."""
    assert stage3_synthetic.loans.height == 8


def test_stage3_property_count_unchanged_for_synthetic(
    stage3_synthetic: Stage3Output,
) -> None:
    """All 12 properties post-Stage-2 reference loans whose calc_loan_id
    survives Stage 3, so the intersection filter is a no-op."""
    assert stage3_synthetic.properties.height == 12


def test_stage3_loans_carry_calc_loan_id_and_calc_borrower_id(
    stage3_synthetic: Stage3Output,
) -> None:
    assert "calc_loan_id" in stage3_synthetic.loans.columns
    assert "calc_borrower_id" in stage3_synthetic.loans.columns


def test_stage3_properties_carry_calc_property_id(
    stage3_synthetic: Stage3Output,
) -> None:
    assert "calc_property_id" in stage3_synthetic.properties.columns
