"""Stage 5 integration test against the staged synthetic fixture.

Pins the (collateral_group_id, loans, collaterals, is_full_set,
structure_type) frame produced by Stage 5. Compared against what we
expect for the synthetic fixture's 5-group partition.

Synthetic fixture partition (built from the staged loans+collaterals
CSVs, see analyst-walkthrough.Rmd for the deliberate construction):

    gid 1: {NEW_001} - {PORIG_01}            -> Type 1 (1L, 1C)
    gid 2: {NEW_002} - {PORIG_02, PORIG_03}  -> Type 2 (1L, 2C)
    gid 3: {NEW_003, NEW_004} - {PORIG_04}   -> Type 3 (2L, 1C)
    gid 4: {NEW_005, NEW_006} - {PORIG_05, PORIG_06}    -> Type 4 (full set)
    gid 5: {NEW_007, NEW_008} - {PORIG_07, PORIG_08}    -> Type 5 (cross)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from esma_milan.pipeline.classification import (
    TYPE_1,
    TYPE_2,
    TYPE_3,
    TYPE_4,
    TYPE_5,
    Stage5Output,
    run_stage5,
)
from esma_milan.pipeline.filters import Stage2Output, run_stage2
from esma_milan.pipeline.graph import GraphResult, run_stage4
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
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return run_stage3(stage2_synthetic.loans, stage2_synthetic.properties)


@pytest.fixture(scope="module")
def stage4_synthetic(stage3_synthetic: Stage3Output) -> GraphResult:
    return run_stage4(stage3_synthetic.loans, stage3_synthetic.properties)


@pytest.fixture(scope="module")
def stage5_synthetic(stage4_synthetic: GraphResult) -> Stage5Output:
    return run_stage5(stage4_synthetic)


def test_stage5_emits_five_groups(stage5_synthetic: Stage5Output) -> None:
    assert stage5_synthetic.classifications.height == 5


def test_stage5_one_row_per_structure_type(stage5_synthetic: Stage5Output) -> None:
    """The synthetic fixture is deliberately built so each of the 5
    structure types appears exactly once."""
    types = stage5_synthetic.classifications["structure_type"].to_list()
    assert sorted(types) == sorted([TYPE_1, TYPE_2, TYPE_3, TYPE_4, TYPE_5])


def test_stage5_matches_expected_per_gid(
    stage5_synthetic: Stage5Output,
    stage4_synthetic: GraphResult,
) -> None:
    """Pin each gid -> (loans, collaterals, is_full_set, structure_type)
    against the deliberate construction of the synthetic fixture."""
    df = stage5_synthetic.classifications

    # Stage 4 deterministically labels groups in lex-min-loan-id order:
    #   NEW_001 -> 1, NEW_002 -> 2, NEW_003 -> 3, NEW_005 -> 4, NEW_007 -> 5
    loan_groups = stage4_synthetic.loan_groups
    gid_for = {
        row["calc_loan_id"]: row["collateral_group_id"]
        for row in loan_groups.iter_rows(named=True)
    }
    assert gid_for["NEW_001"] == 1
    assert gid_for["NEW_002"] == 2
    assert gid_for["NEW_003"] == 3 and gid_for["NEW_004"] == 3
    assert gid_for["NEW_005"] == 4 and gid_for["NEW_006"] == 4
    assert gid_for["NEW_007"] == 5 and gid_for["NEW_008"] == 5

    by_gid = {row[0]: row for row in df.iter_rows()}
    assert by_gid[1] == (1, 1, 1, False, TYPE_1)
    assert by_gid[2] == (2, 1, 2, False, TYPE_2)
    assert by_gid[3] == (3, 2, 1, False, TYPE_3)
    assert by_gid[4] == (4, 2, 2, True, TYPE_4)
    assert by_gid[5] == (5, 2, 2, False, TYPE_5)


def test_stage5_dtypes_match_r_fixture(stage5_synthetic: Stage5Output) -> None:
    """Schema must match Sheet 8 in expected_r_output.xlsx exactly."""
    schema = stage5_synthetic.classifications.schema
    assert schema["collateral_group_id"] == pl.Int64
    assert schema["loans"] == pl.Int64
    assert schema["collaterals"] == pl.Int64
    assert schema["is_full_set"] == pl.Boolean
    assert schema["structure_type"] == pl.String


def test_stage5_sorted_by_gid(stage5_synthetic: Stage5Output) -> None:
    gids = stage5_synthetic.classifications["collateral_group_id"].to_list()
    assert gids == [1, 2, 3, 4, 5]
