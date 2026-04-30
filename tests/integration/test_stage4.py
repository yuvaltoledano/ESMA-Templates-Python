"""Stage 4 integration test against the staged synthetic fixture.

Pins the bipartite-graph output for the synthetic fixture's 5 group
structures (Type 1..5).

Synthetic-fixture edges (after Stage 3 picks NEW_* loan IDs and PORIG_*
property IDs):

  Group 1 (Type 1: 1L:1P):  NEW_001 ↔ PORIG_01
  Group 2 (Type 2: 1L:2P):  NEW_002 ↔ {PORIG_02, PORIG_03}
  Group 3 (Type 3: 2L:1P):  {NEW_003, NEW_004} ↔ PORIG_04
  Group 4 (Type 4: 2L:2P):  {NEW_005, NEW_006} ↔ {PORIG_05, PORIG_06}
                            (full matrix, 4 edges)
  Group 5 (Type 5: 2L:2P):  {NEW_007, NEW_008} ↔ {PORIG_07, PORIG_08}
                            (3 edges, missing NEW_007 ↔ PORIG_08)

Total: 12 edges, 5 components, 8 loan nodes, 8 property nodes. The
deterministic group-id labelling sorts components by their lex-smallest
node; all components have a NEW_* loan as their smallest member, so:

  min(NEW_001, ...) = NEW_001  -> group 1
  min(NEW_002, ...) = NEW_002  -> group 2
  min(NEW_003, ...) = NEW_003  -> group 3
  min(NEW_005, ...) = NEW_005  -> group 4
  min(NEW_007, ...) = NEW_007  -> group 5
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

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


def test_synthetic_yields_five_groups(stage4_synthetic: GraphResult) -> None:
    """Five connected components, one per structure type."""
    assert stage4_synthetic.loan_groups["collateral_group_id"].n_unique() == 5
    assert stage4_synthetic.collateral_groups["collateral_group_id"].n_unique() == 5


def test_synthetic_has_eight_loan_nodes_and_eight_property_nodes(
    stage4_synthetic: GraphResult,
) -> None:
    assert stage4_synthetic.loan_groups.height == 8
    assert stage4_synthetic.collateral_groups.height == 8


def test_synthetic_has_twelve_edges(stage4_synthetic: GraphResult) -> None:
    assert stage4_synthetic.edges_with_group.height == 12


def test_synthetic_loan_to_group_assignment(stage4_synthetic: GraphResult) -> None:
    """Pins the deterministic group-id for every loan, exercising the
    sorted-min-node canonicalisation."""
    pairs = dict(
        zip(
            stage4_synthetic.loan_groups["calc_loan_id"].to_list(),
            stage4_synthetic.loan_groups["collateral_group_id"].to_list(),
            strict=True,
        )
    )
    assert pairs == {
        "NEW_001": 1,
        "NEW_002": 2,
        "NEW_003": 3,
        "NEW_004": 3,
        "NEW_005": 4,
        "NEW_006": 4,
        "NEW_007": 5,
        "NEW_008": 5,
    }


def test_synthetic_property_to_group_assignment(stage4_synthetic: GraphResult) -> None:
    pairs = dict(
        zip(
            stage4_synthetic.collateral_groups["calc_property_id"].to_list(),
            stage4_synthetic.collateral_groups["collateral_group_id"].to_list(),
            strict=True,
        )
    )
    assert pairs == {
        "PORIG_01": 1,
        "PORIG_02": 2,
        "PORIG_03": 2,
        "PORIG_04": 3,
        "PORIG_05": 4,
        "PORIG_06": 4,
        "PORIG_07": 5,
        "PORIG_08": 5,
    }


def test_synthetic_type_5_group_has_three_edges(stage4_synthetic: GraphResult) -> None:
    """Type 5 (cross-collateralised) is identified by edge count < L*P.
    Group 5 has 2 loans, 2 properties, 3 edges (one missing) - this
    pins that the graph builder didn't synthesise the missing edge."""
    type5_edges = stage4_synthetic.edges_with_group.filter(
        # Group 5 contains NEW_007 / NEW_008.
        # Use the loan_exposure_id to identify the group.
    )
    n_edges_in_g5 = stage4_synthetic.edges_with_group.filter(
        stage4_synthetic.edges_with_group["collateral_group_id"] == 5
    ).height
    assert n_edges_in_g5 == 3
    _ = type5_edges  # silence unused


def test_synthetic_type_4_group_has_four_edges(stage4_synthetic: GraphResult) -> None:
    """Type 4 (full set): every loan-property combination present."""
    n_edges_in_g4 = stage4_synthetic.edges_with_group.filter(
        stage4_synthetic.edges_with_group["collateral_group_id"] == 4
    ).height
    assert n_edges_in_g4 == 4
