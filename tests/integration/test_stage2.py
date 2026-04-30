"""Stage 2 integration test against the staged synthetic fixture.

Confirms that on the synthetic CSV - all 8 loans active, all 12 properties
RRE with positive valuations - Stage 2 is a no-op (nothing dropped).
The dropped-loan branch coverage lives in tests/unit/test_filters.py
per the user's directive on Stage 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esma_milan.pipeline.filters import Stage2Output, run_stage2
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


def test_synthetic_all_loans_survive(stage2_synthetic: Stage2Output) -> None:
    """All 8 loans in the synthetic fixture have active status (7 PERF,
    1 ARRE) and CPB > 0. Stage 2 must keep all of them."""
    assert stage2_synthetic.loans.height == 8


def test_synthetic_all_properties_survive(stage2_synthetic: Stage2Output) -> None:
    """All 12 collateral rows in the synthetic fixture are RRE with
    valid valuations and reference active loans. Stage 2 must keep
    all of them."""
    assert stage2_synthetic.properties.height == 12
