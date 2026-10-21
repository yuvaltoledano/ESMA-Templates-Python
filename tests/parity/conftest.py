"""Shared parity-test fixtures.

Each fixture pairs an input directory (loans.csv, collaterals.csv) with
the staged R-reference output that the Python pipeline must reproduce
cell-for-cell. The ESMA taxonomy is not a fixture: the pipeline reads the
copy bundled with the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures"


@dataclass(frozen=True)
class ParityFixture:
    name: str
    loans: Path
    collaterals: Path
    expected_output: Path

    def __post_init__(self) -> None:
        for p in (self.loans, self.collaterals, self.expected_output):
            if not p.exists():
                raise FileNotFoundError(f"parity fixture missing: {p}")


def _build_fixture(name: str) -> ParityFixture:
    base = FIXTURES_ROOT / name
    return ParityFixture(
        name=name,
        loans=base / "loans.csv",
        collaterals=base / "collaterals.csv",
        expected_output=base / "expected_r_output.xlsx",
    )


@pytest.fixture(scope="session")
def synthetic_fixture() -> ParityFixture:
    return _build_fixture("synthetic")


@pytest.fixture(scope="session")
def real_fixture() -> ParityFixture:
    return _build_fixture("real_anonymised")
