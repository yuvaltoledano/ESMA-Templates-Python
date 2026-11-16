"""The exact list of results on the two parity fixtures.

The fixtures are parity fixtures, not compliant examples, so they have
findings. `expected/<fixture>.csv` is the reviewed list: rule, result kind,
file, field and count, and no cell value. A rule is never weakened to make
a fixture quiet; a deliberate change to a rule changes these files in the
same PR.
"""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest

from esma_milan.quality import assess_files

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXPECTED = Path(__file__).resolve().parent / "expected"
HEADER = ["rule_id", "result_kind", "file", "field", "count"]


def result_counts(fixture: str) -> list[list[str]]:
    folder = FIXTURES / fixture
    results = assess_files(folder / "loans.csv", folder / "collaterals.csv").results
    counted = (
        results.select(pl.col(c).cast(pl.String) for c in HEADER[:4])
        .group_by(HEADER[:4]).len()
        .sort(HEADER[:4])
    )
    return [[*row[:4], str(row[4])] for row in counted.iter_rows()]


@pytest.mark.parametrize("fixture", ["synthetic", "real_anonymised"])
def test_results_equal_the_reviewed_list(fixture: str) -> None:
    with (EXPECTED / f"{fixture}.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == HEADER
    assert result_counts(fixture) == rows[1:]
