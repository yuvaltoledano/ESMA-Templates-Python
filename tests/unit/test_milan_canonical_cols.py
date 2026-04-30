"""Sanity checks on MILAN_EXPECTED_OUTPUT_COLS.

These match against the synthetic fixture so any future drift in the
canonical column list is caught immediately.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from esma_milan.pipeline.milan_map import MILAN_EXPECTED_OUTPUT_COLS

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_EXPECTED = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "expected_r_output.xlsx"


def test_milan_canonical_column_count() -> None:
    assert len(MILAN_EXPECTED_OUTPUT_COLS) == 175


def test_milan_canonical_columns_match_synthetic_fixture() -> None:
    if not SYNTHETIC_EXPECTED.exists():
        pytest.skip("synthetic fixture not present")
    wb = openpyxl.load_workbook(SYNTHETIC_EXPECTED, read_only=True, data_only=True)
    ws = wb["MILAN template pool"]
    ws.reset_dimensions()
    header = next(ws.iter_rows(values_only=True))
    assert tuple(header) == MILAN_EXPECTED_OUTPUT_COLS, (
        "MILAN_EXPECTED_OUTPUT_COLS drifted from the staged R fixture's header row"
    )
