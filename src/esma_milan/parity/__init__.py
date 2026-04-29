"""Parity comparison library.

Diffs two Excel workbooks (typically a Python output and a staged R
reference output) cell-by-cell and reports differences in a structured form.
Used by both pytest tests and scripts/run_parity_check.py.
"""

from esma_milan.parity.diff import (
    CellDiff,
    DiffReport,
    SheetDiff,
    diff_workbooks,
    format_report,
)

__all__ = [
    "CellDiff",
    "DiffReport",
    "SheetDiff",
    "diff_workbooks",
    "format_report",
]
