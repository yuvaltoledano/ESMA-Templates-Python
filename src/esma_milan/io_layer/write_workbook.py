"""10-sheet output workbook writer (openpyxl write-only mode).

The no-op stub here writes an empty workbook with the canonical sheet names
and nothing else, so the parity harness can detect every cell as a diff
against the staged expected_r_output.xlsx. Full implementation lands in
Stage 10.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

from esma_milan.config import OUTPUT_SHEET_ORDER

HEADER_FILL = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
HEADER_FONT = Font(bold=True)
LEFT_ALIGN = Alignment(horizontal="left", vertical="center")


def write_stub_workbook(path: Path) -> None:
    """Write an empty 10-sheet workbook with the canonical sheet names.

    Used by the no-op runner.py so the parity harness can demonstrate
    failure against the staged expected_r_output.xlsx. Full implementation
    in Stage 10.
    """
    wb = openpyxl.Workbook(write_only=True)
    for sheet_name in OUTPUT_SHEET_ORDER:
        wb.create_sheet(sheet_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
