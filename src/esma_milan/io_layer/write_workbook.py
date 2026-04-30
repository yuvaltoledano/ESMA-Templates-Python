"""10-sheet output workbook writer (openpyxl write-only mode).

Mirrors the workbook layout in r_reference/R/pipeline.R:903-933 - ten
sheets in canonical order, each empty-or-populated according to which
pipeline stages have produced data so far.

Cell-level styling (bold + #D9D9D9 fill on header, left-align on data,
autofilter, column widths = 25) is omitted while stages stack up, since
the parity diff library reads values-only and compares values, not
formatting. Full styling lands with the Stage-10 writer.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import polars as pl

from esma_milan.config import OUTPUT_SHEET_ORDER


def write_pipeline_workbook(
    path: Path,
    *,
    populated_sheets: dict[str, pl.DataFrame] | None = None,
) -> None:
    """Write a 10-sheet workbook in canonical order.

    Each entry in `populated_sheets` (sheet_name -> Polars DataFrame)
    contributes a header row followed by one row per DataFrame row.
    Sheet names not present in `populated_sheets` are created empty so
    the parity harness's sheet-order check still passes.

    Polars dtypes are written via `iter_rows()` which surfaces native
    Python types (int / float / bool / str / date / datetime). openpyxl
    serialises these directly into Excel.
    """
    wb = openpyxl.Workbook(write_only=True)
    populated = populated_sheets or {}

    for sheet_name in OUTPUT_SHEET_ORDER:
        ws = wb.create_sheet(sheet_name)
        df = populated.get(sheet_name)
        if df is None:
            continue
        ws.append(list(df.columns))
        for row in df.iter_rows():
            ws.append(list(row))

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# Compatibility shim: the original API took no populated-sheets dict.
# Removed once nothing imports it.
def write_stub_workbook(path: Path) -> None:
    """Deprecated: call write_pipeline_workbook(path) instead."""
    write_pipeline_workbook(path)
