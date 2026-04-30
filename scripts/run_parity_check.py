#!/usr/bin/env python3
"""Diff a Python pipeline output against an expected R workbook.

    uv run python scripts/run_parity_check.py path/to/python.xlsx path/to/expected.xlsx

Exits 0 on parity, 1 on diff, 2 on file-level errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from esma_milan.parity import diff_workbooks, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="Python pipeline output (.xlsx)")
    parser.add_argument("expected", type=Path, help="R reference output (.xlsx)")
    parser.add_argument(
        "--max-diffs-per-sheet",
        type=int,
        default=50,
        help="Truncate per-sheet cell-diff lists at this many entries (default 50).",
    )
    parser.add_argument(
        "--max-shown",
        type=int,
        default=30,
        help="Maximum cell diffs printed per sheet (default 30).",
    )
    args = parser.parse_args(argv)

    try:
        report = diff_workbooks(
            args.actual,
            args.expected,
            max_cell_diffs_per_sheet=args.max_diffs_per_sheet,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(format_report(report, max_diffs_shown=args.max_shown))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
