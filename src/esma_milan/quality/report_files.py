"""The four report files, written beside the workbook.

Mirrors nothing in the R reference (specification 10.3).

    <stem> data quality detail.csv     every finding row
    <stem> dropped loans.csv           every loan lost at line 4 or 6
    <stem> data quality report.xlsx    the report workbook
    <stem> data quality summary.json   the summary document

`<stem>` is the MILAN workbook's name without its extension. Each file is
written to a temporary name and renamed, so a file with a final name is
always whole. A file that cannot be written is recorded in the summary
with the reason and the others are still written. The summary is written
last, so its `detail_files` block states what exists.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from esma_milan.quality.assess import QualityReport
from esma_milan.quality.detail_csv import (
    DETAIL_SCHEMA_LINE,
    DROPPED_SCHEMA_LINE,
    write_detail_csv,
    write_dropped_loans_csv,
)
from esma_milan.quality.export import SUMMARY_SUFFIX, summary_dict, summary_filename, write_json
from esma_milan.quality.report_workbook import write_report_workbook

DETAIL_SUFFIX = " data quality detail.csv"
DROPPED_SUFFIX = " dropped loans.csv"
WORKBOOK_SUFFIX = " data quality report.xlsx"

FaultHandler = Callable[[str, Exception], None]


def report_filenames(deal_name: str, cutoff: str | None) -> dict[str, str]:
    """The four names, by kind."""
    summary = summary_filename(deal_name, cutoff)
    stem = summary.removesuffix(SUMMARY_SUFFIX)
    return {
        "detail": stem + DETAIL_SUFFIX, "dropped_loans": stem + DROPPED_SUFFIX,
        "report_workbook": stem + WORKBOOK_SUFFIX, "summary": summary,
    }


def write_report_files(
    report: QualityReport,
    folder: Path,
    deal_name: str,
    cutoff: str | None,
    on_fault: FaultHandler | None = None,
) -> Path:
    """Write the four files into `folder` and return the summary's path.
    `on_fault(step, exception)` is called for a file that could not be
    written; the fault never propagates from here, except from the
    summary itself."""
    folder.mkdir(parents=True, exist_ok=True)
    names = report_filenames(deal_name, cutoff)
    files: list[dict[str, object]] = []

    def attempt(kind: str, step: str, entry: dict[str, object], write: Callable[[], dict[str, object]]) -> None:
        path = folder / names[kind]
        entry.update(name=names[kind], kind=kind.replace("_", " "))
        try:
            entry.update(write())
            entry.update(state="available", size=path.stat().st_size)
        except Exception as exc:
            entry.update(state="failed", reason=f"{type(exc).__name__}: {exc}")
            if on_fault is not None:
                on_fault(step, exc)
        files.append(entry)

    attempt(
        "detail", "write_detail_csv",
        {"media_type": "text/csv", "schema_line": DETAIL_SCHEMA_LINE},
        lambda: {"rows": write_detail_csv(report.assessment, folder / names["detail"])},
    )
    waterfall = report.waterfall
    if waterfall is None:
        files.append({
            "name": None, "kind": "dropped loans", "state": "not available",
            "reason": "the loans file could not be split into records",
        })
    else:
        dropped = waterfall.dropped
        attempt(
            "dropped_loans", "write_dropped_loans_csv",
            {"media_type": "text/csv", "schema_line": DROPPED_SCHEMA_LINE},
            lambda: {"rows": write_dropped_loans_csv(dropped, folder / names["dropped_loans"])},
        )

    summary = summary_dict(report, detail_files=files)

    def workbook() -> dict[str, object]:
        cut = write_report_workbook(
            summary, report.assessment, waterfall.dropped if waterfall is not None else None,
            folder / names["report_workbook"], detail_name=names["detail"],
        )
        return {
            "sheets": cut["sheets"], "finding_rows_included": cut["rows_included"],
            "finding_rows_total": cut["rows_total"], "statement": cut["statement"],
        }

    attempt(
        "report_workbook", "write_report_workbook",
        {"media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        workbook,
    )
    path = folder / names["summary"]
    write_json(summary, path)
    return path


__all__ = [
    "DETAIL_SUFFIX", "DROPPED_SUFFIX", "WORKBOOK_SUFFIX", "report_filenames", "write_report_files",
]
