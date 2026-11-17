"""The data quality report workbook.

Mirrors nothing in the R reference (specification 10.3). It is a file of
its own beside the MILAN workbook; the MILAN workbook is not touched.

    DQ Summary        section, item, value: the run, the judgments, the
                      waterfall, the files written, provenance
    DQ Fields         the expected-field inventory, one row a field
    DQ Columns        the actual-column inventory, one row a column
    DQ Checks         every catalogue rule with its four counts
    DQ Findings       one row per finding group, with its sentence
    DQ Rows           the rows of the detail CSV (continued on DQ Rows 2,
                      ... past a sheet's row limit)
    DQ Dropped loans  the rows of the dropped-loans CSV

Written with openpyxl in write-only mode. Every sheet has a bold header, a
frozen first row and a filter.

**Text stays text.** Every value that comes from the user's files is
written as a text cell: an identifier of digits keeps its leading zeros,
and a value that starts with `=` is text, not a formula (openpyxl would
make it a formula; the cell's type is set). No cell of this workbook is a
formula. A control character cannot be stored in a workbook, so text is
shown escaped exactly as in the detail CSV (`detail_csv.escape`). Text
longer than a cell can hold is cut, with its length and the name of the
detail CSV, which has it whole.

**The cut.** A workbook of millions of rows takes many minutes to write and
few programs open it. Above `DQ_XLSX_MAX_DETAIL_ROWS` finding rows, `DQ
Rows` holds the first rows of each finding group in proportion to the
group's size, every group keeping at least one, and `DQ Summary` says how
many rows are included, how many there are and where the complete list is.
Only this rendering is cut: the detail CSV always holds every row, and
every other sheet is complete.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from esma_milan.config import DQ_XLSX_MAX_DETAIL_ROWS
from esma_milan.quality.catalogue import RULES
from esma_milan.quality.detail_csv import DETAIL_COLUMNS, detail_chunks, dropped_frame, escape
from esma_milan.quality.waterfall import DROPPED_SCHEMA

if TYPE_CHECKING:
    from openpyxl.worksheet._write_only import WriteOnlyWorksheet

    from esma_milan.quality.assess import FileAssessment

SHEETS: tuple[str, ...] = (
    "DQ Summary", "DQ Fields", "DQ Columns", "DQ Checks", "DQ Findings", "DQ Rows",
    "DQ Dropped loans",
)
# Excel's limits: rows in a sheet (`runner.EXCEL_MAX_ROWS`) and characters in a cell.
SHEET_MAX_ROWS = 1_048_576
CELL_MAX_CHARS = 32_767

_BOLD = Font(bold=True)
_NEEDS_ESCAPE = frozenset(chr(c) for c in (*range(0x20), 0x7F)) | {"\\"}
# Detail columns that are numbers in the workbook; every other one is text.
_NUMERIC_DETAIL = ("record", "physical_line", "column_position")
_NUMERIC_DROPPED = ("record", "physical_line", "waterfall_step")


def _text(value: str, detail_name: str) -> str:
    """Text as a cell can hold it: escaped, and cut at the cell limit."""
    if not _NEEDS_ESCAPE.isdisjoint(value):
        value = escape(value)
    if len(value) > CELL_MAX_CHARS:
        note = f"... ({len(value)} characters; the full value is in {detail_name})"
        value = value[: CELL_MAX_CHARS - len(note)] + note
    return value


class _Book:
    """A write-only workbook whose every string is written as text."""

    def __init__(self, detail_name: str) -> None:
        self.workbook = Workbook(write_only=True)
        self.detail_name = detail_name
        self.sheets: list[str] = []

    def sheet(self, title: str, header: Sequence[str], widths: Sequence[int] = ()) -> WriteOnlyWorksheet:
        sheet = self.workbook.create_sheet(title)
        self.sheets.append(title)
        sheet.freeze_panes = "A2"
        for index in range(1, len(header) + 1):
            width = widths[index - 1] if index <= len(widths) else 18
            sheet.column_dimensions[get_column_letter(index)].width = width
        cells = []
        for name in header:
            cell = WriteOnlyCell(sheet, value=name)
            cell.font = _BOLD
            cells.append(cell)
        sheet.append(cells)
        return sheet

    def cell(self, sheet: WriteOnlyWorksheet, value: object) -> object:
        """One value as openpyxl takes it. A string is always a text cell."""
        if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            return value
        text = _text(value if isinstance(value, str) else str(value), self.detail_name)
        if text.startswith("="):
            cell = WriteOnlyCell(sheet, value=text)
            cell.data_type = "s"
            return cell
        return text

    def rows(self, sheet: WriteOnlyWorksheet, rows: Iterable[Sequence[object]]) -> int:
        written = 0
        for row in rows:
            sheet.append([self.cell(sheet, value) for value in row])
            written += 1
        return written

    @staticmethod
    def finish(sheet: WriteOnlyWorksheet, columns: int, data_rows: int) -> None:
        sheet.auto_filter.ref = f"A1:{get_column_letter(columns)}{data_rows + 1}"


def _flat(value: object) -> str:
    """A nested value of the summary as one line of text."""
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return "; ".join(f"{k}: {_flat(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "; ".join(_flat(v) for v in value)
    return str(value)


def _measure(line: Mapping[str, Any]) -> tuple[object, str]:
    balance = line.get("balance_by_currency")
    if not isinstance(balance, Mapping):
        return line.get("records"), ""
    return line.get("records"), "; ".join(
        f"{currency}: {figure['balance']} ({figure['records']} record(s))"
        for currency, figure in balance.items()
    )


def summary_rows(summary: Mapping[str, Any], cut: Mapping[str, object]) -> Iterator[tuple[object, ...]]:
    """The `DQ Summary` sheet: (section, item, value, balance, basis)."""
    def row(section: str, item: str, value: object, balance: str = "", basis: str = "") -> tuple[object, ...]:
        return (section, item, value if isinstance(value, (int, float)) else _flat(value), balance, basis)

    pipeline = summary["pipeline"]
    yield row("Run", "execution", summary["execution"])
    yield row("Run", "pipeline completed", pipeline["completed"])
    for key in ("statement", "error_type", "stage", "reason_code", "message"):
        if pipeline.get(key) is not None:
            yield row("Run", f"pipeline {key.replace('_', ' ')}", pipeline[key])
    yield row("Run", "assessment", summary["assessment"])
    for block in summary["files"]:
        yield row("Files assessed", f"{block['file']}: name", block["name"])
        yield row("Files assessed", f"{block['file']}: readable", block["readable"])
        if block.get("records") is not None:
            yield row("Files assessed", f"{block['file']}: records", block["records"])
            yield row("Files assessed", f"{block['file']}: columns", block["columns"])
        if "unreadable" in block:
            yield row("Files assessed", f"{block['file']}: not readable", block["unreadable"]["detail"])

    violations = summary["compliance"]["violations"]
    for key, value in violations.items():
        yield row("Compliance", f"violations: {key.replace('_', ' ')}", value)
    for key, value in summary["compliance"]["coverage"].items():
        yield row("Compliance", f"coverage: {key.replace('_', ' ')}", value)
    yield row("Plausibility", "observations", summary["plausibility"]["observations"])
    completeness = summary["assessment_completeness"]
    yield row("Completeness", "state", completeness["state"])
    yield row("Completeness", "statement", completeness["statement"])
    yield row("Completeness", "rules not implemented in this release",
              completeness["rules_not_implemented_in_this_release"]["count"])
    for rule, part in completeness["rules_partly_implemented"].items():
        yield row("Completeness", f"partly implemented: {rule}", part)
    for rule, block in completeness["rules_not_evaluable_on_these_files"].items():
        yield row("Completeness", f"not evaluable on these files: {rule}", block)
    for key in ("policy_outcome", "release_permitted", "override", "lineage"):
        yield row("Later releases", key.replace("_", " "), summary[key])

    waterfall = summary["waterfall"]
    if isinstance(waterfall, Mapping):
        for note in ("statement", "balance_note", "priority_note"):
            yield row("Waterfall", note.replace("_", " "), waterfall[note])
        for line in waterfall["lines"]:
            records, balance = _measure(line)
            yield row("Waterfall", f"{line['line']}. {line['label']}", records, balance, str(line["basis"]))
            for part in line.get("parts", []):
                records, balance = _measure(part)
                order = f"{line['line']}.{part['order']} " if "order" in part else "   "
                yield row("Waterfall", f"{order}{part['label']}", records, balance, str(part.get("basis", "")))
                inner = part.get("of_which_status_not_recognised")
                if inner is not None:
                    records, balance = _measure(inner)
                    yield row("Waterfall", "      of which: status not recognised", records, balance)
        for step, state in waterfall["reconciliation"].items():
            yield row("Waterfall", f"reconciliation at {step.replace('_', ' ')}", state)
        for label, count in waterfall["collateral_counts"].items():
            yield row("Collateral", label, count)
    else:
        yield row("Waterfall", "waterfall", waterfall)
    dropped = summary["dropped_loans"]
    if isinstance(dropped, Mapping):
        yield row("Dropped loans", "loans", dropped["loans"])
        for line in dropped["by_line"]:
            yield row("Dropped loans", f"line {line['waterfall_step']}: {line['assigned_line']}",
                      line["loans"], "", line["basis"])
    links = summary["links"]
    if isinstance(links, Mapping):
        for label, figures in links["by_convention"].items():
            yield row("Links", label, figures)
        for key in ("collateral_records_matching_no_loan_record", "loan_records_with_no_collateral_record",
                    "note", "stage3_comparison"):
            if key in links:
                yield row("Links", key.replace("_", " "), links[key])
    populations = summary["populations"]
    if isinstance(populations.get("counts"), Mapping):
        for label, count in populations["counts"].items():
            yield row("Populations", label, count)
        yield row("Populations", "labels", populations.get("labels"))
    identity = summary["identity_confidence"]
    yield row("Identity", "selected loan identifier", identity["selected_loan_identifier"])
    for label, count in identity["counts"].items():
        yield row("Identity", label, count)

    yield row("This workbook", "finding rows on DQ Rows", cut["rows_included"])
    yield row("This workbook", "finding rows in the detail CSV", cut["rows_total"])
    yield row("This workbook", "statement", cut["statement"])
    for artifact in summary.get("artifacts", []):
        yield row("Files written", str(artifact.get("kind")), artifact.get("name") or artifact.get("state"))
    detail_files = summary.get("detail_files")
    if isinstance(detail_files, list):  # the files written before this workbook
        for entry in detail_files:
            yield row("Files written", str(entry.get("kind")), entry.get("name") or entry.get("state"))
    for statement in summary.get("scope_statements", []):
        yield row("Outside the scope of two files", "statement", statement)
    provenance = summary["provenance"]
    for key in ("assessment_id", "assessment_started_utc", "as_of_date", "package_version",
                "rule_set_version", "esma_package_version", "taxonomy_sha256",
                "reference_lists_fetched", "options"):
        yield row("Provenance", key.replace("_", " "), provenance.get(key))
    for item in provenance["inputs"]:
        yield row("Provenance", f"{item['file']} sha256", item["sha256"])
    yield row("Provenance", "summary schema version", summary["schema_version"])


_FIELD_COLUMNS: tuple[str, ...] = (
    "file", "field", "name", "taxonomy_format", "nd1_4_allowed", "nd5_allowed", "optional_element",
    "critical", "present", "columns", "value_states", "value_states_retained_pool",
    "format_conformity", "distinct_present_values", "code_distribution", "surrounding_whitespace",
    "results", "rules_passed", "worst_severity_by_basis", "pipeline_use_and_lineage",
)
_COLUMN_COLUMNS: tuple[str, ...] = (
    "file", "position", "header", "resolved_as", "field", "pipeline_column_name", "records", "empty",
    "distinct_values", "longest_value", "sample_values", "statement",
)
_CHECK_COLUMNS: tuple[str, ...] = (
    "rule_id", "title", "basis", "severity", "status", "applicable", "evaluated", "failed",
    "not_evaluable", "other_result_kinds", "label", "not_evaluable_reasons", "applies_to",
    "regulatory_source", "counts_on_critical_fields", "not_implemented_part",
)
_FINDING_COLUMNS: tuple[str, ...] = (
    "finding_id", "sentence", "rule_id", "result_kind", "basis", "severities", "file", "field",
    "rows", "records", "enters_compliance_judgment", "defect_key", "reasons",
)


def _table(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> Iterator[list[object]]:
    for item in rows:
        yield [
            value if isinstance(value, (int, float)) and not isinstance(value, bool) else _flat(value)
            for value in (item.get(name) for name in columns)
        ]


def _checks(summary: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every catalogue rule: the ones that ran with their counts, the rest
    with their status."""
    ran = {rule["rule_id"]: rule for rule in summary["rules"]}
    for rule in RULES.values():
        if rule.rule_id in ran:
            yield ran[rule.rule_id]
        else:
            yield {
                "rule_id": rule.rule_id, "title": rule.title, "basis": rule.basis,
                "severity": rule.severity, "status": rule.status, "applies_to": rule.applies_to,
                "label": "not evaluated: not implemented in this release",
                "not_implemented_part": rule.not_implemented_part,
            }


def detail_quotas(counts: pl.DataFrame, limit: int) -> dict[str, int] | None:
    """How many rows of each finding group `DQ Rows` holds: None when every
    row fits; else each group's share of `limit` in proportion to its
    size, and at least one row."""
    groups = counts.group_by("rule_id", "result_kind", "file", "field").agg(pl.col("rows").sum())
    total = int(groups["rows"].sum())
    if total <= limit:
        return None
    return {
        f"{rule}:{kind}:{file}:{'None' if field is None else field}": max(1, rows * limit // total)
        for rule, kind, file, field, rows in groups.iter_rows()
    }


def _limited(chunks: Iterator[pl.DataFrame], quotas: dict[str, int] | None) -> Iterator[pl.DataFrame]:
    """The first rows of each finding group, up to its quota."""
    if quotas is None:
        yield from chunks
        return
    taken: dict[str, int] = dict.fromkeys(quotas, 0)
    for chunk in chunks:
        seen = chunk.group_by("finding_id").len()
        allowance = pl.DataFrame({
            "finding_id": seen["finding_id"],
            "_left": [quotas.get(g, 0) - taken.get(g, 0) for g in seen["finding_id"].to_list()],
        }, schema={"finding_id": pl.String, "_left": pl.Int64})
        kept = (
            chunk.join(allowance, on="finding_id", how="left", maintain_order="left")
            .filter(pl.int_range(pl.len()).over("finding_id") < pl.col("_left"))
            .drop("_left")
        )
        for group, n in kept.group_by("finding_id").len().iter_rows():
            taken[group] = taken.get(group, 0) + n
        if kept.height:
            yield kept


def _row_sheets(
    book: _Book, title: str, columns: Sequence[str], numeric: Sequence[str],
    chunks: Iterator[pl.DataFrame], sheet_rows: int, widths: Sequence[int] = (),
) -> int:
    """Write frames to `title`, continuing on `title 2`, ... when a sheet
    is full. Returns the number of data rows written."""
    index, in_sheet, total = 1, 0, 0
    sheet = book.sheet(title, columns, widths)
    text_columns = [c for c in columns if c not in numeric]
    for chunk in chunks:
        # Rows that need a cell of their own kind: a leading "=", or text too long for a cell.
        special = pl.any_horizontal([
            (pl.col(c).str.starts_with("=") | (pl.col(c).str.len_chars() > CELL_MAX_CHARS)).fill_null(False)
            for c in text_columns
        ])
        prepared = chunk.select(
            pl.col(c).cast(pl.Int64) if c in numeric else pl.col(c).cast(pl.String) for c in columns
        ).with_columns(special.alias("_special"))
        for *values, is_special in prepared.iter_rows():
            if in_sheet == sheet_rows:
                book.finish(sheet, len(columns), in_sheet)
                index, in_sheet = index + 1, 0
                sheet = book.sheet(f"{title} {index}", columns, widths)
            sheet.append([book.cell(sheet, v) for v in values] if is_special else values)
            in_sheet += 1
            total += 1
    book.finish(sheet, len(columns), in_sheet)
    return total


def write_report_workbook(
    summary: Mapping[str, Any],
    assessment: FileAssessment,
    dropped: pl.DataFrame | None,
    path: Path,
    *,
    detail_name: str,
    max_detail_rows: int = DQ_XLSX_MAX_DETAIL_ROWS,
    sheet_rows: int = SHEET_MAX_ROWS - 1,
) -> dict[str, object]:
    """Write the report workbook to `path` (through a temporary name).
    `summary` is the summary document; `dropped` the dropped-loans list.
    Returns what `DQ Rows` holds: `rows_included`, `rows_total`, `cut`
    and the statement printed on `DQ Summary`."""
    counts = assessment.counts
    total = int(counts["rows"].sum())
    quotas = detail_quotas(counts, max_detail_rows)
    included = total if quotas is None else sum(quotas.values())
    cut: dict[str, object] = {
        "rows_included": included, "rows_total": total, "cut": quotas is not None,
        "statement": f"DQ Rows holds every finding row; {detail_name} holds the same rows."
        if quotas is None else
        f"DQ Rows holds {included} of {total} finding rows: the first rows of each finding group, "
        f"in proportion to the group's size and at least one of each. Every row is in {detail_name}.",
    }

    book = _Book(detail_name)
    sheet = book.sheet("DQ Summary", ("section", "item", "value", "balance", "basis"), (26, 58, 60, 44, 16))
    book.finish(sheet, 5, book.rows(sheet, summary_rows(summary, cut)))

    sheet = book.sheet("DQ Fields", _FIELD_COLUMNS, (10, 12, 40))
    book.finish(sheet, len(_FIELD_COLUMNS), book.rows(sheet, _table(summary["expected_fields"], _FIELD_COLUMNS)))

    columns = [
        {"file": file, **column} for file, rows in summary["actual_columns"].items() for column in rows
    ]
    sheet = book.sheet("DQ Columns", _COLUMN_COLUMNS, (10, 10, 40, 20))
    book.finish(sheet, len(_COLUMN_COLUMNS), book.rows(sheet, _table(columns, _COLUMN_COLUMNS)))

    sheet = book.sheet("DQ Checks", _CHECK_COLUMNS, (10, 70, 8, 10, 22))
    book.finish(sheet, len(_CHECK_COLUMNS), book.rows(sheet, _table(_checks(summary), _CHECK_COLUMNS)))

    sheet = book.sheet("DQ Findings", _FINDING_COLUMNS, (34, 110))
    book.finish(sheet, len(_FINDING_COLUMNS), book.rows(sheet, _table(summary["findings"], _FINDING_COLUMNS)))

    written = _row_sheets(
        book, "DQ Rows", DETAIL_COLUMNS, _NUMERIC_DETAIL,
        _limited(detail_chunks(assessment), quotas), sheet_rows, (34, 10, 12, 8, 10, 12),
    )
    cut["rows_included"] = written
    listed = dropped if dropped is not None else pl.DataFrame(schema=DROPPED_SCHEMA)
    _row_sheets(
        book, "DQ Dropped loans", tuple(DROPPED_SCHEMA), _NUMERIC_DROPPED,
        iter([dropped_frame(listed)]), sheet_rows,
    )

    temporary = path.with_name(path.name + ".part")
    try:
        book.workbook.save(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    cut["sheets"] = list(book.sheets)
    return cut


__all__ = ["SHEETS", "detail_quotas", "summary_rows", "write_report_workbook"]
