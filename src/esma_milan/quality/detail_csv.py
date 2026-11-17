"""The two detail files: every finding row, and every dropped loan.

Mirrors nothing in the R reference (specification 10.3).

    <stem> data quality detail.csv    one row per finding row
    <stem> dropped loans.csv          one row per loan lost at line 4 or 6
                                      of the waterfall

Both are UTF-8 CSV with "\\n" line ends. The first line is a schema line,
the second the header:

    # esma-milan data quality detail; schema 1; escape 1

**Escape 1.** These files quote cells of the user's files, which can hold
anything. So that every CSV record is one line and nothing is lost, the
columns that carry text of the files (`ESCAPED_COLUMNS`) are written with:

    a backslash                       as two backslashes
    a character below U+0020, U+007F  as backslash, x and two hexadecimal
                                      digits (a line feed is \\x0A)

`raw_encoding` is `escaped` when a column of the row was changed by this
and `plain` when none was. To read a value back: left to right, `\\xNN`
is the character NN and `\\\\` is one backslash (`unescape`). Nothing else
is altered: a value that starts with `=` is written as it is, leading
zeros and long digit strings are kept, and nothing is trimmed. A cell
whose bytes were not valid UTF-8 shows the repaired text, and its original
bytes in `original_bytes_hex`.

The detail file is ordered by file (loans, collateral, pool), record (rows
about a whole column or file first), column position and rule. It is
written in windows of records read from the store, so the whole detail is
never in memory, and nothing is sampled or left out.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from esma_milan.quality.catalogue import FIELDS, RULES

if TYPE_CHECKING:
    from esma_milan.quality.assess import FileAssessment

DETAIL_SCHEMA_LINE = "# esma-milan data quality detail; schema 1; escape 1"
DROPPED_SCHEMA_LINE = "# esma-milan dropped loans; schema 1; escape 1"

DETAIL_COLUMNS: tuple[str, ...] = (
    "finding_id", "rule_id", "result_kind", "basis", "severity", "file", "record",
    "physical_line", "column_position", "header", "field", "value_state", "raw_value",
    "raw_encoding", "original_bytes_hex", "nd4_date", "expected", "reason", "note",
    "loan_identifier_original", "loan_identifier_new", "obligor_identifier_new",
    "collateral_identifier_new", "population", "counts",
)
# The columns that can carry text of the user's files.
ESCAPED_COLUMNS: tuple[str, ...] = (
    "header", "raw_value", "expected", "note", "loan_identifier_original", "loan_identifier_new",
    "obligor_identifier_new", "collateral_identifier_new",
)
DROPPED_ESCAPED_COLUMNS: tuple[str, ...] = (
    "loan_identifier_original", "loan_identifier_new", "obligor_identifier_new", "status", "balance",
    "currency",
)

FILE_ORDER: tuple[str, ...] = ("loans", "collateral", "pool")
WINDOW_RECORDS = 20000
"""Records read from the store at a time."""

_NEEDS_ESCAPE = r"[\\\x00-\x1f\x7f]"
_CONTROL = [chr(c) for c in (*range(0x20), 0x7F)]
_UNESCAPE = re.compile(r"\\(\\|x[0-9A-F]{2})")


def escape(text: str) -> str:
    """Escape 1, on one value."""
    out = text.replace("\\", "\\\\")
    for ch in _CONTROL:
        if ch in out:
            out = out.replace(ch, f"\\x{ord(ch):02X}")
    return out


def unescape(text: str) -> str:
    """The inverse of `escape`."""
    return _UNESCAPE.sub(lambda m: "\\" if m.group(1) == "\\" else chr(int(m.group(1)[1:], 16)), text)


def _escape_expr(column: str) -> pl.Expr:
    expr = pl.col(column).str.replace_all("\\", "\\\\", literal=True)
    for ch in _CONTROL:
        expr = expr.str.replace_all(ch, f"\\x{ord(ch):02X}", literal=True)
    return expr.alias(column)


def escape_frame(frame: pl.DataFrame, columns: tuple[str, ...], flag: str | None = None) -> pl.DataFrame:
    """Apply escape 1 to `columns`. Only the rows that need it are
    rewritten; the order of the rows is kept. With `flag`, a column of
    that name says `escaped` or `plain` for each row."""
    needs = pl.any_horizontal([pl.col(c).str.contains(_NEEDS_ESCAPE).fill_null(False) for c in columns])
    marked = frame.with_columns(needs.alias("_escape"))
    if flag is not None:
        marked = marked.with_columns(
            pl.when(pl.col("_escape")).then(pl.lit("escaped")).otherwise(pl.lit("plain")).alias(flag)
        )
    if not marked["_escape"].any():
        return marked.drop("_escape")
    indexed = marked.with_row_index("_order")
    dirty = indexed.filter(pl.col("_escape")).with_columns([_escape_expr(c) for c in columns])
    clean = indexed.filter(~pl.col("_escape"))
    return pl.concat([clean, dirty]).sort("_order").drop("_order", "_escape")


def _rule_columns() -> pl.DataFrame:
    """Per rule: its basis and whether its violations count on a critical field."""
    return pl.DataFrame({
        "rule_id": list(RULES),
        "basis": [r.basis for r in RULES.values()],
        "_rule_counts": [r.counts for r in RULES.values()],
    })


def _record_columns(assessment: FileAssessment) -> dict[str, pl.LazyFrame]:
    """Per file: record -> the identifiers and population the detail shows."""
    out: dict[str, pl.LazyFrame] = {}
    store = assessment.store
    text = pl.lit(None, dtype=pl.String)

    def scan(file: str) -> pl.LazyFrame | None:
        audit = assessment.loans if file == "loans" else assessment.collateral
        if audit.records is not None:
            return audit.records.lazy()
        if store is not None and store.has_records(audit.file):
            return store.scan_records(audit.file)
        return None

    loans = scan("loans")
    if loans is not None:
        out["loans"] = loans.select(
            "record",
            pl.col("RREL2").alias("loan_identifier_original"),
            pl.col("RREL3").alias("loan_identifier_new"),
            pl.col("RREL5").alias("obligor_identifier_new"),
            text.alias("collateral_identifier_new"),
            "population",
        )
    collateral = scan("collateral")
    if collateral is not None:
        links = assessment.scan_links()
        if links is not None and loans is not None:
            # The populations of the loans a collateral record links to.
            linked = (
                links.filter(pl.col("match") == "raw")
                .join(loans.select(pl.col("record").alias("loan_record"), "population"), on="loan_record")
                .group_by("collateral_record")
                .agg(pl.col("population").unique().sort().str.join("; ").alias("population"))
                .collect()
                .lazy()
            )
        else:
            linked = pl.LazyFrame(
                schema={"collateral_record": pl.UInt32, "population": pl.String}
            )
        out["collateral"] = collateral.select(
            "record",
            text.alias("loan_identifier_original"),
            pl.col("RREC2").alias("loan_identifier_new"),
            text.alias("obligor_identifier_new"),
            pl.col("RREC4").alias("collateral_identifier_new"),
        ).join(linked, left_on="record", right_on="collateral_record", how="left")
    return out


def _repairs(assessment: FileAssessment, file: str) -> pl.DataFrame:
    """Per file: (record, column_position) -> the original bytes of a repaired cell."""
    schema = {"record": pl.UInt32, "column_position": pl.UInt16, "original_bytes_hex": pl.String}
    audit = assessment.loans if file == "loans" else assessment.collateral if file == "collateral" else None
    if audit is None or audit.encoding_repaired.height == 0:
        return pl.DataFrame(schema=schema)
    return audit.encoding_repaired.select(
        pl.col("record").cast(pl.UInt32),
        pl.col("column_position").cast(pl.UInt16),
        "original_bytes_hex",
    ).unique(["record", "column_position"], keep="first")


def detail_chunks(assessment: FileAssessment, window: int = WINDOW_RECORDS) -> Iterator[pl.DataFrame]:
    """Every finding row with the columns of the detail file, escaped, in
    the file's order, in frames of at most `window` records."""
    results = assessment.scan_results().with_columns(
        pl.col("file").cast(pl.String), pl.col("rule_id").cast(pl.String),
        pl.col("result_kind").cast(pl.String), pl.col("field").cast(pl.String),
    )
    rules = _rule_columns()
    critical = sorted(code for code, entry in FIELDS.items() if entry.critical)
    record_columns = _record_columns(assessment)
    identifier_columns = (
        "loan_identifier_original", "loan_identifier_new", "obligor_identifier_new",
        "collateral_identifier_new", "population",
    )

    def finish(rows: pl.DataFrame, file: str) -> pl.DataFrame:
        info = record_columns.get(file)
        if info is not None and rows["record"].null_count() < rows.height:
            low, high = rows["record"].min(), rows["record"].max()
            rows = rows.join(
                info.filter(pl.col("record").is_between(low, high)).collect(), on="record", how="left"
            )
        else:
            rows = rows.with_columns([pl.lit(None, dtype=pl.String).alias(c) for c in identifier_columns])
        rows = (
            rows.join(_repairs(assessment, file), on=["record", "column_position"], how="left")
            .join(rules, on="rule_id", how="left")
            .with_columns(
                pl.format(
                    "{}:{}:{}:{}", "rule_id", "result_kind", "file", pl.col("field").fill_null("None")
                ).alias("finding_id"),
                pl.when(
                    (pl.col("result_kind") == "violation") & pl.col("_rule_counts")
                    & pl.col("field").is_in(critical)
                ).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("counts"),
                *[pl.col(c).cast(pl.String) for c in ("severity", "value_state", "header", "reason",
                                                      "expected", "note", "raw_value")],
            )
            .sort(
                ["record", "column_position", "rule_id", "field", "result_kind", "reason", "raw_value"],
                nulls_last=False, maintain_order=True,
            )
        )
        return escape_frame(rows, ESCAPED_COLUMNS, flag="raw_encoding").select(DETAIL_COLUMNS)

    for file in FILE_ORDER:
        of_file = results.filter(pl.col("file") == file)
        whole = of_file.filter(pl.col("record").is_null()).collect()
        if whole.height:
            yield finish(whole, file)
        last = of_file.select(pl.col("record").max()).collect().item()
        if last is None:
            continue
        for start in range(1, last + 1, window):
            rows = of_file.filter(pl.col("record").is_between(start, start + window - 1)).collect()
            if rows.height:
                yield finish(rows, file)


def _write(path: Path, schema_line: str, header: tuple[str, ...], chunks: Iterator[pl.DataFrame]) -> int:
    """Write to a temporary name beside `path` and rename, so a file with
    the final name is always whole. Returns the number of data rows."""
    temporary = path.with_name(path.name + ".part")
    rows = 0
    try:
        with temporary.open("wb") as handle:
            handle.write((schema_line + "\n" + ",".join(header) + "\n").encode("utf-8"))
            for chunk in chunks:
                chunk.write_csv(handle, include_header=False, line_terminator="\n")
                rows += chunk.height
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return rows


def write_detail_csv(assessment: FileAssessment, path: Path) -> int:
    """Write every finding row. Returns the number of rows written."""
    return _write(path, DETAIL_SCHEMA_LINE, DETAIL_COLUMNS, detail_chunks(assessment))


def dropped_frame(dropped: pl.DataFrame) -> pl.DataFrame:
    """The dropped-loans list as it is written: escaped."""
    return escape_frame(dropped, DROPPED_ESCAPED_COLUMNS)


def write_dropped_loans_csv(dropped: pl.DataFrame, path: Path) -> int:
    """Write the dropped-loans list (`waterfall.DROPPED_SCHEMA`)."""
    return _write(path, DROPPED_SCHEMA_LINE, tuple(dropped.columns), iter([dropped_frame(dropped)]))


__all__ = [
    "DETAIL_COLUMNS", "DETAIL_SCHEMA_LINE", "DROPPED_SCHEMA_LINE", "detail_chunks", "dropped_frame",
    "escape", "unescape", "write_detail_csv", "write_dropped_loans_csv",
]
