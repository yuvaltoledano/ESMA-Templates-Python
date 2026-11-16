"""The audit read: the two files as they were supplied.

Mirrors nothing in the R reference (specification section 5).

The audit does not use the pipeline's cleaned frames and never hands a
frame to the pipeline. It reads each file itself:

  1. The bytes are decoded as the loader decodes them
     (`io_layer.read_csv._utf8_source`): valid UTF-8 as it is, otherwise
     field by field through `sanitize_utf8_value`. The audit has its own
     copy of those few steps because the loader's function logs an event,
     and because the audit also keeps the original bytes of every repaired
     cell.
  2. The header is read with the standard `csv` module, so the original
     text and position of every column are kept, repeated header texts
     included.
  3. The body is read by Polars with every column as text and no null
     tokens, from the same bytes and with the same parser the loader uses,
     so the records are split exactly as the pipeline splits them. Record
     n here (1-based) is index n - 1 of `PipelineResult.loan_record_index`
     / `collateral_record_index`. A null and an empty string are both
     "empty"; quoted and unquoted empties are not distinguished.

A file that cannot be split into records (a row with more fields than the
header, an unterminated quote, no header, line ends the two parsers read
differently) is reported with the position of the failure and yields no
records and no counts (specification 3.4).

Headers are resolved to fields by the pipeline's own two pure functions
(`clean_name` and the taxonomy dictionary), within the file's own schema.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl

from esma_milan.config import DQ_OPTIONAL_ELEMENT_HEADERS
from esma_milan.io_layer.clean_names import clean_name
from esma_milan.io_layer.read_taxonomy import bundled_taxonomy_path, load_taxonomy
from esma_milan.io_layer.sanitize_utf8 import sanitize_utf8_value
from esma_milan.quality.catalogue import FIELDS, FileKind, fields_of

Resolution = Literal[
    "esma_code",
    "taxonomy_field_name",
    "cleaned_name",
    "currency_companion",
    "optional_element",
    "export_extension",
    "unmapped",
]
RESOLVED_TO_FIELD: frozenset[str] = frozenset(
    {"esma_code", "taxonomy_field_name", "cleaned_name", "optional_element"}
)
EXPORT_EXTENSIONS: frozenset[str] = frozenset({"sec_id", "pool_cutoff_date"})

_COMPANION = re.compile(r"^(.+)_currency$")


@dataclass(frozen=True)
class ColumnResolution:
    """How one physical column was recognised."""

    position: int
    """1-based position in the file."""
    header: str
    """The header text as supplied."""
    how: Resolution
    field: str | None
    """The field code it resolved to; for a currency companion, the code
    of the amount it describes; None otherwise."""
    cleaned: str
    """`clean_name(header)`."""
    pipeline_name: str
    """The column name the pipeline gives this header
    (`clean_name(taxonomy.get(header, header))`); it differs from the
    audit's resolution where the header belongs to the other file's schema."""

    @property
    def frame_column(self) -> str:
        return f"c{self.position}"


@dataclass(frozen=True)
class LongRow:
    record: int
    physical_line: int
    fields: int


@dataclass(frozen=True)
class Unreadable:
    """Why a file could not be split into records, and where."""

    reason: str
    detail: str
    record: int | None = None
    physical_line: int | None = None
    long_rows: tuple[LongRow, ...] = ()
    header_fields: int | None = None


@dataclass(frozen=True)
class RawFile:
    """One file as read by the audit."""

    file: FileKind
    name: str
    sha256: str
    size: int
    unreadable: Unreadable | None
    headers: tuple[str, ...]
    columns: tuple[ColumnResolution, ...]
    frame: pl.DataFrame | None
    """Columns `record` (UInt32, 1-based), `physical_line` (UInt32, the
    line the record starts on) and `c1` .. `cN` (text, null for an
    unquoted empty cell). None when the file is unreadable."""
    encoding_repaired: pl.DataFrame
    """One row per cell whose bytes were not valid UTF-8: `record` (0 for
    the header), `column_position`, `original_bytes_hex`."""

    @property
    def n_records(self) -> int:
        return 0 if self.frame is None else self.frame.height

    def column_of(self, field: str) -> ColumnResolution | None:
        """The first column resolved to `field` itself (not a companion)."""
        for column in self.columns:
            if column.field == field and column.how in RESOLVED_TO_FIELD:
                return column
        return None

    def companion_of(self, field: str) -> ColumnResolution | None:
        for column in self.columns:
            if column.field == field and column.how == "currency_companion":
                return column
        return None

    def extension(self, cleaned: str) -> ColumnResolution | None:
        for column in self.columns:
            if column.how == "export_extension" and column.cleaned == cleaned:
                return column
        return None


@lru_cache(maxsize=1)
def _taxonomy() -> dict[str, str]:
    return load_taxonomy(bundled_taxonomy_path())


def resolve_headers(headers: tuple[str, ...], file: FileKind) -> tuple[ColumnResolution, ...]:
    """Resolve each header within the schema of `file` (specification 5.3).

    The order is: exact ESMA code, exact taxonomy field name, cleaned name,
    currency companion of a {MONETARY} field (or of an optional element),
    optional element, export extension, unmapped. A code is matched
    exactly, as the pipeline matches it: `rrel30` and `RREL30 ` are not
    codes.
    """
    schema = {f.code: f for f in fields_of(file)}
    by_name = {f.name: f.code for f in schema.values()}
    by_cleaned = {clean_name(f.name): f.code for f in schema.values()}
    optional = DQ_OPTIONAL_ELEMENT_HEADERS if file == "collateral" else {}
    taxonomy = _taxonomy()

    out: list[ColumnResolution] = []
    for index, header in enumerate(headers):
        cleaned = clean_name(header)
        how: Resolution = "unmapped"
        field: str | None = None
        companion = _COMPANION.match(cleaned)
        if header in schema:
            how, field = "esma_code", header
        elif header in by_name:
            how, field = "taxonomy_field_name", by_name[header]
        elif cleaned in by_cleaned:
            how, field = "cleaned_name", by_cleaned[cleaned]
        elif cleaned in optional:
            how, field = "optional_element", optional[cleaned]
        elif companion is not None:
            stem = companion.group(1)
            target = stem.upper() if stem.upper() in schema else optional.get(stem)
            if target is not None and FIELDS[target].kind == "MONETARY":
                how, field = "currency_companion", target
        elif cleaned in EXPORT_EXTENSIONS:
            how = "export_extension"
        out.append(
            ColumnResolution(
                position=index + 1,
                header=header,
                how=how,
                field=field,
                cleaned=cleaned,
                pipeline_name=clean_name(taxonomy.get(header, header)),
            )
        )
    return tuple(out)


def _decode(raw: bytes) -> tuple[str, bytes, list[tuple[int, int, str]]]:
    """Decode as the loader does. Returns the text, the UTF-8 bytes Polars
    reads, and (record, column position, original bytes in hexadecimal)
    for every repaired field; record 0 is the header."""
    try:
        return raw.decode("utf-8"), raw, []
    except UnicodeDecodeError:
        pass
    # The same steps as io_layer.read_csv._repair_csv_encoding: Latin-1
    # maps every byte to one code point, so each parsed field can be turned
    # back into its original bytes.
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    repaired: list[tuple[int, int, str]] = []
    record = -1  # -1 until the header (the first non-empty row) has been seen
    for row in csv.reader(io.StringIO(raw.decode("latin-1"), newline="")):
        if record < 0 and row:
            record = 0
        elif record >= 0:
            record += 1
        fixed: list[str] = []
        for position, cell in enumerate(row, start=1):
            cell_bytes = cell.encode("latin-1")
            try:
                fixed.append(cell_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                value = sanitize_utf8_value(cell_bytes)
                fixed.append(value if value is not None else "")
                repaired.append((max(record, 0), position, cell_bytes.hex()))
        writer.writerow(fixed)
    text = out.getvalue()
    return text, text.encode("utf-8"), repaired


def _header(text: str) -> tuple[tuple[str, ...] | None, int]:
    """The first non-empty record and the physical line it ends on."""
    reader = csv.reader(io.StringIO(text.removeprefix("﻿"), newline=""))
    for row in reader:
        if row:
            return tuple(row), reader.line_num
    return None, 0


def _locate_failure(text: str, n_header: int, detail: str) -> Unreadable:
    """Find where a file Polars could not parse goes wrong: every record
    with more fields than the header, or the record in which a quote opens
    and never closes."""
    body = text.removeprefix("﻿")
    long_rows: list[LongRow] = []
    reader = csv.reader(io.StringIO(body, newline=""))
    record = -1
    line_start = 1
    for row in reader:
        if record < 0:
            if row:
                record = 0
        else:
            record += 1
            if len(row) > n_header:
                long_rows.append(LongRow(record, line_start, len(row)))
        line_start = reader.line_num + 1
    if long_rows:
        return Unreadable(
            reason="rows_with_more_fields_than_the_header",
            detail=f"{len(long_rows)} record(s) have more fields than the header's {n_header}",
            record=long_rows[0].record,
            physical_line=long_rows[0].physical_line,
            long_rows=tuple(long_rows),
            header_fields=n_header,
        )
    strict = csv.reader(io.StringIO(body, newline=""), strict=True)
    record = -1
    line_start = 1
    try:
        for row in strict:
            if record < 0:
                if row:
                    record = 0
            else:
                record += 1
            line_start = strict.line_num + 1
    except csv.Error as exc:
        return Unreadable(
            reason="quote_not_closed_or_malformed",
            detail=str(exc),
            record=max(record + 1, 0),
            physical_line=line_start,
            header_fields=n_header,
        )
    return Unreadable(reason="not_parseable", detail=detail, header_fields=n_header)


def _repaired_frame(repaired: list[tuple[int, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        repaired,
        schema={"record": pl.UInt32, "column_position": pl.UInt16, "original_bytes_hex": pl.String},
        orient="row",
    )


def read_file(path: Path, file: FileKind) -> RawFile:
    """Read one uploaded file for the audit."""
    raw = path.read_bytes()
    text, data, repaired = _decode(raw)
    headers, header_last_line = _header(text)

    def result(
        frame: pl.DataFrame | None, unreadable: Unreadable | None
    ) -> RawFile:
        names = headers or ()
        return RawFile(
            file=file,
            name=path.name,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            unreadable=unreadable,
            headers=names,
            columns=resolve_headers(names, file) if unreadable is None else (),
            frame=frame,
            encoding_repaired=_repaired_frame(repaired),
        )

    if headers is None:
        return result(None, Unreadable(reason="no_header", detail="the file has no header record"))
    try:
        frame = pl.read_csv(io.BytesIO(data), infer_schema_length=0)
    except pl.exceptions.NoDataError as exc:
        return result(None, Unreadable(reason="no_header", detail=str(exc)))
    except pl.exceptions.PolarsError as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return result(None, _locate_failure(text, len(headers), first_line))
    if frame.width != len(headers):
        return result(
            None,
            Unreadable(
                reason="records_split_differently",
                detail=(
                    f"the header has {len(headers)} cell(s) read one way and {frame.width} "
                    "read the other; the line ends are probably a lone carriage return"
                ),
                header_fields=len(headers),
            ),
        )

    frame.columns = [f"c{i}" for i in range(1, frame.width + 1)]
    # The line a record starts on: the header's last line, one line per
    # earlier record, and the line breaks inside earlier cells.
    inner_breaks = pl.sum_horizontal(
        [pl.col(c).str.count_matches("\n", literal=True).fill_null(0) for c in frame.columns]
    )
    frame = frame.with_row_index("record", offset=1).with_columns(
        (
            pl.lit(header_last_line, dtype=pl.UInt32)
            + pl.col("record")
            + inner_breaks.cum_sum().shift(1, fill_value=0).cast(pl.UInt32)
        ).alias("physical_line")
    )
    return result(frame.select("record", "physical_line", pl.all().exclude("record", "physical_line")), None)
