"""ESMA loans/collaterals CSV reader.

Mirrors `read_and_clean()` in r_reference/R/utils.R:57-126:

  1. Read the CSV with readr's NA-token list (config.CSV_NA_TOKENS).
  2. Force the listed columns to character type (matches readr's
     `col_types = cols(<each col> = col_character(), .default = col_guess())`).
  3. Rename source columns via the taxonomy {field_code: field_name} dict
     (only columns whose source name is a taxonomy key are renamed; the
     rest pass through unchanged - mirrors R's
     `cols_to_rename = intersect(names(data_raw), names(name_map))`).
  4. Apply snake_case clean_names to all columns.
  5. Hard-error if the rename produces duplicate column names (matches
     the R code's `stop(...)` at utils.R:104-112).
  6. Hard-error on an empty file (matches utils.R:115).
  7. Log a structured "successfully read" event (matches utils.R:120).

Two mechanisms here have no readr equivalent and are deliberate
deviations from the R reference:

Taxonomy-typed numeric fields
  Polars infers each column's type over the whole file
  (`infer_schema_length=None`; readr samples 10,000 rows). Then every
  present column named in `numeric_cols` - the taxonomy fields whose
  FORMAT is {MONETARY}, {PERCENTAGE} or {INTEGER-n}, see
  config.NUMERIC_TAXONOMY_FORMATS and read_taxonomy.numeric_field_names
  - is cast to Float64 (R's double). An Int64 or Float64 column is
  widened; any other inferred type is cast through String with readr's
  whitespace trimming. A non-null source value that does not parse as a
  number raises ValueError naming the file, field, rows and raw values.
  Deviations: (1) a numeric field that is ND for its first 10,000 rows
  is read correctly, where readr guesses `logical` and nulls every later
  value; (2) a non-numeric value in a numeric field halts the run, where
  readr nulls it with a warning.

Non-UTF-8 input
  Polars requires valid UTF-8; readr reads any bytes and R repairs text
  at write time via `sanitize_utf8()` (utils.R:743-790, called from
  `safe_for_excel()`). Here the file's bytes are checked once: valid
  UTF-8 goes straight to Polars, unchanged from before. Otherwise the
  CSV is parsed with the standard-library `csv` module on a lossless
  Latin-1 decode, every field (header included) is restored to its
  original bytes and run through `sanitize_utf8.sanitize_utf8_value`
  - the same per-value cascade R applies - and the repaired rows are
  re-serialised to an in-memory UTF-8 CSV that Polars reads with the
  same arguments. Repairing at read time rather than write time gives
  the same output because every intermediate string operation in the
  pipeline compares ASCII codes only.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Collection, Iterable
from pathlib import Path

import polars as pl
import structlog

from esma_milan.config import CSV_NA_TOKENS
from esma_milan.io_layer.clean_names import clean_name
from esma_milan.io_layer.sanitize_utf8 import sanitize_utf8_value

log = structlog.get_logger(__name__)

# Number of (row, raw value) pairs listed in the non-numeric-value error.
_MAX_BAD_VALUES_REPORTED: int = 10


def read_and_clean(
    file_path: Path,
    taxonomy: dict[str, str],
    character_cols: Iterable[str] | None = None,
    numeric_cols: Collection[str] | None = None,
) -> pl.DataFrame:
    """Read an ESMA CSV, apply the taxonomy rename, and snake-case columns.

    Args:
        file_path: Path to the CSV.
        taxonomy: {field_code: field_name} mapping from
                  esma_milan.io_layer.read_taxonomy.load_taxonomy().
        character_cols: Optional iterable of source-side column names that
                        must be read as String rather than Polars' inferred
                        type. Mirrors R's `character_cols` argument; the
                        pipeline passes ("RREL2","RREL3","RREL4","RREL5")
                        for loans and ("RREC2","RREC3","RREC4") for
                        collaterals to keep ID columns from being silently
                        coerced to integers.
        numeric_cols: Optional collection of post-rename snake_case column
                      names to read as Float64 (the taxonomy's numeric
                      fields; see read_taxonomy.numeric_field_names).
                      Names not present in the file are ignored.

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: on duplicate column names after rename (data-quality
                    error worth halting on - taxonomy maps two source
                    fields to the same target name), on an empty file,
                    or on a non-numeric value in a `numeric_cols` column.

    Returns:
        A Polars DataFrame with snake_case columns and ESMA NA tokens
        converted to nulls.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    schema_overrides: dict[str, pl.DataType] | None = None
    if character_cols:
        # Polars' schema_overrides is keyed on the SOURCE column names
        # (i.e. what's actually in the CSV header), so we apply this
        # before rename. If a listed column doesn't appear in the file,
        # Polars silently ignores the override.
        schema_overrides = {col: pl.String() for col in character_cols}

    df = pl.read_csv(
        _utf8_source(file_path),
        null_values=list(CSV_NA_TOKENS),
        schema_overrides=schema_overrides,
        infer_schema_length=None,  # whole file; readr samples guess_max=10000
    )

    # Rename: source column -> taxonomy field name (if mapped),
    # otherwise pass through unchanged. Then clean_names every column.
    new_names = [clean_name(taxonomy.get(c, c)) for c in df.columns]

    # Critical data-quality check: rename produced duplicates.
    seen: set[str] = set()
    duplicates: list[str] = []
    for n in new_names:
        if n in seen and n not in duplicates:
            duplicates.append(n)
        seen.add(n)
    if duplicates:
        raise ValueError(
            f"CRITICAL ERROR: Duplicated column names found after renaming: "
            f"{', '.join(duplicates)}. This indicates a problem with the "
            f"taxonomy mapping. Multiple ESMA field codes may be mapping to "
            f"the same field name."
        )

    df.columns = new_names

    if df.height == 0:
        raise ValueError(f"ERROR: No data rows found in file: {file_path.name}")

    if numeric_cols:
        df = _cast_numeric_columns(df, numeric_cols, file_path, taxonomy)

    log.info(
        "read_and_clean",
        file=file_path.name,
        n_rows=df.height,
        n_cols=df.width,
    )

    return df


# ---------------------------------------------------------------------------
# Taxonomy-typed numeric fields
# ---------------------------------------------------------------------------


def _cast_numeric_columns(
    df: pl.DataFrame,
    numeric_cols: Collection[str],
    file_path: Path,
    taxonomy: dict[str, str],
) -> pl.DataFrame:
    """Cast every present `numeric_cols` column to Float64, erroring on
    any non-null value the cast cannot parse."""
    present = [c for c in df.columns if c in numeric_cols]
    if not present:
        return df

    casts: list[pl.Expr] = []
    for col in present:
        if df.schema[col] in (pl.Int64, pl.Float64):
            # Widening (Int64) or no-op (Float64): cannot produce nulls.
            casts.append(pl.col(col).cast(pl.Float64))
        else:
            # String, Boolean or anything else Polars inferred: parse the
            # text. Trimming mirrors readr's default `trim_ws = TRUE`, so
            # " 100" reads as 100 here as it does in R. Boolean goes
            # through its text form ("true"/"false"), which does not
            # parse, so a TRUE/FALSE-only monetary column is an error
            # rather than a silent 1.0/0.0.
            casts.append(
                pl.col(col)
                .cast(pl.String)
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
            )
    out = df.with_columns(casts)

    for col in present:
        if df.schema[col] in (pl.Int64, pl.Float64):
            continue
        bad_mask = df[col].is_not_null() & out[col].is_null()
        n_bad = int(bad_mask.sum())
        if n_bad == 0:
            continue
        bad_rows = [
            int(row)
            for row in df.with_row_index("_row")
            .filter(bad_mask)
            .head(_MAX_BAD_VALUES_REPORTED)["_row"]
            .to_list()
        ]
        raw_values = _raw_column_text(file_path, df.columns.index(col), bad_rows)
        samples = [
            f"  data row {row + 1}: {value!r}"
            for row, value in zip(bad_rows, raw_values, strict=True)
        ]
        more = n_bad - len(samples)
        code = next((k for k, v in taxonomy.items() if clean_name(v) == col), None)
        field = f"{code} ({col})" if code else col
        raise ValueError(
            f"ERROR: {n_bad} non-numeric value(s) in numeric field {field} "
            f"of file {file_path.name}; the taxonomy types this field as "
            f"numeric and every value must parse as a number or be an ND token.\n"
            + "\n".join(samples)
            + (f"\n  ... and {more} more" if more > 0 else "")
        )

    return out


def _raw_column_text(
    file_path: Path, column_position: int, data_rows: list[int]
) -> list[str | None]:
    """The file's own text in one column for the given 0-based data rows.

    The error samples come from here rather than from the parsed frame so
    they show what the file holds: Polars has already normalised a
    TRUE/FALSE column to Boolean, which would print as True/False (or
    'true'/'false' after a String cast). Only the error path reads the
    file a second time; `infer_schema_length=0` reads every column as
    String, and `column_position` is the column's position in the file,
    which the rename in `read_and_clean` leaves unchanged.
    """
    raw = pl.read_csv(
        _utf8_source(file_path),
        columns=[column_position],
        infer_schema_length=0,
    )
    values = raw.to_series(0).to_list()
    return [values[row] for row in data_rows]


# ---------------------------------------------------------------------------
# Non-UTF-8 input
# ---------------------------------------------------------------------------


def _utf8_source(file_path: Path) -> Path | io.BytesIO:
    """Return `file_path` unchanged when its bytes are valid UTF-8;
    otherwise an in-memory UTF-8 CSV with every field repaired."""
    raw = file_path.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return file_path
    repaired, n_fields = _repair_csv_encoding(raw)
    log.warning(
        "csv_encoding_repaired",
        file=file_path.name,
        n_fields_repaired=n_fields,
    )
    return io.BytesIO(repaired)


def _repair_csv_encoding(raw: bytes) -> tuple[bytes, int]:
    """Re-encode a non-UTF-8 CSV to UTF-8, repairing each field through
    `sanitize_utf8_value`'s UTF-8 -> Windows-1252 -> Latin-1 -> strip
    cascade (the per-value behaviour of R's `sanitize_utf8()`).

    Latin-1 maps every byte to one code point, so decoding with it and
    re-encoding each parsed field recovers the field's original bytes
    exactly; the CSV structure (',' '"' and line breaks) is ASCII and
    unaffected. Returns the UTF-8 bytes and the number of fields that
    needed repair (fields that were already valid UTF-8 pass through).
    """
    text = raw.decode("latin-1")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    n_repaired = 0
    for row in csv.reader(io.StringIO(text, newline="")):
        fixed: list[str] = []
        for field in row:
            field_bytes = field.encode("latin-1")
            try:
                fixed.append(field_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                n_repaired += 1
                repaired = sanitize_utf8_value(field_bytes)
                fixed.append(repaired if repaired is not None else "")
        writer.writerow(fixed)
    return out.getvalue().encode("utf-8"), n_repaired
