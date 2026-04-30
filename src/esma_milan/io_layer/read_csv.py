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
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl
import structlog

from esma_milan.config import CSV_NA_TOKENS
from esma_milan.io_layer.clean_names import clean_name

log = structlog.get_logger(__name__)


def read_and_clean(
    file_path: Path,
    taxonomy: dict[str, str],
    character_cols: Iterable[str] | None = None,
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

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: on duplicate column names after rename (data-quality
                    error worth halting on - taxonomy maps two source
                    fields to the same target name) or on an empty file.

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
        file_path,
        null_values=list(CSV_NA_TOKENS),
        schema_overrides=schema_overrides,
        infer_schema_length=10000,  # matches readr's guess_max=10000
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

    log.info(
        "read_and_clean",
        file=file_path.name,
        n_rows=df.height,
        n_cols=df.width,
    )

    return df
