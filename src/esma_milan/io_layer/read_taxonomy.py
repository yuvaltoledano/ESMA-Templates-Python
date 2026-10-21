"""ESMA taxonomy loader.

The taxonomy is a fixed ESMA artefact: ESMA defines the RRE template and
it changes only when ESMA republishes it. The workbook ships inside this
package as ``esma_milan/data/esma_rre_taxonomy.xlsx`` and
`bundled_taxonomy_path()` locates it through `importlib.resources`, so the
pipeline needs neither a working-directory convention nor an upload. The
R reference carries the same file at
``r_reference/inputs/ESMA template taxonomy.xlsx``;
tests/unit/test_bundled_taxonomy.py keeps the two byte-identical.

Mirrors the taxonomy read in r_reference/R/pipeline.R:134 and the use
inside r_reference/R/utils.R:86-89:

    name_map <- taxonomy |>
      dplyr::filter(!is.na(`FIELD CODE`), !is.na(`FIELD NAME`)) |>
      dplyr::select(`FIELD CODE`, `FIELD NAME`) |>
      tibble::deframe()

Reads sheet 1 only. Filters rows where either FIELD CODE or FIELD NAME is
empty/None. Returns a {field_code -> field_name} dict that
read_csv.read_and_clean() consumes when renaming source columns.

`load_taxonomy_formats()` is a sibling that reads the same sheet and
returns {field_code -> FORMAT} (e.g. "RREL30" -> "{MONETARY}"). The
FORMAT column is the taxonomy's per-field schema; `numeric_field_names()`
derives from it the set of snake_case column names that
read_csv.read_and_clean() reads as Float64 (see
config.NUMERIC_TAXONOMY_FORMATS). R has no equivalent: readr guesses
types from a 10,000-row sample, which is the mechanism this replaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path

import openpyxl

from esma_milan.config import INTEGER_TAXONOMY_FORMAT_PREFIX, NUMERIC_TAXONOMY_FORMATS
from esma_milan.io_layer.clean_names import clean_name

FIELD_CODE_COLUMN: str = "FIELD CODE"
FIELD_NAME_COLUMN: str = "FIELD NAME"
FORMAT_COLUMN: str = "FORMAT"

BUNDLED_TAXONOMY_PACKAGE: str = "esma_milan.data"
BUNDLED_TAXONOMY_FILENAME: str = "esma_rre_taxonomy.xlsx"


def bundled_taxonomy_path() -> Path:
    """Filesystem path of the ESMA RRE taxonomy shipped with the package.

    Resolved through `importlib.resources`, so it is independent of the
    current working directory and of how the package was installed
    (editable checkout or wheel). `as_file` yields the on-disk file for a
    directory install and is a no-op on exit, so the path stays valid
    after the `with` block. The package is never installed zipped; were
    it, `as_file` would yield a temporary extraction that is removed on
    exit, and the check below turns that into a clear error instead of a
    dangling path.

    Raises:
        FileNotFoundError: if the bundled workbook is not a regular file
            (a broken or zipped install).
    """
    resource = files(BUNDLED_TAXONOMY_PACKAGE).joinpath(BUNDLED_TAXONOMY_FILENAME)
    with as_file(resource) as path:
        resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Bundled ESMA taxonomy not found at {resolved}; the package must be "
            "installed as a directory (editable checkout or wheel), not zipped."
        )
    return resolved


def load_taxonomy(path: Path) -> dict[str, str]:
    """Read an ESMA taxonomy XLSX and return a {field_code: field_name} dict.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if either FIELD CODE or FIELD NAME is missing from sheet 1.
    """
    return _load_keyed_column(path, FIELD_NAME_COLUMN)


def load_taxonomy_formats(path: Path) -> dict[str, str]:
    """Read an ESMA taxonomy XLSX and return a {field_code: FORMAT} dict.

    Same sheet, filtering and last-one-wins semantics as `load_taxonomy`;
    rows whose FIELD CODE or FORMAT cell is empty are skipped.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if either FIELD CODE or FORMAT is missing from sheet 1.
                    A taxonomy without FORMAT cannot type the numeric
                    fields, and silently falling back to type inference
                    would reintroduce the sample-inference failure this
                    column exists to prevent.
    """
    return _load_keyed_column(path, FORMAT_COLUMN)


def is_numeric_taxonomy_format(fmt: str) -> bool:
    """True for FORMAT values read as Float64: {MONETARY}, {PERCENTAGE}, {INTEGER-n}."""
    return fmt in NUMERIC_TAXONOMY_FORMATS or fmt.startswith(INTEGER_TAXONOMY_FORMAT_PREFIX)


def numeric_field_names(
    taxonomy: Mapping[str, str],
    formats: Mapping[str, str],
) -> frozenset[str]:
    """Snake_case names of every taxonomy field whose FORMAT is numeric.

    Keyed on the post-rename name (`clean_name(FIELD NAME)`), the same
    transform `read_csv.read_and_clean` applies, so the set matches both
    ESMA-coded headers (renamed via `taxonomy`) and already-snake_case
    headers (which the rename leaves untouched). Codes present in
    `formats` but absent from `taxonomy` have no target name and are
    skipped.
    """
    return frozenset(
        clean_name(taxonomy[code])
        for code, fmt in formats.items()
        if code in taxonomy and is_numeric_taxonomy_format(fmt)
    )


def _load_keyed_column(path: Path, value_column: str) -> dict[str, str]:
    """Return {FIELD CODE: <value_column>} from sheet 1 of the taxonomy."""
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file does not exist: {path}")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        ws.reset_dimensions()  # force re-scan (some writers cache 1x1)

        rows_iter = ws.iter_rows(values_only=True)
        header = next(rows_iter, None)
        if header is None:
            raise ValueError(f"Taxonomy sheet 1 in {path} is empty")

        header_strs = [_norm(h) for h in header]
        code_idx = _column_index(header_strs, FIELD_CODE_COLUMN)
        value_idx = _column_index(header_strs, value_column)

        mapping: dict[str, str] = {}
        for row in rows_iter:
            code = _cell(row, code_idx)
            value = _cell(row, value_idx)
            if code is None or value is None:
                continue
            # Mirror R's deframe(): later entries overwrite earlier ones with
            # the same key. The synthetic taxonomy doesn't exercise this; the
            # behaviour is documented to match R exactly should it ever arise.
            mapping[code] = value
        return mapping
    finally:
        wb.close()


def _column_index(header_strs: list[str], column: str) -> int:
    try:
        return header_strs.index(column)
    except ValueError:
        raise ValueError(
            f"Taxonomy must contain {column!r} column. "
            f"Found columns: {[h for h in header_strs if h]}"
        ) from None


def _norm(value: object) -> str:
    """Normalise a header cell to a stripped string. None -> ''."""
    if value is None:
        return ""
    return str(value).strip()


def _cell(row: tuple[object, ...], idx: int) -> str | None:
    """Return a stripped string for a data cell, or None if missing/empty."""
    if idx >= len(row):
        return None
    v = row[idx]
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None
