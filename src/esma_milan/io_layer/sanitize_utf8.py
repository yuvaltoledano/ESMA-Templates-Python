"""UTF-8 sanitisation for safe Excel writes.

Mirrors `sanitize_utf8()` in r_reference/R/utils.R:743-789.

ESMA loan-system CSVs frequently contain Windows-1252 (CP1252) or Latin-1
bytes that slip through readers and later cause openpyxl/openxlsx to error
on invalid byte sequences. This helper performs a best-effort cascade:

  Pass 1: strict UTF-8 validation (passthrough for already-valid).
  Pass 2: re-decode as Windows-1252 (the right map for European loan
          systems - 0x80..0x9F = euro sign, smart quotes, em/en dash).
  Pass 3: re-decode as Latin-1 for the handful of bytes (0x81 0x8D 0x8F
          0x90 0x9D) that are valid in Latin-1 but undefined in CP1252.
  Pass 4: lossy strip of any remaining invalid bytes.

A None/NaN input is returned as None. The implementation operates on
Python `str` (already-decoded text) and on raw `bytes` produced by
upstream readers. For Polars Series in the pipeline, sanitisation runs at
the row level on the values backing each Utf8 column.
"""

from __future__ import annotations

import polars as pl


def sanitize_utf8_value(value: object) -> str | None:
    """Sanitise a single string-or-bytes value to valid UTF-8.

    None / NaN -> None. Already-valid UTF-8 strings pass through unchanged.
    Bytes are decoded via the cascade described in the module docstring.
    Non-string, non-bytes scalars (int, float, etc.) are stringified
    via str().
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN check
        return None

    # Bytes path: decode via the cascade.
    if isinstance(value, (bytes, bytearray)):
        return _decode_bytes(bytes(value))

    if isinstance(value, str):
        # Python `str` is already a sequence of code points - it cannot
        # carry "invalid UTF-8 bytes" the way an R character with raw
        # bytes can. The Python-side equivalent of R's pass-1 check is
        # to round-trip via UTF-8 encode/decode and let any unencodable
        # surrogate or unpaired code point raise; in practice all str
        # values reach this function through some decoder already, so
        # we just return as-is. The Polars wrapper handles binary/raw
        # columns separately if they exist.
        return value

    # Anything else: stringify and trust Python's repr.
    return str(value)


def _decode_bytes(raw: bytes) -> str:
    """Apply the four-pass decoder cascade to a bytes object."""
    # Pass 1: strict UTF-8.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Pass 2: Windows-1252. This decoder accepts 0x80..0x9F as printable
    # glyphs (euro, smart quotes, em/en dash), unlike pass 3.
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        pass

    # Pass 3: Latin-1 (ISO-8859-1). Latin-1 covers 0x81 0x8D 0x8F 0x90
    # 0x9D - bytes that are undefined in CP1252 but valid (as C1 control
    # codes) in Latin-1. Latin-1 cannot fail on any byte, but we keep
    # the try/except for symmetry.
    try:
        return raw.decode("latin-1")
    except UnicodeDecodeError:
        pass

    # Pass 4: strict UTF-8 with errors='replace'/'ignore' to strip
    # everything we still can't decode. R's final `iconv(..., to='ASCII',
    # sub='')` is equivalent to ascii errors='ignore'.
    return raw.decode("utf-8", errors="ignore")


def sanitize_utf8(series: pl.Series) -> pl.Series:
    """Apply sanitize_utf8_value across every element of a Polars Series.

    Non-string/non-bytes columns are returned unchanged - sanitisation
    only matters for text columns destined for the Excel writer.
    """
    if series.dtype not in (pl.String, pl.Binary):
        return series
    sanitised = [sanitize_utf8_value(v) for v in series.to_list()]
    return pl.Series(name=series.name, values=sanitised, dtype=pl.String)
