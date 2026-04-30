"""Tests for esma_milan.io_layer.sanitize_utf8.

Mirrors the R fixture-byte tests in
r_reference/tests/testthat/test-pipeline-calculations.R::"safe_for_excel
sanitizes invalid UTF-8 byte sequences" and "...decodes Windows-1252
punctuation".

The Python entry points are different (Python `str` is decoded text, not
raw bytes; bytes carrying invalid sequences are simulated by passing
explicit `bytes` values to sanitize_utf8_value), but the byte sequences
under test are exactly the ones the R tests exercise:

  - "José" encoded as Latin-1: 0x4A 0x6F 0x73 0xE9
  - Stray 0xFF: 0x61 0x62 0xFF 0x63
  - Windows-1252 punctuation: 0x80 (€), 0x93 ("), 0x94 ("), 0x97 (—)
  - Latin-1-only bytes: 0x81 0x8D 0x8F 0x90 0x9D
"""

from __future__ import annotations

import polars as pl

from esma_milan.io_layer.sanitize_utf8 import (
    sanitize_utf8,
    sanitize_utf8_value,
)

# ---------------------------------------------------------------------------
# Scalar entry point
# ---------------------------------------------------------------------------


def test_passthrough_for_ascii_string() -> None:
    assert sanitize_utf8_value("Alice") == "Alice"


def test_passthrough_for_valid_utf8_string() -> None:
    # Already-decoded Python str: no work to do.
    assert sanitize_utf8_value("José") == "José"


def test_none_returns_none() -> None:
    assert sanitize_utf8_value(None) is None


def test_nan_returns_none() -> None:
    import math
    assert sanitize_utf8_value(math.nan) is None


def test_decodes_latin1_jose_bytes() -> None:
    """Mirrors the R test's `bad_latin1 <- as.raw(c(0x4A, 0x6F, 0x73, 0xE9))`.

    "José" in Latin-1 is invalid UTF-8 because 0xE9 is a continuation
    byte without the expected lead byte. The cascade should fall through
    to CP1252 (which decodes 0xE9 as 'é') and produce "José".
    """
    raw = bytes([0x4A, 0x6F, 0x73, 0xE9])
    result = sanitize_utf8_value(raw)
    assert result == "José"


def test_strips_stray_0xff_byte() -> None:
    """Mirrors `bad_stray <- as.raw(c(0x61, 0x62, 0xFF, 0x63))`.

    0xFF is undefined in CP1252 but valid in Latin-1 (as C1 0xFF =
    Latin small letter y with diaeresis - "ÿ"). The cascade picks
    Latin-1 here, preserving the surrounding ASCII.
    """
    raw = bytes([0x61, 0x62, 0xFF, 0x63])
    result = sanitize_utf8_value(raw)
    # We don't pin the exact glyph for 0xFF (different decoders give
    # different results); we pin the surrounding ASCII and that the
    # output is non-None and contains a recoverable representation.
    assert result is not None
    assert "ab" in result
    assert "c" in result.split("ab", 1)[1]


def test_decodes_windows_1252_euro_sign() -> None:
    """Mirrors `euro <- as.raw(c(0x80))` -> Unicode EURO SIGN (U+20AC).

    Latin-1 would treat 0x80 as a C1 control character; CP1252 maps it
    to the euro glyph. The cascade must prefer CP1252 to surface the
    intended glyph.
    """
    assert sanitize_utf8_value(bytes([0x80])) == "€"  # €


def test_decodes_windows_1252_smart_quotes() -> None:
    """Mirrors `smart_open <- 0x93`, `smart_close <- 0x94`."""
    assert sanitize_utf8_value(bytes([0x93])) == "“"  # "
    assert sanitize_utf8_value(bytes([0x94])) == "”"  # "


def test_decodes_windows_1252_em_dash() -> None:
    """Mirrors `em_dash <- 0x97`."""
    assert sanitize_utf8_value(bytes([0x97])) == "—"  # —


def test_no_c1_control_characters_leak_through() -> None:
    """Mirrors the R assertion `expect_false(any(grepl("[-]", ...)))`.

    A naive Latin-1 fallback for 0x80-0x9F would silently turn euro/smart
    quotes/em-dash into U+0080-U+009F C1 control characters. The cascade
    must prefer CP1252 first to avoid this corruption.
    """
    for byte in (0x80, 0x93, 0x94, 0x97):
        result = sanitize_utf8_value(bytes([byte]))
        assert result is not None
        # No code point in the C1 control range should appear.
        assert not any(0x0080 <= ord(c) <= 0x009F for c in result), (
            f"C1 control leaked through for byte 0x{byte:02x}: {result!r}"
        )


def test_decodes_latin1_only_bytes_undefined_in_cp1252() -> None:
    """Bytes 0x81 0x8D 0x8F 0x90 0x9D are undefined in CP1252 but defined
    in Latin-1 (as C1 controls). The cascade should fall through to
    Latin-1 for those (matching the R helper's pass-3) rather than
    stripping them in pass-4.
    """
    for byte in (0x81, 0x8D, 0x8F, 0x90, 0x9D):
        result = sanitize_utf8_value(bytes([byte]))
        # Result is non-None and exactly one code point long (preserved).
        assert result is not None, f"byte 0x{byte:02x} stripped"
        assert len(result) == 1, f"byte 0x{byte:02x} produced {result!r}"


def test_already_valid_utf8_bytes_pass_through() -> None:
    """If the input bytes are valid UTF-8 (e.g. already-encoded
    multi-byte chars), pass-1 succeeds and returns the decoded text."""
    raw = "Café — €100".encode()
    assert sanitize_utf8_value(raw) == "Café — €100"


# ---------------------------------------------------------------------------
# Polars Series wrapper
# ---------------------------------------------------------------------------


def test_polars_series_passthrough_for_non_string_dtype() -> None:
    s = pl.Series("amount", [1, 2, 3])
    result = sanitize_utf8(s)
    # Non-string dtype: returned unchanged.
    assert result.to_list() == [1, 2, 3]


def test_polars_series_string_column_passes_valid_utf8_through() -> None:
    s = pl.Series("name", ["Alice", "Bob", "Carol"])
    assert sanitize_utf8(s).to_list() == ["Alice", "Bob", "Carol"]


def test_polars_series_string_column_handles_none() -> None:
    s = pl.Series("name", ["Alice", None, "Carol"])
    result = sanitize_utf8(s)
    assert result.to_list() == ["Alice", None, "Carol"]
