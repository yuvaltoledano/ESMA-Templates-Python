"""Value states and audit parsing of one cell.

Mirrors nothing in the R reference (specification 6.1 and 6.2).

Every cell is classified once, before any rule, into one of nine states:

  empty           no character other than whitespace
  invalid_marker  text that stands for "no value" and is not an ND option
  nd1 nd2 nd3     exactly those tokens
  nd4_dated       ND4-YYYY-MM-DD with a real date (the date is kept)
  nd4_malformed   an attempt at ND4 that is not that form
  nd5             exactly ND5
  present         anything else, valid for the field or not

The order matters and is the specification's: exact ND tokens first, in
every field; then the dated ND4; then the field's own domain, so a value
that is valid for the field is never a missing marker (`NA` is Namibia in
a country field); then, in typed fields only, the look-alike markers. In
free text everything that is not an exact ND token is content.

Numbers are validated from the raw text with `decimal.Decimal`, never a
float. `classify` is called once per distinct value of a column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from esma_milan.config import (
    DQ_INVALID_MISSING_MARKER_PATTERN,
    DQ_INVALID_MISSING_MARKERS,
    EXCEL_EPOCH,
)
from esma_milan.io_layer.date_parsing import DEFAULT_SERIAL_RANGE, ISO_DATE_RE, SERIAL_RE
from esma_milan.quality.catalogue import ND4_PREFIX_KINDS, FieldEntry
from esma_milan.quality.refdata import ReferenceData

STATES: tuple[str, ...] = (
    "empty", "invalid_marker", "nd1", "nd2", "nd3", "nd4_dated", "nd4_malformed", "nd5",
    "present",
)
ND_TOKENS: frozenset[str] = frozenset({"ND1", "ND2", "ND3", "ND5"})

# XML whitespace: what the XSD types of numbers, booleans and dates
# collapse. Not a general strip(): a non-breaking space is not removed.
XML_WHITESPACE = " \t\r\n"

_DATED_ND4 = re.compile(r"^ND4-([0-9]{4}-[0-9]{2}-[0-9]{2})$")
_ND4_ATTEMPT = re.compile(r"^ND4-[0-9-]+$")
_MARKER_PATTERN = re.compile(DQ_INVALID_MISSING_MARKER_PATTERN)
_MARKERS_UPPER = frozenset(m.upper() for m in DQ_INVALID_MISSING_MARKERS)
_DECIMAL = re.compile(r"^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)$")
_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_YEAR = re.compile(r"^[0-9]{4}$")
_NUTS = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}$")
_LEI = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Digit limits by kind: (total digits, fractional digits).
_DIGITS: dict[str, tuple[int, int]] = {"MONETARY": (18, 5), "PERCENTAGE": (11, 10)}
_BOOLEAN_READING = {"TRUE": "true", "YES": "true", "Y": "true", "FALSE": "false", "NO": "false", "N": "false"}


@dataclass(frozen=True)
class Classified:
    """The state of one cell and, for a present value, how it parsed."""

    state: str
    ok: bool = False
    """Present and valid for the field (format and, for country and
    currency codes, membership)."""
    parsed: Decimal | date | int | bool | str | None = None
    alternative: str | None = None
    """`xml_whitespace` or `xml_boolean`: valid, but not the template form."""
    reason: str | None = None
    """Why a present value is not valid."""
    expected: str | None = None
    nd4_date: date | None = None

    @property
    def present(self) -> bool:
        return self.state == "present"


def real_date(text: str) -> date | None:
    """`YYYY-MM-DD` and a real calendar date, else None."""
    if not _ISO_DATE.match(text):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def lei_fault(text: str) -> str | None:
    """None for a well-formed LEI; `shape` (the package pattern) or
    `checksum` (ISO 17442, ISO 7064 MOD 97-10) otherwise."""
    if not _LEI.match(text):
        return "shape"
    number = int("".join(str(int(ch, 36)) for ch in text))
    return None if number % 97 == 1 else "checksum"


def value_digits(value: Decimal) -> tuple[int, int]:
    """(total digits, fractional digits) counted on the value: leading
    zeros and trailing fractional zeros are not digits of the value."""
    _, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)
    text = "".join(map(str, digits))
    if exponent >= 0:
        integer, fraction = text + "0" * exponent, ""
    else:
        integer, fraction = text[:exponent], text[exponent:].rjust(-exponent, "0")
    integer, fraction = integer.lstrip("0"), fraction.rstrip("0")
    return len(integer) + len(fraction), len(fraction)


def pipeline_date_reading(text: str) -> str | None:
    """How Stage 1's date parser reads a value that is not `YYYY-MM-DD`:
    `alternative_date_form` (slashes or a trailing time), `excel_serial`
    (a number in the parser's range), or None (it sets the value to
    missing). Uses the parser's own patterns."""
    if ISO_DATE_RE.match(text):
        try:
            date.fromisoformat(text[:10].replace("/", "-"))
        except ValueError:
            return None
        return "alternative_date_form"
    if SERIAL_RE.match(text):
        try:
            parsed = EXCEL_EPOCH + timedelta(days=float(text))
        except (ValueError, OverflowError):
            return None
        if DEFAULT_SERIAL_RANGE[0] <= parsed <= DEFAULT_SERIAL_RANGE[1]:
            return "excel_serial"
    return None


def _number(entry: FieldEntry, raw: str) -> Classified:
    text = raw.strip(XML_WHITESPACE)
    alternative = "xml_whitespace" if text != raw else None
    if _CONTROL.search(text):
        return Classified("present", reason="control_character")
    if not _DECIMAL.match(text):
        return Classified("present", reason="lexical", expected="a decimal number in ASCII digits")
    try:
        value = Decimal(text)
    except InvalidOperation:  # pragma: no cover - the pattern admits only decimals
        return Classified("present", reason="lexical")
    if entry.kind == "INTEGER":
        if value != value.to_integral_value() or not 0 <= value <= 9999:
            return Classified("present", parsed=value, reason="integer_range",
                              expected="a whole number from 0 to 9999")
    else:
        total, fraction = value_digits(value)
        max_total, max_fraction = _DIGITS[entry.kind]
        if total > max_total or fraction > max_fraction:
            return Classified(
                "present", parsed=value, reason="digits",
                expected=f"at most {max_total} digits, {max_fraction} of them fractional",
            )
    return Classified("present", ok=True, parsed=value, alternative=alternative)


def _validate(entry: FieldEntry, raw: str, ref: ReferenceData) -> Classified:
    """A candidate present value against the field's own domain (6.2)."""
    kind = entry.kind
    if kind == "ALPHANUM":
        return Classified("present", ok=True, parsed=raw)
    if kind in ("MONETARY", "PERCENTAGE", "INTEGER"):
        return _number(entry, raw)

    text = raw.strip(XML_WHITESPACE)
    alternative = "xml_whitespace" if text != raw else None
    if _CONTROL.search(text if kind in ("DATE", "YEAR", "BOOLEAN") else raw):
        return Classified("present", reason="control_character")
    if kind == "DATE":
        parsed = real_date(text)
        if parsed is not None:
            return Classified("present", ok=True, parsed=parsed, alternative=alternative)
        return Classified("present", reason=pipeline_date_reading(text) or "not_a_date",
                          expected="YYYY-MM-DD, a real calendar date")
    if kind == "YEAR":
        if _YEAR.match(text):
            return Classified("present", ok=True, parsed=int(text), alternative=alternative)
        return Classified("present", reason="not_a_year", expected="YYYY")
    if kind == "BOOLEAN":
        if text in ("Y", "N"):
            return Classified("present", ok=True, parsed=text == "Y", alternative=alternative)
        if text in ("true", "false", "1", "0"):
            return Classified("present", ok=True, parsed=text in ("true", "1"),
                              alternative="xml_boolean")
        reading = _BOOLEAN_READING.get(text.upper())
        return Classified(
            "present", reason="boolean_spelling",
            expected="Y or N" + (f" (read as {reading})" if reading else ""),
        )

    # Code types: no whitespace collapse (specification 6.2).
    near = raw.strip().upper()
    if kind == "LIST":
        assert entry.codes is not None
        if raw in entry.codes:
            return Classified("present", ok=True, parsed=raw)
        return Classified("present", reason="trim_or_case" if near in entry.codes else "not_a_code",
                          expected="one code of the field's list")
    if kind == "NUTS":
        if _NUTS.match(raw):
            return Classified("present", ok=True, parsed=raw)
        return Classified("present", reason="trim_or_case" if _NUTS.match(near) else "shape",
                          expected="two capital letters and three characters")
    if kind in ("COUNTRY", "CURRENCY"):
        length = 2 if kind == "COUNTRY" else 3
        members = ref.countries if kind == "COUNTRY" else ref.currencies
        expected = "a two-letter ISO 3166-1 code" if kind == "COUNTRY" else "a three-letter ISO 4217 code"
        if len(raw) == length and raw.isascii() and raw.isalpha() and raw.isupper():
            if raw in members:
                return Classified("present", ok=True, parsed=raw)
            return Classified("present", parsed=raw, reason="not_in_list", expected=expected)
        return Classified("present", reason="trim_or_case" if near in members else "shape",
                          expected=expected)
    if kind == "LEI":
        fault = lei_fault(raw)
        if fault is None:
            return Classified("present", ok=True, parsed=raw)
        return Classified(
            "present", reason=fault,
            expected="20 characters, [A-Z0-9]{18}[0-9]{2}" if fault == "shape"
            else "valid ISO 17442 check digits",
        )
    raise AssertionError(f"unhandled kind {kind}")  # pragma: no cover


def is_invalid_marker(raw: str) -> bool:
    """A look-alike of "no value" in a typed field (rule P3d): the closed
    list, `ND` with one digit, or an ND token or dated ND4 that is exact
    only after trimming. Called only for values that are not valid for
    the field and are not exact ND tokens."""
    text = raw.strip(XML_WHITESPACE)
    return (
        text.upper() in _MARKERS_UPPER
        or text in ND_TOKENS
        or _MARKER_PATTERN.match(text) is not None
    )


def classify(entry: FieldEntry, raw: str | None, ref: ReferenceData) -> Classified:
    """The state of one cell of `entry`'s column (specification 6.1)."""
    if raw is None or raw.strip() == "":
        return Classified("empty")
    if raw in ND_TOKENS:
        return Classified(raw.lower())
    dated = _DATED_ND4.match(raw)
    if dated is not None:
        nd4_date = real_date(dated.group(1))
        if nd4_date is not None:
            return Classified("nd4_dated", nd4_date=nd4_date)
    if entry.code == "RREL65" and ("," in raw or ";" in raw):
        # Several elements in one cell: the RREL65 validator takes each
        # element in turn (column_rules.formats), before any ND4 prefix test.
        return Classified("present", reason="several_elements")
    if raw.startswith("ND4"):
        trimmed = _DATED_ND4.match(raw.strip(XML_WHITESPACE))
        if trimmed is not None and real_date(trimmed.group(1)) is not None:
            # A dated ND4 that is exact only after trimming: like any other
            # padded ND token, a marker in a typed field and content in text.
            if entry.kind == "ALPHANUM":
                return Classified("present", ok=True, parsed=raw)
            return Classified("invalid_marker")
        if entry.kind in ND4_PREFIX_KINDS:
            return Classified("nd4_malformed")
        if entry.kind == "LEI" and lei_fault(raw) is None:
            return Classified("present", ok=True, parsed=raw)
        if raw == "ND4" or _ND4_ATTEMPT.match(raw):
            return Classified("nd4_malformed")
    verdict = _validate(entry, raw, ref)
    if verdict.ok or entry.kind == "ALPHANUM":
        return verdict
    if is_invalid_marker(raw):
        return Classified("invalid_marker")
    return verdict
