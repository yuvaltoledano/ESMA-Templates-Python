"""Value states, in the classification order of specification 6.1."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from esma_milan.quality.catalogue import FIELDS
from esma_milan.quality.refdata import load_reference_data
from esma_milan.quality.states import Classified, classify, value_digits

REF = load_reference_data()
VALID_LEI = "ND4X0000000000000085"


def _c(field: str, raw: str | None) -> Classified:
    return classify(FIELDS[field], raw, REF)


@pytest.mark.parametrize("raw", [None, "", " ", "\t", "\u00a0", " \r\n"])
def test_empty(raw: str | None) -> None:
    for field in ("RREL30", "RREL22", "RREL69"):
        assert _c(field, raw).state == "empty"


@pytest.mark.parametrize("field", ["RREL30", "RREL22", "RREL2", "RREL69", "RREL81", "RREL80"])
def test_exact_nd_tokens_are_nd_states_in_every_field(field: str) -> None:
    for token in ("ND1", "ND2", "ND3", "ND5"):
        assert _c(field, token).state == token.lower()
    dated = _c(field, "ND4-2027-01-01")
    assert dated.state == "nd4_dated" and dated.nd4_date == date(2027, 1, 1)


def test_malformed_nd4_in_typed_fields() -> None:
    for field in ("RREL30", "RREL23", "RREL12", "RREL10", "RREL69", "RREL28", "RREL81", "RREL11"):
        for raw in ("ND4", "ND4-2027-13-01", "ND4-27-01-01", "ND4X"):
            assert _c(field, raw).state == "nd4_malformed", (field, raw)


def test_nd4_prefix_proves_nothing_in_an_identifier() -> None:
    for raw in ("ND4000123", "ND4XYZ", "ND4-ALPHA"):
        value = _c("RREL2", raw)
        assert value.state == "present" and value.ok
    assert _c("RREL2", "ND4").state == "nd4_malformed"
    assert _c("RREL2", "ND4-2027-13-01").state == "nd4_malformed"


def test_nd4_in_an_lei_field() -> None:
    valid = _c("RREL80", VALID_LEI)
    assert valid.state == "present" and valid.ok
    assert _c("RREL80", "ND4").state == "nd4_malformed"
    assert _c("RREL80", "ND4-2027-13-01").state == "nd4_malformed"
    # Starts with ND4, is not an LEI and is not an ND4 attempt: a present, invalid LEI.
    other = _c("RREL80", "ND4X0000000000000000")
    assert other.state == "present" and not other.ok and other.reason == "checksum"


def test_a_value_valid_for_the_field_is_never_a_marker() -> None:
    namibia = _c("RREL81", "NA")
    assert namibia.state == "present" and namibia.ok and namibia.parsed == "NA"
    assert _c("RREL81", "ND").state == "invalid_marker"  # not a country
    assert _c("RREL30", "NA").state == "invalid_marker"


@pytest.mark.parametrize(
    "raw",
    ["NA", "na", "N/A", "n/a", "#N/A", "NULL", "null", "None", "NONE", "-", "ND", "nd",
     "ND6", "nd5", "ND 5", "nd 1", " NA ", "\tNULL"],
)
def test_invalid_markers_in_typed_fields(raw: str) -> None:
    for field in ("RREL30", "RREL23", "RREL69", "RREL10", "RREL28", "RREL11", "RREL80"):
        assert _c(field, raw).state == "invalid_marker", (field, raw)


@pytest.mark.parametrize("raw", [" ND5", "ND5 ", "\tND1", " ND4-2027-01-01", "ND4-2027-01-01 "])
def test_padded_nd_tokens(raw: str) -> None:
    assert _c("RREL30", raw).state == "invalid_marker"
    assert _c("RREL23", raw).state == "invalid_marker"
    text = _c("RREL22", raw)
    assert text.state == "present" and text.ok  # content in free text


def test_markers_are_content_in_free_text() -> None:
    for raw in ("NULL", "-", "NONE", "NA", "ND", "N/A", "nd5", "ND6"):
        value = _c("RREL22", raw)
        assert value.state == "present" and value.ok, raw
    assert _c("RREL22", "ND5").state == "nd5"
    assert _c("RREL22", "ND4-2027-01-01").state == "nd4_dated"


def test_nan_is_a_format_failure_not_a_marker() -> None:
    value = _c("RREL30", "NaN")
    assert value.state == "present" and not value.ok and value.reason == "lexical"


def test_a_valid_nd_is_never_also_a_failed_value() -> None:
    for field in ("RREL30", "RREL23"):
        value = _c(field, "ND5")
        assert value.state == "nd5" and value.reason is None and value.parsed is None


@pytest.mark.parametrize(
    ("text", "digits"),
    [("1.234560000", (6, 5)), ("1.234567", (7, 6)), ("000123.4500", (5, 2)), ("1000", (4, 0)),
     ("0.5", (1, 1)), ("0", (0, 0)), ("1E+3", (4, 0)), ("-12.340", (4, 2))],
)
def test_digits_are_counted_on_the_value(text: str, digits: tuple[int, int]) -> None:
    assert value_digits(Decimal(text)) == digits
