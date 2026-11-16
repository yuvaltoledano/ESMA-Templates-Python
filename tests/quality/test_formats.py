"""P4, P5, P7 and the RREL65 validator (specification 6.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from esma_milan.quality.catalogue import FIELDS
from esma_milan.quality.refdata import load_reference_data
from esma_milan.quality.states import classify
from tests.quality.builders import LEI, Pair, assess, make_lei

REF = load_reference_data()


def _verdict(field: str, raw: str) -> tuple[bool, str | None, str | None]:
    value = classify(FIELDS[field], raw, REF)
    assert value.state == "present", (field, raw, value.state)
    return value.ok, value.alternative, value.reason


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        # {MONETARY}: lexical form, then 18 digits of which 5 fractional
        ("RREL30", "12", (True, None, None)),
        ("RREL30", ".5", (True, None, None)),
        ("RREL30", "1.", (True, None, None)),
        ("RREL30", "+01.50", (True, None, None)),
        ("RREL30", "-1000.5", (True, None, None)),
        ("RREL30", " 12 ", (True, "xml_whitespace", None)),
        ("RREL30", "\t12.5\r\n", (True, "xml_whitespace", None)),
        ("RREL30", "1.234560000", (True, None, None)),
        ("RREL30", "1.234567", (False, None, "digits")),
        ("RREL30", "1234567890123.12345", (True, None, None)),  # 18 digits, 5 fractional
        ("RREL30", "12345678901234.12345", (False, None, "digits")),  # 19 digits
        ("RREL30", "1_000", (False, None, "lexical")),
        ("RREL30", "1e3", (False, None, "lexical")),
        ("RREL30", "１２", (False, None, "lexical")),  # noqa: RUF001 - full-width digits on purpose
        ("RREL30", "1,000.00", (False, None, "lexical")),
        ("RREL30", "€100", (False, None, "lexical")),
        ("RREL30", "\u00a012", (False, None, "lexical")),  # a non-breaking space is not collapsed
        ("RREL30", "1\x1f2", (False, None, "control_character")),
        # {PERCENTAGE}: 11 digits of which 10 fractional
        ("RREL43", "7.0", (True, None, None)),
        ("RREL43", "-0.25", (True, None, None)),
        ("RREL43", "1.0123456789", (True, None, None)),
        ("RREL43", "1.01234567891", (False, None, "digits")),
        ("RREL43", "12.0123456789", (False, None, "digits")),  # 12 digits in total
        # {INTEGER-9999}
        ("RREL25", "0", (True, None, None)),
        ("RREL25", "9999", (True, None, None)),
        ("RREL25", "12.0", (True, None, None)),
        ("RREL25", "+12", (True, None, None)),
        ("RREL25", "-0", (True, None, None)),
        ("RREL25", "12.5", (False, None, "integer_range")),
        ("RREL25", "10000", (False, None, "integer_range")),
        ("RREL25", "-1", (False, None, "integer_range")),
        # {DATEFORMAT}
        ("RREL23", "2019-05-10", (True, None, None)),
        ("RREL23", " 2019-05-10 ", (True, "xml_whitespace", None)),
        ("RREL23", "2019/05/10", (False, None, "alternative_date_form")),
        ("RREL23", "2019-05-10 00:00:00", (False, None, "alternative_date_form")),
        ("RREL23", "43595", (False, None, "excel_serial")),
        ("RREL23", "2019-02-30", (False, None, "not_a_date")),
        ("RREL23", "10/05/2019", (False, None, "not_a_date")),
        ("RREL23", "2019-05-10Z", (False, None, "not_a_date")),
        # {YEAR}
        ("RREL12", "2021", (True, None, None)),
        ("RREL12", " 2021", (True, "xml_whitespace", None)),
        ("RREL12", "21", (False, None, "not_a_year")),
        # {Y/N}
        ("RREL10", "Y", (True, None, None)),
        ("RREL10", "N", (True, None, None)),
        ("RREL10", "true", (True, "xml_boolean", None)),
        ("RREL10", "0", (True, "xml_boolean", None)),
        ("RREL10", " true ", (True, "xml_boolean", None)),
        ("RREL10", " Y", (True, "xml_whitespace", None)),
        ("RREL10", "TRUE", (False, None, "boolean_spelling")),
        ("RREL10", "Yes", (False, None, "boolean_spelling")),
        ("RREL10", "y", (False, None, "boolean_spelling")),
        ("RREL10", "maybe", (False, None, "boolean_spelling")),
        # {LIST}: no whitespace collapse
        ("RREL69", "PERF", (True, None, None)),
        ("RREL69", " PERF", (False, None, "trim_or_case")),
        ("RREL69", "perf", (False, None, "trim_or_case")),
        ("RREL69", "PERFORMING", (False, None, "not_a_code")),
        # {NUTS}: shape only here; membership is P11
        ("RREL11", "NL329", (True, None, None)),
        ("RREL11", "nl329", (False, None, "trim_or_case")),
        ("RREL11", "NL-NH", (False, None, "shape")),
        # {COUNTRYCODE_2}, {CURRENCYCODE_3}
        ("RREL81", "GB", (True, None, None)),
        ("RREL81", " GB ", (False, None, "trim_or_case")),
        ("RREL81", "XX", (False, None, "not_in_list")),
        ("RREL81", "GBR", (False, None, "shape")),
        ("RREL28", "EUR", (True, None, None)),
        ("RREL28", "DEM", (True, None, None)),  # a historic code is valid
        ("RREL28", "EURO", (False, None, "shape")),
        ("RREL28", "eur", (False, None, "trim_or_case")),
        ("RREL28", "XQZ", (False, None, "not_in_list")),
        # {LEI}
        ("RREL80", LEI, (True, None, None)),
        ("RREL80", LEI[:-2] + ("00" if LEI[-2:] != "00" else "01"), (False, None, "checksum")),
        ("RREL80", LEI.lower(), (False, None, "shape")),
        ("RREL80", LEI[:-1], (False, None, "shape")),
    ],
)
def test_type_rules(field: str, raw: str, expected: tuple[bool, str | None, str | None]) -> None:
    assert _verdict(field, raw) == expected


def test_boolean_interpretation_is_shown_when_unambiguous() -> None:
    assert classify(FIELDS["RREL10"], "TRUE", REF).expected == "Y or N (read as true)"
    assert classify(FIELDS["RREL10"], "maybe", REF).expected == "Y or N"


def test_padding_p4_and_p5_agree(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL30", " 12 ", record=1).set("RREL43", " 12.5 ", record=1)
    pair.set("RREL10", " true ", record=1).set("RREL29", ".5", record=1).set("RREL31", "1.", record=1)
    pair.set("RREL69", " PERF", record=2).set("RREL81", " GB ", record=2)
    assessed = assess(pair, tmp_path, "loans")
    # A padded number is valid for its type: an accepted alternative, no P5 violation.
    assert assessed.rules_on("RREL30") == {("P4(a)", "accepted alternative")}
    assert assessed.rules_on("RREL43") == {("P4(a)", "accepted alternative")}
    assert assessed.rules_on("RREL10") == {("P5", "accepted alternative")}
    assert assessed.rules_on("RREL29") == set() and assessed.rules_on("RREL31") == set()
    assert assessed.found("P4(a)", "RREL30")[0]["raw_value"] == " 12 "  # the cell as read is kept
    # A padded code is a P5 violation and nothing from P4(a); the pipeline effect is P4(d).
    assert assessed.rules_on("RREL69") == {("P5", "violation"), ("P4(d)", "violation")}
    assert assessed.rules_on("RREL81") == {("P5", "violation")}
    assert assessed.outcome("P5", "RREL69") == (2, 2, 1, 0)
    assert assessed.outcome("P4(a)", "RREL30") == (2, 2, 0, 0)


def test_p5_does_not_test_alphanum_fields(tmp_path: Path) -> None:
    assessed = assess(Pair().set("RREL82", "Example Bank N.V."), tmp_path, "loans")
    assert assessed.rules_on("RREL82") == {("P7(b)", "violation")}
    assert assessed.outcome("P5", "RREL82") == (0, 0, 0, 0)


def test_p7_length_and_characters(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL82", "A" * 100, record=1).set("RREL82", "A" * 101, record=2)
    pair.set("RREL79", "CAFÉ BANK", record=1).set("RREL79", "BANK & CO. (NL) 12/3", record=2)
    assessed = assess(pair, tmp_path, "loans")
    assert [(r["record"], r["reason"]) for r in assessed.found("P7(a)")] == [(2, "too_long")]
    assert [(r["field"], r["record"]) for r in assessed.found("P7(b)")] == [("RREL79", 1)]


def test_p7c_unique_identifier_construction(tmp_path: Path) -> None:
    bad_lei = LEI[:-2] + ("00" if LEI[-2:] != "00" else "01")
    pair = Pair().set("RREL1", f"{bad_lei}N202401", record=1).set("RREL1", f"{LEI}X202401", record=2)
    assessed = assess(pair, tmp_path, "loans")
    assert [(r["record"], r["reason"]) for r in assessed.found("P7(c)")] == [
        (1, "identifier_lei_check_digits"), (2, "identifier_pattern"),
    ]
    assert assess(Pair(), tmp_path, "collateral").found("P7(c)") == []


def test_lei_failures_name_their_own_source() -> None:
    shape = classify(FIELDS["RREL80"], "TOO-SHORT", REF)
    checksum = classify(FIELDS["RREL80"], make_lei("5299000TESTBANK001")[:-2] + "99", REF)
    assert shape.reason == "shape" and "[A-Z0-9]{18}[0-9]{2}" in (shape.expected or "")
    assert checksum.reason == "checksum" and "ISO 17442" in (checksum.expected or "")


def _rrel65(tmp_path: Path, cell: str) -> set[tuple[str, str]]:
    return assess(Pair().set("RREL65", cell, record=1), tmp_path, "loans").rules_on("RREL65")


def test_rrel65_elements(tmp_path: Path) -> None:
    context = ("P3g", "context")  # what the pipeline's split selects, whatever the elements
    assert _rrel65(tmp_path, "2020-01-01;2021-02-02") == {context}
    assert _rrel65(tmp_path, "2020-01-01, 2021-02-02") == {context}
    assert _rrel65(tmp_path, "2020-01-01;not a date") == {context, ("P5", "violation")}
    assert _rrel65(tmp_path, "2020-01-01;") == {context, ("P5", "violation")}
    # An ND option beside a date: two recognised elements, a question, no invalid fragment.
    assert _rrel65(tmp_path, "2020-01-01;ND5") == {context, ("P3f", "question")}
    # A date element after the cut-off is the violation; the date in a dated ND4 is not.
    assert _rrel65(tmp_path, "2020-01-01;2024-07-01") == {context, ("P6(b)", "violation")}
    assert _rrel65(tmp_path, "2020-01-01;ND4-2027-01-01") == {context, ("P3f", "question")}
    assert _rrel65(tmp_path, "ND4-2024-01-01;ND1") == {context, ("P3e", "violation")}


def test_rrel65_single_values_take_the_generic_path(tmp_path: Path) -> None:
    assert _rrel65(tmp_path, "2020-01-01") == set()
    assert _rrel65(tmp_path, "ND4-2027-01-01") == {("P3g", "context")}  # a valid ND4 state
    assessed = assess(Pair().set("RREL65", "2020-01-01;bad", record=1), tmp_path, "loans")
    row = assessed.found("P3g", "RREL65")[0]
    assert "keeps 2020-01-01" in str(row["expected"])  # the fragment disappears from the pipeline's value


def test_rrel65_pipeline_selection_is_what_stage1_keeps(tmp_path: Path) -> None:
    from esma_milan.io_layer.date_parsing import collapse_multi_date_restructuring

    for cell in ("2020-01-01;2021-02-02", "2020-01-01; bad", "bad;worse", "2020-01-01,20210315", "ND5;2019-12-31"):
        assessed = assess(Pair().set("RREL65", cell, record=1), tmp_path, "loans")
        kept = collapse_multi_date_restructuring([cell])[0]
        expected = f"keeps {kept}" if kept else "keeps no date"
        assert expected in str(assessed.found("P3g", "RREL65")[0]["expected"]), cell
    # 20210315 is not YYYY-MM-DD: P5 reports the fragment although the pipeline reads it.
    basic = assess(Pair().set("RREL65", "2020-01-01,20210315", record=1), tmp_path, "loans")
    assert ("P5", "violation") in basic.rules_on("RREL65")
