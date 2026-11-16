"""Currency evidence (P13): cases 1, 2, 4 and 5, and the optional elements."""

from __future__ import annotations

from pathlib import Path

from tests.quality.builders import Pair, assess


def test_case_1_the_companion_states_the_currency(tmp_path: Path) -> None:
    loans = assess(Pair(), tmp_path, "loans")
    assert loans.found("P13") == []
    assert loans.outcome("P13", "RREL30") == (2, 2, 0, 0)


def test_case_2_the_dedicated_field_states_it(tmp_path: Path) -> None:
    # RREL30 = 100 with an empty companion and RREL28 = EUR: stated.
    pair = Pair().set("RREL30", "100").set("RREL30_CURRENCY", "")
    assert assess(pair, tmp_path, "loans").rules_on("RREL30") == set()
    # The companion column physically absent: stated, and a layout note (basis A, Info).
    absent = assess(Pair().drop("RREL30_CURRENCY"), tmp_path, "loans")
    rows = absent.found("P13", "RREL30")
    assert [(r["result_kind"], r["severity"], r["reason"], r["record"]) for r in rows] == [
        ("context", "Info", "companion_column_absent", None)
    ]
    assert absent.outcome("P13", "RREL30") == (2, 2, 0, 0)
    for field, dedicated in (("RREL29", "RREL28"), ("RREL16", "RREL18")):
        pair = Pair().set(f"{field}_CURRENCY", "ND5")
        assert assess(pair, tmp_path, "loans").found("P13", field, "violation") == [], dedicated
    pair = Pair().set("RREC13_CURRENCY", "")
    assert assess(pair, tmp_path, "collateral").found("P13", "RREC13") == []


def test_case_4_the_currency_is_not_stated(tmp_path: Path) -> None:
    # The companion empty and RREL28 empty or ND: a compliance violation.
    for dedicated in ("", "ND1", "EURO"):
        pair = Pair().set("RREL30_CURRENCY", "", record=1).set("RREL28", dedicated, record=1)
        loans = assess(pair, tmp_path, "loans")
        assert [(r["record"], r["result_kind"]) for r in loans.found("P13", "RREL30")] == [
            (1, "violation")
        ], dedicated
    # Both the companion column and the dedicated field column absent.
    pair = Pair().drop("RREL30_CURRENCY").drop("RREL28")
    loans = assess(pair, tmp_path, "loans")
    assert sorted((r["result_kind"], r["record"]) for r in loans.found("P13", "RREL30")) == [
        ("context", None), ("violation", 1), ("violation", 2),
    ]


def test_the_mappings_stay_narrow(tmp_path: Path) -> None:
    # RREL28 says nothing about RREL31; RREC22 says nothing about RREC17.
    loans = assess(Pair().set("RREL31_CURRENCY", "", record=2), tmp_path, "loans")
    assert [r["record"] for r in loans.found("P13", "RREL31", "violation")] == [2]
    collateral = assess(Pair().set("RREC17_CURRENCY", "", record=1), tmp_path, "collateral")
    assert [r["record"] for r in collateral.found("P13", "RREC17", "violation")] == [1]


def test_no_currency_is_needed_where_there_is_no_amount(tmp_path: Path) -> None:
    pair = Pair().set("RREL20", "ND5").set("RREL20_CURRENCY", "")
    loans = assess(pair, tmp_path, "loans")
    assert loans.rules_on("RREL20") == set()
    assert loans.rules_on("RREL20_CURRENCY") == set()
    assert loans.outcome("P13", "RREL20") == (0, 0, 0, 0)


def test_an_amount_that_does_not_parse_is_not_evaluable(tmp_path: Path) -> None:
    loans = assess(Pair().set("RREL30", "abc", record=1), tmp_path, "loans")
    assert loans.outcome("P13", "RREL30") == (2, 1, 0, 1)


def test_the_companion_has_its_own_format_and_membership_rules(tmp_path: Path) -> None:
    pair = Pair().set("RREL30_CURRENCY", "EURO", record=1).set("RREL30_CURRENCY", "XQZ", record=2)
    pair.set("RREL28", "ND1")  # so the companion is the only evidence
    loans = assess(pair, tmp_path, "loans")
    assert [r["record"] for r in loans.found("P5", "RREL30_CURRENCY")] == [1]
    assert [r["record"] for r in loans.found("P11", "RREL30_CURRENCY")] == [2]
    assert all(r["column_position"] is not None for r in loans.found("P5", "RREL30_CURRENCY"))
    assert [r["record"] for r in loans.found("P13", "RREL30", "violation")] == [1, 2]
    # With RREL28 = EUR the currency is stated, and EURO keeps its own format finding.
    pair = Pair().set("RREL30_CURRENCY", "EURO", record=1)
    loans = assess(pair, tmp_path, "loans")
    assert loans.found("P13", "RREL30") == []
    assert loans.rules_on("RREL30_CURRENCY") == {("P5", "violation")}


def test_optional_elements(tmp_path: Path) -> None:
    header, companion = "current_gross_annual_rental_income", "current_gross_annual_rental_income_currency"
    absent = assess(Pair(), tmp_path, "collateral")
    assert [o for o in absent.assessment.outcomes if o.field == "OPT_RENT_CURRENT"] == []  # never P1
    valid = assess(Pair().set(header, "12000").set(companion, "EUR"), tmp_path, "collateral")
    assert valid.rules_on("OPT_RENT_CURRENT") == set()
    assert valid.outcome("P5", "OPT_RENT_CURRENT") == (2, 2, 0, 0)
    pair = Pair().set(header, "-5", record=1).set(header, "ND1", record=2).set(companion, "EUR")
    bad = assess(pair, tmp_path, "collateral")
    assert bad.rules_on("OPT_RENT_CURRENT") == {("P6(a)", "violation"), ("P3a", "violation")}
    assert [r["record"] for r in bad.found("P6(a)", "OPT_RENT_CURRENT")] == [1]
    assert [r["record"] for r in bad.found("P3a", "OPT_RENT_CURRENT")] == [2]
    nd5 = assess(Pair().set(header, "ND5").set(companion, ""), tmp_path, "collateral")
    assert nd5.rules_on("OPT_RENT_CURRENT") == set()
    no_currency = assess(Pair().set(header, "12000"), tmp_path, "collateral")
    assert {r["result_kind"] for r in no_currency.found("P13", "OPT_RENT_CURRENT")} == {
        "violation", "context",
    }
