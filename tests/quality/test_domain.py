"""P6 (value domains) and P11 (reference lists)."""

from __future__ import annotations

from pathlib import Path

from esma_milan.quality.column_rules.domain import classification_year_results
from esma_milan.quality.refdata import load_reference_data
from tests.quality.builders import Assessed, Pair, assess

REF = load_reference_data()


def test_p6a_amounts_that_must_not_be_negative(tmp_path: Path) -> None:
    pair = Pair()
    for code in ("RREL29", "RREL67", "RREL30", "RREL39", "RREL73", "RREL74"):
        pair.set(code, "-1000", record=1)
    loans = assess(pair, tmp_path, "loans")
    assert sorted(str(r["field"]) for r in loans.found("P6(a)")) == ["RREL29", "RREL67"]
    for code in ("RREL30", "RREL39", "RREL73", "RREL74"):  # may be negative: no sign finding
        assert loans.rules_on(code) == set()
        assert loans.outcome("P6(a)", code) == (0, 0, 0, 0)
    assert loans.outcome("P6(a)", "RREL29") == (2, 2, 1, 0)
    collateral = assess(Pair().set("RREC13", "-1", record=2), tmp_path, "collateral")
    assert [(r["field"], r["record"]) for r in collateral.found("P6(a)")] == [("RREC13", 2)]


def test_p6a_is_not_evaluable_on_a_value_that_does_not_parse(tmp_path: Path) -> None:
    loans = assess(Pair().set("RREL29", "abc", record=1), tmp_path, "loans")
    assert loans.outcome("P6(a)", "RREL29") == (2, 1, 0, 1)
    assert loans.rules_on("RREL29") == {("P5", "violation")}


def test_p6b_dates_not_later_than_the_cut_off(tmp_path: Path) -> None:
    pair = Pair().set("RREL7", "2024-07-01", record=1).set("RREL7", "2024-06-30", record=2)
    pair.set("RREL24", "2020-01-01")  # a maturity date before the cut-off: no published rule
    pair.set("RREL60", "2020-01-01")  # nor for a lock-out end date
    loans = assess(pair, tmp_path, "loans")
    rows = loans.found("P6(b)")
    assert [(r["field"], r["record"], r["result_kind"], r["note"]) for r in rows] == [
        ("RREL7", 1, "violation", None)
    ]
    assert loans.rules_on("RREL24") == set() and loans.rules_on("RREL60") == set()
    assert loans.outcome("P6(b)", "RREL7") == (2, 2, 1, 0)


def test_p6b_on_collateral_says_where_the_cut_off_came_from(tmp_path: Path) -> None:
    own = assess(Pair().set("RREC15", "2024-07-01", record=1), tmp_path, "collateral")
    assert [(r["record"], r["note"]) for r in own.found("P6(b)", "RREC15")] == [(1, None)]
    pair = Pair().set("RREC15", "2024-07-01", record=1).drop("Pool_Cutoff_Date", "collateral")
    borrowed = assess(pair, tmp_path, "collateral")
    assert [(r["record"], r["note"]) for r in borrowed.found("P6(b)", "RREC15")] == [
        (1, "pool cut-off taken from the loans file")
    ]


def test_p6b_without_an_established_cut_off(tmp_path: Path) -> None:
    pair = Pair().set("RREL7", "2030-01-01").set("RREL6", "2024-06-29", record=1)
    loans = assess(pair, tmp_path, "loans")
    assert [r["record"] for r in loans.found("P6(b)", "RREL7")] == [2]
    assert loans.outcome("P6(b)", "RREL7") == (2, 1, 1, 1)
    # Two cut-offs in the loans file: a collateral file without its own has none.
    pair = Pair().set("RREL6", "2024-05-31", record=1).set("Pool_Cutoff_Date", "2024-05-31", record=1)
    pair.drop("Pool_Cutoff_Date", "collateral").set("RREC15", "2030-01-01")
    collateral = assess(pair, tmp_path, "collateral")
    assert collateral.found("P6(b)") == []
    assert collateral.outcome("P6(b)", "RREC15") == (2, 0, 0, 2)


def test_p6c_and_p6d_ranges(tmp_path: Path) -> None:
    pair = Pair().set("RREL59", "100", record=1).set("RREL59", "100.5", record=2)
    pair.set("RREL12", "1980", record=1)
    loans = assess(pair, tmp_path, "loans")
    assert [r["record"] for r in loans.found("P6(c)")] == [2]
    assert [r["record"] for r in loans.found("P6(d)")] == [1]
    assert assess(Pair().set("RREL59", "0"), tmp_path, "loans").found("P6(c)") == []
    assert assess(Pair().set("RREL59", "-1", record=1), tmp_path, "loans").found("P6(c)") != []


def test_p11_country_and_currency(tmp_path: Path) -> None:
    pair = Pair().set("RREL81", "XX", record=1).set("RREL84", "GR", record=1)
    pair.set("RREL28", "DEM", record=1).set("RREL18", "XQZ", record=2)
    loans = assess(pair, tmp_path, "loans")
    assert [(r["field"], r["record"]) for r in loans.found("P11")] == [("RREL18", 2), ("RREL81", 1)]
    assert loans.rules_on("RREL28") == set()  # a historic currency is valid
    assert loans.rules_on("RREL81") == {("P11", "violation")}  # the format is right: no P5 row


def _region(tmp_path: Path, code: str, year: str = "2021") -> Assessed:
    return assess(Pair().set("RREL11", code, record=1).set("RREL12", year), tmp_path, "loans")


def test_p11_nuts_membership(tmp_path: Path) -> None:
    assert _region(tmp_path, "NL329").rules_on("RREL11") == set()
    assert _region(tmp_path, "NL999").rules_on("RREL11") == {("P11", "violation")}
    # DE915 is a NUTS 2013 code that NUTS 2016 recoded.
    # Under 2013, DE915 (record 1) is a member; the pair's NL329 (record 2) was created in 2016.
    assert [r["record"] for r in _region(tmp_path, "DE915", "2013").found("P11", "RREL11")] == [2]
    recoded = assess(
        Pair().set("RREL11", "DE915").set("RREL12", "2013", record=1).set("RREL12", "2016", record=2),
        tmp_path, "loans",
    )
    assert [(r["record"], r["result_kind"]) for r in recoded.found("P11", "RREL11")] == [
        (2, "violation")  # the record's own year decides
    ]


def test_p11_zzz_codes(tmp_path: Path) -> None:
    for code in ("GBZZZ", "ATZZZ", "ELZZZ", "USZZZ"):  # ISO code or NUTS prefix
        assert _region(tmp_path, code).rules_on("RREL11") == set(), code
    assert _region(tmp_path, "QQZZZ").rules_on("RREL11") == {("P11", "violation")}


def test_p11_limited_assessment_is_never_invalid(tmp_path: Path) -> None:
    uncovered = _region(tmp_path, "UKI31", "2024")  # the 2024 list does not cover the UK
    row = uncovered.found("P11", "RREL11")[0]
    assert (row["result_kind"], row["reason"]) == ("limitation", "limited_assessment_shape_only")
    assert "UK" in str(row["expected"])
    assert _region(tmp_path, "UKI31", "2021").rules_on("RREL11") == set()
    not_bundled = _region(tmp_path, "NL329", "2006")
    assert {r["result_kind"] for r in not_bundled.found("P11", "RREL11")} == {"limitation"}
    assert not_bundled.outcome("P11", "RREL11") == (2, 0, 0, 2)
    no_year = _region(tmp_path, "NL329", "ND1")
    assert {r["result_kind"] for r in no_year.found("P11", "RREL11")} == {"limitation"}


def test_p11_collateral_regions_use_the_loans_files_single_year(tmp_path: Path) -> None:
    collateral = assess(Pair().set("RREC6", "NL999", record=1), tmp_path, "collateral")
    row = collateral.found("P11", "RREC6")[0]
    assert (row["record"], row["result_kind"]) == (1, "violation")
    assert row["note"] == "classification year taken from the loans file"
    two_years = assess(Pair().set("RREL12", "2016", record=1), tmp_path, "collateral")
    assert {r["result_kind"] for r in two_years.found("P11", "RREC6")} == {"limitation"}


def _year_rows(tmp_path: Path, pair: Pair) -> list[tuple[str, str]]:
    loans, collateral = assess(pair, tmp_path, "loans"), assess(pair, tmp_path, "collateral")
    profiles = loans.assessment.profiles + collateral.assessment.profiles
    rows, _ = classification_year_results(
        [p for p in profiles if p.entry.code == "RREL12"],
        [p for p in profiles if p.entry.kind == "NUTS"],
        REF,
    )
    return [(str(r["result_kind"]), str(r["reason"])) for r in rows]


def test_p11_classification_year_of_the_submission(tmp_path: Path) -> None:
    assert _year_rows(tmp_path, Pair()) == []
    assert _year_rows(tmp_path, Pair().set("RREL12", "2016", record=1)) == [
        ("violation", "more_than_one_classification_year")
    ]
    assert _year_rows(tmp_path, Pair().set("RREL12", "2006")) == [
        ("limitation", "limited_assessment_shape_only")
    ]
    # Q&A 1313: when every region is a country-plus-ZZZ code, any year is accepted.
    all_zzz = Pair().set("RREL12", "2006").set("RREL11", "GBZZZ").set("RREC6", "GBZZZ")
    assert _year_rows(tmp_path, all_zzz) == []
