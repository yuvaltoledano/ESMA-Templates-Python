"""P8 (possible placeholder), P9 (cardinality) and P12 (plausibility windows)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.quality.builders import Pair, assess

PLACEHOLDER = "possible placeholder; confirm whether this is a measurement"


@pytest.mark.parametrize(
    ("field", "value", "severity"),
    [("RREL16", "0", "Warn"), ("RREL16", "0.00", "Warn"), ("RREL34", "0", "Info"),
     ("RREL25", "9999", "Info"), ("RREL58", "9999", "Warn"), ("RREL68", "9999", "Warn")],
)
def test_p8_placeholder_values_in_loan_cells(tmp_path: Path, field: str, value: str, severity: str) -> None:
    loans = assess(Pair().set(field, value, record=1), tmp_path, "loans")
    rows = loans.found("P8", field)
    assert [(r["record"], r["result_kind"], r["severity"]) for r in rows] == [(1, "question", severity)]
    assert str(rows[0]["expected"]).startswith(PLACEHOLDER)
    assert loans.outcome("P8", field) == (2, 2, 0, 0)  # a question is not a failure


def test_p8_placeholder_values_in_collateral_cells(tmp_path: Path) -> None:
    pair = Pair().set("RREC13", "0", record=1).set("RREC12", "999.99999", record=2)
    pair.set("RREC8", "9999", record=1)
    collateral = assess(pair, tmp_path, "collateral")
    assert sorted((r["field"], r["record"]) for r in collateral.found("P8")) == [
        ("RREC12", 2), ("RREC13", 1), ("RREC8", 1),
    ]


def test_p8_never_on_measured_zeros(tmp_path: Path) -> None:
    loans = assess(Pair().set("RREL20", "0").set("RREL59", "0").set("RREL30", "0"), tmp_path, "loans")
    assert loans.found("P8") == []
    assert assess(Pair(), tmp_path, "loans").found("P8") == []


def test_p8_zero_on_every_record(tmp_path: Path) -> None:
    every = assess(Pair().set("RREL33", "0"), tmp_path, "loans")
    assert [(r["field"], r["record"], r["reason"]) for r in every.found("P8")] == [
        ("RREL33", None, "zero_on_every_record")
    ]
    assert assess(Pair().set("RREL33", "0", record=1), tmp_path, "loans").found("P8") == []
    # RREC21: only when RREC20 holds no sale date at all.
    assert [r["field"] for r in assess(Pair().set("RREC21", "0"), tmp_path, "collateral").found("P8")] == ["RREC21"]
    sold = Pair().set("RREC21", "0").set("RREC20", "2023-01-01", record=1)
    assert assess(sold, tmp_path, "collateral").found("P8") == []


def test_p8_single_token_in_a_free_text_field(tmp_path: Path) -> None:
    token = assess(Pair().set("RREL22", "N"), tmp_path, "loans")
    assert [(r["field"], r["reason"]) for r in token.found("P8")] == [
        ("RREL22", "single_token_on_every_record")
    ]
    assert assess(Pair().set("RREL22", "N", record=1), tmp_path, "loans").found("P8") == []
    assert assess(Pair(), tmp_path, "loans").found("P8") == []  # "HELP TO BUY" has spaces


def test_p9_constant_column(tmp_path: Path) -> None:
    loans = assess(Pair(), tmp_path, "loans")
    constant = {r["field"] for r in loans.found("P9")}
    assert "RREL82" in constant and "RREL69" in constant
    assert "RREL2" not in constant and "RREL30" not in constant
    row = loans.found("P9", "RREL82")[0]
    assert (row["result_kind"], row["severity"], row["record"]) == ("context", "Info", None)


def test_p12_date_windows(tmp_path: Path) -> None:
    pair = Pair().set("RREL23", "1949-12-31", record=1).set("RREL24", "2084-07-01", record=1)
    pair.set("RREL24", "2084-06-30", record=2)  # exactly 60 years after the cut-off
    pair.set("RREL60", "1900-01-01", record=2)  # a field that does not reach MILAN
    loans = assess(pair, tmp_path, "loans")
    assert [(r["field"], r["record"], r["reason"], r["severity"]) for r in loans.found("P12")] == [
        ("RREL23", 1, "date_before_window", "Warn"),
        ("RREL24", 1, "date_after_window", "Warn"),
        ("RREL60", 2, "date_before_window", "Info"),
    ]
    assert all(r["result_kind"] == "question" for r in loans.found("P12"))
    assert all("please confirm" in str(r["expected"]) for r in loans.found("P12"))


def test_p12_percentage_column_all_below_one(tmp_path: Path) -> None:
    fraction = assess(Pair().set("RREL40", "0.28", record=1).set("RREL40", "0.35", record=2), tmp_path, "loans")
    assert [(r["field"], r["record"], r["severity"]) for r in fraction.found("P12")] == [
        ("RREL40", None, "Warn")
    ]
    mixed = Pair().set("RREL40", "0.28", record=1).set("RREL40", "28", record=2)
    assert assess(mixed, tmp_path, "loans").found("P12") == []
    assert assess(Pair().set("RREL40", "0"), tmp_path, "loans").found("P12") == []
    # Margins and RREL59 are normally below 1 and are excluded.
    excluded = Pair()
    for code in ("RREL46", "RREL50", "RREL52", "RREL54", "RREL59"):
        excluded.set(code, "0.75")
    assert assess(excluded, tmp_path, "loans").found("P12") == []


def test_p12_lien_position_zero(tmp_path: Path) -> None:
    collateral = assess(Pair().set("RREC8", "0", record=1), tmp_path, "collateral")
    assert [(r["field"], r["record"], r["reason"]) for r in collateral.found("P12")] == [
        ("RREC8", 1, "zero_is_not_a_lien_position")
    ]
