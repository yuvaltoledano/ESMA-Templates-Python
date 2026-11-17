"""The fast path gives exactly what the ordinary path gives."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import polars as pl
import pytest

from esma_milan.quality.audit_read import RawFile, read_file
from esma_milan.quality.catalogue import FIELDS
from esma_milan.quality.column_rules import assess_columns
from esma_milan.quality.column_rules.context import PoolFacts, pool_facts
from esma_milan.quality.column_rules.fast import plain_expr
from tests.quality.builders import Pair

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _both(raw: RawFile, pool: PoolFacts) -> None:
    """Assess `raw` with and without the fast path; everything must be equal."""
    fast = assess_columns(raw, pool, fast_path=True)
    slow = assess_columns(raw, pool, fast_path=False)
    order = ["record", "column_position", "rule_id", "result_kind", "reason"]
    a = fast.results.with_columns(pl.col(pl.Categorical).cast(pl.String)).sort(order, nulls_last=True)
    b = slow.results.with_columns(pl.col(pl.Categorical).cast(pl.String)).sort(order, nulls_last=True)
    assert a.equals(b), "result rows differ"
    assert sorted(map(dataclasses.astuple, fast.outcomes), key=str) == sorted(
        map(dataclasses.astuple, slow.outcomes), key=str
    ), "outcomes differ"
    for mine, theirs in zip(fast.profiles, slow.profiles, strict=True):
        assert mine == theirs, (mine.entry.code, "profiles differ")


@pytest.mark.parametrize("fixture", ["synthetic", "real_anonymised"])
def test_equal_on_the_fixtures(fixture: str) -> None:
    loans = read_file(FIXTURES / fixture / "loans.csv", "loans")
    pool = pool_facts(loans)
    _both(loans, pool)
    _both(read_file(FIXTURES / fixture / "collaterals.csv", "collateral"), pool)


def _column_pair(values: dict[str, list[str]], n: int) -> Pair:
    pair = Pair()
    loan, collateral = pair.loans[0], pair.collateral[0]
    pair.loans = [{**loan, "RREL2": f"L{i}", "RREL3": f"L{i}"} for i in range(n)]
    pair.collateral = [{**collateral, "RREC2": f"L{i}", "RREC4": f"P{i}"} for i in range(n)]
    for header, column in values.items():
        rows = pair.loans if header.startswith("RREL") else pair.collateral
        for i, value in enumerate(column):
            rows[i][header] = value
    return pair


BOUNDARIES: dict[str, list[str]] = {
    # 13 and 14 integer digits, 5 and 6 fractional, zero, sign, padding, ND forms, markers
    "RREL30": ["1234567890123", "12345678901234", "1.12345", "1.123456", "0", "0.00", "-5", " 12 ",
               "ND5", "ND1", "ND4-2030-01-01", "NA", "", "1e3", "+7", "007.50", ".5", "1.", "abc"],
    "RREL16": ["0", "1", "0.0", "100000.5", "ND1", "N/A", "25000", "25000", "1234567890123.12345"],
    "RREL29": ["100", "100.00", "0100", "100", "12345678901234567890", "5", "-1", "ND5", " 9"],
    # a percentage: 11 digits, the scale test, RREL59 bounds on the slow path
    "RREL43": ["2.5", "0.5", "12345.123456", "123456.1", "1.1234567", "0", "100", "-0.25", "ND5"],
    "RREL40": ["0.28", "0.35", "0.99", "0.5", "0.1", "0.01", "0.999999", "0.2", "0.3"],
    "RREL59": ["0", "100", "100.5", "150", "50", "-1", "ND5", "10", "10"],
    # integers: the placeholder 9999 and zero
    "RREL25": ["12", "9999", "0", "10000", "0012", "12.0", "ND5", "360", "360"],
    "RREL68": ["0", "0", "30", "9999", "ND5", "45", "-3", "1.5", "90"],
    # text: n and n + 1 characters, lower case, padding, ND forms, markers, loader tokens
    "RREL82": ["A" * 100, "A" * 101, "Lower", " PADDED", "PADDED ", "ND5", "ND4X", "ND4-2030-01-01",
               "NA", "ND", "NULL", "-", "nd5", "ND6", "PLAIN NAME", "PLAIN NAME", "CAFÉ", "A\x1fB", " "],
    "RREL22": ["N", "N", "N", "N", "N", "N", "N", "N", "N"],
    "RREL2": ["L0", "L1", "L1", " L3", "ND4000123", "ND4-ALPHA", "ND4", "", "lower"],
    "RREL1": ["X", "Y", "ND5", "", "X", "X", "X", "X", "X"],
}


@pytest.mark.parametrize("header", sorted(BOUNDARIES))
def test_equal_on_constructed_columns_at_every_boundary(tmp_path: Path, header: str) -> None:
    values = BOUNDARIES[header]
    loans_path, _ = _column_pair({header: values}, len(values)).write(tmp_path)
    loans = read_file(loans_path, "loans")
    _both(loans, pool_facts(loans))


def test_equal_with_the_currency_evidence_varying(tmp_path: Path) -> None:
    n = 12
    values = {
        "RREL30": [str(1000 + i) for i in range(n)],
        "RREL30_CURRENCY": ["EUR", "", "USD", "EURO", "ND5", "EUR", "", "", "XQZ", "EUR", "eur", "EUR"],
        "RREL28": ["EUR", "EUR", "", "", "ND1", "EUR", "", "GBP", "EUR", "EUR", "", "EUR"],
        "RREL31": [str(5 + i) for i in range(n)],
        "RREL31_CURRENCY": ["EUR"] * 6 + [""] * 6,
    }
    pair = _column_pair({k: v for k, v in values.items() if k.startswith("RREL3") and "_" not in k}, n)
    for header in ("RREL30_CURRENCY", "RREL28", "RREL31_CURRENCY"):
        for i, value in enumerate(values[header]):
            pair.loans[i][header] = value
    loans_path, _ = pair.write(tmp_path)
    loans = read_file(loans_path, "loans")
    _both(loans, pool_facts(loans))


def test_equal_on_collateral_fields_tied_to_the_collateral_type(tmp_path: Path) -> None:
    types = ["RBLD", "CARX", "GUAR", "OTHR", "", "CBLD", "MIXD", "RBLD", "OTRE"]
    pair = _column_pair({
        "RREC5": types,
        "RREC12": ["65.5", "65.5", "40", "30", "20", "ND5", "0", "999.99999", "70.25"],
        "RREC16": ["80", "80", "ND5", "55", "60", "75.5", "90", "999.999990", "12345.123456"],
        "RREC8": ["1", "1", "1", "2", "0", "9999", "ND5", "3", "1"],
        "RREC13": ["250000", "0", "1500.50", "ND5", "100", "100", "-1", "250000.123456", "1"],
    }, len(types))
    loans_path, collateral_path = pair.write(tmp_path)
    pool = pool_facts(read_file(loans_path, "loans"))
    _both(read_file(collateral_path, "collateral"), pool)


def _random_value(rng: random.Random, kind: str) -> str:
    roll = rng.random()
    if roll < 0.04:
        return rng.choice(["", "ND5", "ND1", "NA", "N/A", "ND4-2030-01-01", "ND4", " ", "nd5"])
    if kind == "text":
        alphabet = "ABCDEFGHIJ0123456789 -_/" + ("abc" if roll < 0.1 else "") + ("é" if roll < 0.06 else "")
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        return f" {text}" if roll > 0.97 else text
    digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 15)))
    fraction = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 7)))
    number = f"{digits}.{fraction}" if fraction else digits
    if roll > 0.95:
        number = rng.choice(["-", "+", " ", "0", "1e", "1,"]) + number
    return number


def test_equal_on_random_columns_with_a_fixed_seed(tmp_path: Path) -> None:
    rng = random.Random(20261117)
    n = 400
    pair = _column_pair({
        "RREL30": [_random_value(rng, "number") for _ in range(n)],
        "RREL43": [_random_value(rng, "number") for _ in range(n)],
        "RREL25": [_random_value(rng, "number")[:4] for _ in range(n)],
        "RREL82": [_random_value(rng, "text") for _ in range(n)],
        "RREL5": [_random_value(rng, "text") for _ in range(n)],
        "RREC13": [_random_value(rng, "number") for _ in range(n)],
        "RREC12": [_random_value(rng, "number") for _ in range(n)],
        "RREC5": [rng.choice(["RBLD", "RBLD", "CARX", "OTHR", "GUAR"]) for _ in range(n)],
        "RREC3": [_random_value(rng, "text") for _ in range(n)],
    }, n)
    loans_path, collateral_path = pair.write(tmp_path)
    loans = read_file(loans_path, "loans")
    pool = pool_facts(loans)
    _both(loans, pool)
    _both(read_file(collateral_path, "collateral"), pool)


def test_which_fields_can_be_plain() -> None:
    assert plain_expr(FIELDS["RREL30"], {}) is not None
    assert plain_expr(FIELDS["RREL2"], {}) is not None
    # A value rule the fast path does not carry: always the ordinary path.
    for code in ("RREL59", "RREL1", "RREC1", "RREL23", "RREL12", "RREL10", "RREL69", "RREL11",
                 "RREL81", "RREL28", "RREL80"):
        assert plain_expr(FIELDS[code], {}) is None, code
    # Tied to the collateral type: plain only when that cell is available.
    assert plain_expr(FIELDS["RREC12"], {}) is None
    assert plain_expr(FIELDS["RREC12"], {"collateral_type": "_x_collateral_type"}) is not None


def test_the_fast_path_is_used_and_conservative(tmp_path: Path) -> None:
    values = ["150000.00", "0", " 12", "ND5", "1e3", "12345678901234", "99.123456", "250000"]
    frame = pl.DataFrame({"_v": values})
    plain = frame.select(plain_expr(FIELDS["RREL30"], {}).alias("p"))["p"].to_list()  # type: ignore[union-attr]
    assert plain == [True, False, False, False, False, False, False, True]
    text = pl.DataFrame({"_v": ["LOAN 1", " LOAN", "LOAN ", "loan", "ND5", "NA", "ND4X", "-", "A" * 1001, "nd5"]})
    plain = text.select(plain_expr(FIELDS["RREL2"], {}).alias("p"))["p"].to_list()  # type: ignore[union-attr]
    assert plain == [True] + [False] * 9
