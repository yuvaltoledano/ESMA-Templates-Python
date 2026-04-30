"""Tests for Stage-9 / B-3 MILAN date derivations.

Mirrors r_reference/R/milan_mapping.R:556-597 (date pre-compute) and
:704-727 (Months Current + Months In Arrears case_when / if_else).

Pinning strategy: all R-emitted output strings are taken from the
synthetic fixture (`tests/fixtures/synthetic/expected_r_output.xlsx`,
sheet "MILAN template pool"), not from re-running R. The fixture is
the parity contract; if R's formatting drifts, the parity tests will
catch it - and these unit tests' assertions stay aligned with the
emitted-byte-equal contract.

Pinned values (verified against the fixture):
  - "Never in Arrears"  - exact case, exact spacing
  - "0"                  - integer-valued floats, no trailing ".0"
  - "12.517453798768"   - 381 days / 30.4375 (NEW_005 in fixture)
  - "1.47843942505133"  - 45  days / 30.4375 (NEW_008 in fixture)
"""

from __future__ import annotations

from datetime import date

import polars as pl

from esma_milan.pipeline.milan_map import (
    _attach_date_derivations,
    _compute_months_current_expr,
    _compute_months_in_arrears_expr,
    _r_as_character_expr,
)

# -----------------------------------------------------------------------------
# String constants pinned to the fixture's R output (NOT to R-source literals).
# If R's formatting ever changes, the parity test for "MILAN template pool"
# will fail at the same time and these values get updated together.
# -----------------------------------------------------------------------------
NEVER_IN_ARREARS = "Never in Arrears"
MONTHS_CUR_NEW_005 = "12.517453798768"   # NEW_005: 2024-06-30 - 2023-06-15 -> 381 days / 30.4375
MONTHS_IA_NEW_008 = "1.47843942505133"   # NEW_008: 45 / 30.4375


# ---------------------------------------------------------------------------
# _r_as_character_expr - format pin
# ---------------------------------------------------------------------------


def test_r_as_character_matches_fixture_strings() -> None:
    """%.15g formatting reproduces R's `as.character()` output for the
    exact float values the fixture has in "Months Current" and
    "Months In Arrears"."""
    df = pl.DataFrame(
        {
            "x": [
                381 / 30.4375,
                45 / 30.4375,
                0.0,
                None,
            ]
        },
        schema={"x": pl.Float64},
    )
    out = df.select(_r_as_character_expr(pl.col("x")).alias("s"))["s"].to_list()
    assert out == [MONTHS_CUR_NEW_005, MONTHS_IA_NEW_008, "0", None]


def test_r_as_character_strips_trailing_decimal_for_integer_floats() -> None:
    """1.0 -> "1", 1234567.0 -> "1234567" (no trailing ".0", matches R)."""
    df = pl.DataFrame({"x": [1.0, 1234567.0, -2.0]}, schema={"x": pl.Float64})
    out = df.select(_r_as_character_expr(pl.col("x")).alias("s"))["s"].to_list()
    assert out == ["1", "1234567", "-2"]


# ---------------------------------------------------------------------------
# _attach_date_derivations
# ---------------------------------------------------------------------------


def _df(
    *,
    pool_cutoff: list[str | None],
    last_arr: list[str | None],
    days_in_arrears: list[str | None],
    primary_income: list[str | None] | None = None,
    secondary_income: list[str | None] | None = None,
    total_credit_limit: list[str | None] | None = None,
) -> pl.DataFrame:
    n = len(pool_cutoff)
    return pl.DataFrame(
        {
            "pool_cutoff_date": pool_cutoff,
            "date_last_in_arrears": last_arr,
            "number_of_days_in_arrears": days_in_arrears,
            "primary_income": primary_income or [None] * n,
            "secondary_income": secondary_income or [None] * n,
            "total_credit_limit": total_credit_limit or [None] * n,
        },
        schema={
            "pool_cutoff_date": pl.Utf8,
            "date_last_in_arrears": pl.Utf8,
            "number_of_days_in_arrears": pl.Utf8,
            "primary_income": pl.Utf8,
            "secondary_income": pl.Utf8,
            "total_credit_limit": pl.Utf8,
        },
    )


def test_months_since_last_arrears_pins_fixture_value_for_new_005() -> None:
    """NEW_005's exact dates -> exact fixture string, byte-equal."""
    df = _df(
        pool_cutoff=["2024-06-30"],
        last_arr=["2023-06-15"],
        days_in_arrears=["0"],
    )
    out = _attach_date_derivations(df)
    assert out["_months_since_last_arrears"].to_list() == [MONTHS_CUR_NEW_005]
    assert out["_pool_cutoff_safe"].to_list() == [date(2024, 6, 30)]
    assert out["_date_last_in_arrears_safe"].to_list() == [date(2023, 6, 15)]
    assert out["_days_in_arrears_num"].to_list() == [0.0]


def test_months_since_last_arrears_null_when_either_date_null() -> None:
    """ND in either date column -> _months_since_last_arrears is null."""
    df = _df(
        pool_cutoff=["2024-06-30", "ND",         "2024-06-30", "ND"],
        last_arr=["ND",         "2024-06-30", "",           "ND"],
        days_in_arrears=["0", "0", "0", "0"],
    )
    out = _attach_date_derivations(df)
    assert out["_months_since_last_arrears"].to_list() == [None, None, None, None]


def test_safe_as_num_columns_collapse_nd() -> None:
    """primary/secondary income + total_credit_limit go through _safe_as_num_expr."""
    df = _df(
        pool_cutoff=["2024-06-30", "2024-06-30"],
        last_arr=["2023-06-15", "2023-06-15"],
        days_in_arrears=["0", "ND"],
        primary_income=["50000", "ND"],
        secondary_income=["", "20000"],
        total_credit_limit=["100000", "ND3"],
    )
    out = _attach_date_derivations(df)
    assert out["_primary_income_num"].to_list() == [50000.0, None]
    assert out["_secondary_income_num"].to_list() == [None, 20000.0]
    assert out["_total_credit_limit_num"].to_list() == [100000.0, None]
    assert out["_days_in_arrears_num"].to_list() == [0.0, None]


def test_date_last_in_arrears_chr_preserves_nd_for_rule_3() -> None:
    """The chr column keeps ND/blank tokens so rule 3's `is_nd` check fires.

    Without the char-preserving column, a parsed-Date null would be
    indistinguishable from a real null Date - rule 3 needs to see the
    original ND string to know "this is intentional, never in arrears"
    vs "this is unparseable, fall to ND".
    """
    df = _df(
        pool_cutoff=["2024-06-30", "2024-06-30"],
        last_arr=["ND", "2024-05-15"],
        days_in_arrears=["0", "0"],
    )
    out = _attach_date_derivations(df)
    assert out["_date_last_in_arrears_chr"].to_list() == ["ND", "2024-05-15"]


# ---------------------------------------------------------------------------
# _compute_months_current_expr - rule application
# ---------------------------------------------------------------------------


def _months_current(
    pool_cutoff: list[str | None],
    last_arr: list[str | None],
    days_in_arrears: list[str | None],
) -> list[str]:
    df = _attach_date_derivations(
        _df(pool_cutoff=pool_cutoff, last_arr=last_arr, days_in_arrears=days_in_arrears)
    )
    return df.select(
        _compute_months_current_expr(
            pl.col("_days_in_arrears_num"),
            pl.col("_months_since_last_arrears"),
            pl.col("_date_last_in_arrears_chr"),
        ).alias("mc")
    )["mc"].to_list()


def test_months_current_rule_1_in_arrears_emits_zero_string() -> None:
    """days > 0 -> "0" regardless of last-arrears date or pool cutoff."""
    out = _months_current(
        pool_cutoff=["2024-06-30", "2024-06-30"],
        last_arr=["2024-05-15", "ND"],
        days_in_arrears=["15", "200"],
    )
    assert out == ["0", "0"]


def test_months_current_rule_2_pins_fixture_string() -> None:
    """days=0 + valid dates -> exact fixture-emitted months string."""
    out = _months_current(
        pool_cutoff=["2024-06-30"],
        last_arr=["2023-06-15"],
        days_in_arrears=["0"],
    )
    assert out == [MONTHS_CUR_NEW_005]


def test_months_current_rule_3_emits_never_in_arrears() -> None:
    """days=0 + ND last-arrears date -> "Never in Arrears" (fixture-pinned)."""
    out = _months_current(
        pool_cutoff=["2024-06-30", "2024-06-30", "2024-06-30"],
        last_arr=["ND", "ND3", ""],
        days_in_arrears=["0", "0", "0"],
    )
    assert out == [NEVER_IN_ARREARS] * 3


def test_months_current_fallback_when_days_in_arrears_null() -> None:
    """days null (unparseable) -> "ND" fallback, regardless of dates."""
    out = _months_current(
        pool_cutoff=["2024-06-30", "2024-06-30"],
        last_arr=["2023-06-15", "ND"],
        days_in_arrears=["ND", "abc"],
    )
    assert out == ["ND", "ND"]


# ---------------------------------------------------------------------------
# _compute_months_in_arrears_expr
# ---------------------------------------------------------------------------


def test_months_in_arrears_pins_fixture_value_for_new_008() -> None:
    """NEW_008's days=45 -> exact fixture string, byte-equal."""
    df = pl.DataFrame({"days": [45.0]}, schema={"days": pl.Float64})
    out = df.select(
        _compute_months_in_arrears_expr(pl.col("days")).alias("mia")
    )["mia"].to_list()
    assert out == [MONTHS_IA_NEW_008]


def test_months_in_arrears_zero_days() -> None:
    """days=0 -> "0" (integer-valued float, no trailing decimal)."""
    df = pl.DataFrame({"days": [0.0]}, schema={"days": pl.Float64})
    out = df.select(
        _compute_months_in_arrears_expr(pl.col("days")).alias("mia")
    )["mia"].to_list()
    assert out == ["0"]


def test_months_in_arrears_null_emits_nd() -> None:
    """days null -> "ND" (matches R if_else)."""
    df = pl.DataFrame({"days": [None]}, schema={"days": pl.Float64})
    out = df.select(
        _compute_months_in_arrears_expr(pl.col("days")).alias("mia")
    )["mia"].to_list()
    assert out == ["ND"]
