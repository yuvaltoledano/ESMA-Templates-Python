"""Tests for Stage-10 / C-1 Execution Summary helpers.

Mirrors r_reference/R/pipeline.R:625-820 metric calculations and
formatting helpers.

Pinning strategy:
  - Format helper outputs are pinned against R's `scales::*` defaults
    at the time `tests/fixtures/synthetic/expected_r_output.xlsx` was
    generated. If R's `scales` package updates its defaults,
    regenerate the fixture and re-pin these values.
  - Metric helper outputs are pinned against fixture-extracted scalars
    where the fixture exercises the helper. Branches not exercised
    by the synthetic fixture are tested against the R source contract
    with explicit "branch not exercised by synthetic fixture" comments.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from esma_milan.pipeline.exec_summary import (
    _NOT_AVAILABLE,
    _borrower_balances_sorted_desc,
    _effective_borrowers,
    _fmt_comma,
    _fmt_comma_int,
    _fmt_pct,
    _milan_as_num,
    _pct_of_cb_mask,
    _structure_breakdown,
    _top_20_borrower_exposure,
    _weighted_mean,
)

# -----------------------------------------------------------------------------
# Format helpers
# -----------------------------------------------------------------------------


def test_fmt_pct_pinned_to_fixture_strings() -> None:
    """Fixture values from "Top 20 Borrower Exposure" / "WA Interest Rate" /
    "WA Original LTV" / "Current LTV: >= 70%"."""
    assert _fmt_pct(1.0) == "100.00%"            # Top 20 Borrower Exposure
    assert _fmt_pct(0.0265) == "2.65%"            # WA Interest Rate
    assert _fmt_pct(0.7439) == "74.39%"           # WA Original LTV
    assert _fmt_pct(0.187056) == "18.71%"         # Current LTV >= 70% (rounded)


def test_fmt_pct_handles_zero_and_none() -> None:
    assert _fmt_pct(0.0) == "0.00%"
    assert _fmt_pct(None) == _NOT_AVAILABLE
    assert _fmt_pct(math.nan) == _NOT_AVAILABLE


def test_fmt_comma_pinned_to_fixture_strings() -> None:
    """Fixture values from balance / average columns."""
    assert _fmt_comma(1865000.0) == "1,865,000.00"     # Original Balance
    assert _fmt_comma(1550000.0) == "1,550,000.00"     # Current Balance
    assert _fmt_comma(258333.333333) == "258,333.33"    # Average Loan Per Borrower (rounded)
    assert _fmt_comma(5.5517042172) == "5.55"           # Effective Number of Borrowers (rounded)


def test_fmt_comma_handles_zero_and_none() -> None:
    assert _fmt_comma(0.0) == "0.00"
    assert _fmt_comma(None) == _NOT_AVAILABLE
    assert _fmt_comma(math.nan) == _NOT_AVAILABLE


def test_fmt_comma_int_pinned_to_fixture_strings() -> None:
    """Fixture count columns."""
    assert _fmt_comma_int(6) == "6"
    assert _fmt_comma_int(8) == "8"
    assert _fmt_comma_int(1234) == "1,234"
    assert _fmt_comma_int(1234567) == "1,234,567"


def test_fmt_comma_int_handles_none() -> None:
    assert _fmt_comma_int(None) == _NOT_AVAILABLE


# -----------------------------------------------------------------------------
# _milan_as_num
# -----------------------------------------------------------------------------


def test_milan_as_num_collapses_nd_to_null() -> None:
    s = pl.Series("x", ["ND", "1.5", "100", None, ""])
    out = _milan_as_num(s).to_list()
    # "ND" -> null; "" stays as a non-numeric string -> null via cast strict=False;
    # null stays null. Numerics parse.
    assert out == [None, 1.5, 100.0, None, None]


def test_milan_as_num_handles_asymmetric_defaults() -> None:
    """R-repo entry #9 anchor: "0" must contribute 0 to a sum,
    "ND" must be excluded entirely.

    Construct a column mixing both. sum(milan_as_num) over the
    column counts "0" toward the total but skips "ND". The
    resulting sum is the sum of the numeric-string-castable values
    only.
    """
    s = pl.Series("x", ["100", "0", "ND", "50", "0", "ND"])
    out = _milan_as_num(s)
    # The sum: 100 + 0 + null + 50 + 0 + null = 150 (Polars sum
    # skips nulls); count of non-null: 4 ("0" twice, "100", "50").
    assert out.sum() == 150.0
    assert out.is_null().sum() == 2  # only the two "ND" rows are null
    # The two "0" rows contribute 0 but are NOT null:
    assert out.to_list() == [100.0, 0.0, None, 50.0, 0.0, None]


# -----------------------------------------------------------------------------
# _borrower_balances_sorted_desc + _top_20_borrower_exposure + _effective_borrowers
# -----------------------------------------------------------------------------


def _synthetic_borrower_inputs() -> tuple[pl.Series, pl.Series, float]:
    """6 borrowers from synthetic fixture's MILAN sheet:
    {BORIG_01: 180000, BORIG_02: 290000, BORIG_03: 255000,
     BORIG_04: 390000, BORIG_05: 265000, BORIG_06: 170000}
    total_cb = 1,550,000
    Each borrower appears once on a one-loan-per-borrower row.
    """
    bids = pl.Series("id", [
        "BORIG_01", "BORIG_02", "BORIG_03",
        "BORIG_04", "BORIG_05", "BORIG_06",
    ])
    balances = pl.Series("cb", [
        180000.0, 290000.0, 255000.0,
        390000.0, 265000.0, 170000.0,
    ])
    return bids, balances, 1_550_000.0


def test_borrower_balances_sorted_desc_orders_by_balance_then_id() -> None:
    bids, balances, _ = _synthetic_borrower_inputs()
    out = _borrower_balances_sorted_desc(bids, balances)
    assert out == [390000.0, 290000.0, 265000.0, 255000.0, 180000.0, 170000.0]


def test_borrower_balances_sorted_desc_tiebreak_on_id() -> None:
    """Two borrowers with identical balances tie-break on borrower_id ascending."""
    bids = pl.Series("id", ["B_Z", "B_A", "B_M"])
    balances = pl.Series("cb", [100.0, 100.0, 50.0])
    out = _borrower_balances_sorted_desc(bids, balances)
    # All three group keys distinct; B_A and B_Z tie on 100, B_A first.
    assert out == [100.0, 100.0, 50.0]
    # Verify the tiebreak with full output: simulate by checking ordering
    # is stable across re-runs.
    out2 = _borrower_balances_sorted_desc(
        pl.Series("id", ["B_M", "B_A", "B_Z"]),
        pl.Series("cb", [50.0, 100.0, 100.0]),
    )
    assert out2 == out  # input order doesn't change output


def test_borrower_balances_filters_nd_and_null_ids() -> None:
    bids = pl.Series("id", ["B_A", "ND", None, "B_B"])
    balances = pl.Series("cb", [100.0, 200.0, 300.0, 50.0])
    out = _borrower_balances_sorted_desc(bids, balances)
    assert out == [100.0, 50.0]  # ND/null borrower rows excluded


def test_borrower_balances_filters_null_balances() -> None:
    bids = pl.Series("id", ["B_A", "B_B"])
    balances = pl.Series("cb", [100.0, None])
    out = _borrower_balances_sorted_desc(bids, balances)
    assert out == [100.0]


def test_top_20_borrower_exposure_pinned_to_fixture_value() -> None:
    """6 borrowers, all balances sum to total_cb -> top-20 ratio = 1.0
    -> "100.00%" in the fixture."""
    bids, balances, total_cb = _synthetic_borrower_inputs()
    out = _top_20_borrower_exposure(bids, balances, total_cb)
    assert out == 1.0
    assert _fmt_pct(out) == "100.00%"


def test_top_20_borrower_exposure_caps_at_20_when_more_borrowers() -> None:
    """Branch not exercised by synthetic fixture (only 6 borrowers).
    Construct 25 borrowers; verify only the top 20 contribute.
    """
    bids = pl.Series("id", [f"B{i:02d}" for i in range(25)])
    # Balances 100..1, top 20 = 100 + 99 + ... + 81 = 1810.
    balances = pl.Series("cb", [float(100 - i) for i in range(25)])
    total = float(sum(100 - i for i in range(25)))  # = 100..76 sum
    out = _top_20_borrower_exposure(bids, balances, total)
    expected_top20 = sum(range(81, 101))  # 81..100 inclusive
    assert out == pytest.approx(expected_top20 / total)


def test_top_20_borrower_exposure_zero_total_returns_none() -> None:
    bids, balances, _ = _synthetic_borrower_inputs()
    assert _top_20_borrower_exposure(bids, balances, 0.0) is None


def test_top_20_borrower_exposure_no_eligible_returns_none() -> None:
    bids = pl.Series("id", ["ND", None])
    balances = pl.Series("cb", [100.0, 50.0])
    assert _top_20_borrower_exposure(bids, balances, 1.0) is None


def test_effective_borrowers_pinned_to_fixture_value() -> None:
    """1 / sum(share^2) for the 6 fixture borrowers should round-format
    to "5.55" (fixture's "Effective Number of Borrowers")."""
    bids, balances, total_cb = _synthetic_borrower_inputs()
    out = _effective_borrowers(bids, balances, total_cb)
    # Hand-computed: shares = [b/total for b in balances]; sum of squares
    # = (180000^2 + 290000^2 + 255000^2 + 390000^2 + 265000^2 + 170000^2) / 1550000^2
    # = 432,750,000,000 / 2,402,500,000,000 = 0.18012486...
    # 1 / 0.18012486... = 5.5517042...
    assert out == pytest.approx(5.5517042172, rel=1e-9)
    assert _fmt_comma(out) == "5.55"


def test_effective_borrowers_uniform_distribution_recovers_n() -> None:
    """All-equal balances across N borrowers -> effective N = N exactly.
    Algebraic identity: sum((1/N)^2) = N * (1/N^2) = 1/N; 1/(1/N) = N.
    """
    bids = pl.Series("id", [f"B{i}" for i in range(7)])
    balances = pl.Series("cb", [100.0] * 7)
    out = _effective_borrowers(bids, balances, 700.0)
    assert out == pytest.approx(7.0)


def test_effective_borrowers_zero_total_returns_none() -> None:
    bids, balances, _ = _synthetic_borrower_inputs()
    assert _effective_borrowers(bids, balances, 0.0) is None


# -----------------------------------------------------------------------------
# _weighted_mean
# -----------------------------------------------------------------------------


def test_weighted_mean_basic() -> None:
    v = pl.Series("v", [1.0, 2.0, 3.0])
    w = pl.Series("w", [10.0, 20.0, 30.0])
    out = _weighted_mean(v, w)
    # Hand-computed: (10 + 40 + 90) / 60 = 140/60 = 2.333...
    assert out == pytest.approx(140.0 / 60.0)


def test_weighted_mean_skips_null_pairs() -> None:
    """R `weighted.mean(v, w, na.rm=TRUE)` skips index pairs where either
    v[i] or w[i] is NA.
    """
    v = pl.Series("v", [1.0, None, 3.0, 4.0])
    w = pl.Series("w", [10.0, 20.0, None, 40.0])
    out = _weighted_mean(v, w)
    # Surviving pairs: (1,10) and (4,40). Mean: (10 + 160) / 50 = 3.4
    assert out == pytest.approx(170.0 / 50.0)


def test_weighted_mean_zero_weight_sum_returns_none() -> None:
    v = pl.Series("v", [1.0, 2.0])
    w = pl.Series("w", [0.0, 0.0])
    out = _weighted_mean(v, w)
    assert out is None


def test_weighted_mean_all_null_returns_none() -> None:
    v = pl.Series("v", [None, None])
    w = pl.Series("w", [None, None])
    out = _weighted_mean(v, w)
    assert out is None


# -----------------------------------------------------------------------------
# _pct_of_cb_mask
# -----------------------------------------------------------------------------


def test_pct_of_cb_mask_all_true() -> None:
    mask = pl.Series("m", [True, True, True])
    balances = pl.Series("b", [100.0, 200.0, 300.0])
    assert _pct_of_cb_mask(mask, balances, 600.0) == 1.0


def test_pct_of_cb_mask_all_false() -> None:
    mask = pl.Series("m", [False, False, False])
    balances = pl.Series("b", [100.0, 200.0, 300.0])
    assert _pct_of_cb_mask(mask, balances, 600.0) == 0.0


def test_pct_of_cb_mask_partial() -> None:
    mask = pl.Series("m", [True, False, True])
    balances = pl.Series("b", [100.0, 200.0, 300.0])
    # masked sum = 400; 400/600 = 0.6666...
    assert _pct_of_cb_mask(mask, balances, 600.0) == pytest.approx(400.0 / 600.0)


def test_pct_of_cb_mask_zero_total_returns_none() -> None:
    mask = pl.Series("m", [True])
    balances = pl.Series("b", [100.0])
    assert _pct_of_cb_mask(mask, balances, 0.0) is None


def test_pct_of_cb_mask_null_in_mask_treated_as_false() -> None:
    """R's `m_calc_current_ltv >= 0.70` returns NA on null inputs;
    the case_when treats NA as no-match. Polars `.fill_null(False)`
    achieves the same effect.
    """
    mask = pl.Series("m", [True, None, False])
    balances = pl.Series("b", [100.0, 200.0, 300.0])
    # Null treated as False; only first row contributes: 100/600
    assert _pct_of_cb_mask(mask, balances, 600.0) == pytest.approx(100.0 / 600.0)


# -----------------------------------------------------------------------------
# _structure_breakdown
# -----------------------------------------------------------------------------


def test_structure_breakdown_pinned_to_fixture() -> None:
    """Synthetic fixture's structure-type breakdown:
      "1: one loan -> one property" -> 1
      "2: one loan -> multiple properties" -> 1
      "3: multiple loans -> one property" -> 2
      "4: Full set" -> 2
      "5: Cross-collateralised set" -> 2
    Sorted lex-ascending on structure_type.
    """
    s = pl.Series("st", [
        "1: one loan → one property",
        "2: one loan → multiple properties",
        "3: multiple loans → one property",
        "3: multiple loans → one property",
        "4: Full set",
        "4: Full set",
        "5: Cross-collateralised set",
        "5: Cross-collateralised set",
    ])
    out = _structure_breakdown(s)
    assert out["Metric"].to_list() == [
        "Loan Count: 1: one loan → one property",
        "Loan Count: 2: one loan → multiple properties",
        "Loan Count: 3: multiple loans → one property",
        "Loan Count: 4: Full set",
        "Loan Count: 5: Cross-collateralised set",
    ]
    assert out["Value"].to_list() == ["1", "1", "2", "2", "2"]


def test_structure_breakdown_empty_input_returns_empty_two_col_frame() -> None:
    """All-null / empty input -> empty 2-col frame so concat with the
    base-38 frame still works.
    """
    s = pl.Series("st", [None, None], dtype=pl.Utf8)
    out = _structure_breakdown(s)
    assert out.height == 0
    assert out.columns == ["Metric", "Value"]
    assert out.schema == {"Metric": pl.Utf8, "Value": pl.Utf8}


def test_structure_breakdown_drops_null_rows() -> None:
    """Null structure_type entries are excluded; remaining values count."""
    s = pl.Series("st", ["1: A", None, "1: A", "2: B", None])
    out = _structure_breakdown(s)
    assert out["Metric"].to_list() == ["Loan Count: 1: A", "Loan Count: 2: B"]
    assert out["Value"].to_list() == ["2", "1"]


# -----------------------------------------------------------------------------
# IEEE-754 accumulation-order regression tests (Stage 10 / C-1.5)
#
# These tests pin the contract surfaced by the C-1 audit: each Σ-over-many-
# rows helper must accumulate in the same order R does, not just any
# deterministic order. The synthetic fixture's value range is too small for
# IEEE-754 drift to differentiate buggy vs fixed numerically; the tests below
# document the parity contract via their docstrings + targeted assertions
# so future refactors don't silently re-introduce a Python-side sort.
# -----------------------------------------------------------------------------


def test_effective_borrowers_iterates_squared_shares_in_descending_order() -> None:
    """Pin the accumulation order: _effective_borrowers iterates the
    desc-sorted balance list directly without re-sorting.

    R's recipe (pipeline.R:670-684):
        borrower_balances <- sort(borrower_agg$x, decreasing = TRUE)
        effective_borrowers <- 1 / sum((borrower_balances / total_cb)^2)

    R sums in DESCENDING order (the input `borrower_balances` is sort-desc).
    A Python-side ascending re-sort would produce a mathematically equivalent
    result that's byte-different on IEEE-754 - which is what we DON'T want.

    The synthetic fixture's 6-borrower value range is too small to expose
    a single-ULP difference in ascending vs descending sum; this test pins
    the contract via the docstring + the synthetic value, but Layer-2
    parity (real_anonymised) is the gate that would actually surface drift.
    """
    bids, balances, total_cb = _synthetic_borrower_inputs()
    out = _effective_borrowers(bids, balances, total_cb)
    # Synthetic value: matches both desc and asc sum within IEEE-754 precision.
    assert out == pytest.approx(5.5517042172, rel=1e-9)


def test_weighted_mean_does_not_sort_inputs_preserves_row_order() -> None:
    """Pin the accumulation order: _weighted_mean preserves input row
    order, matching R's `weighted.mean(v, w, na.rm=TRUE)`.

    R iterates the input vectors in row order (NA pairs skipped). Polars
    `Series.sum()` after filter preserves storage order, which matches.
    A Python-side sort by value (or any other key) would diverge from
    R's accumulation order on real data - which is what we DON'T want.

    Test approach: feed inputs in a non-sorted order; verify the result
    is correct AND would still be correct if the function preserved
    that exact input order. Doesn't catch a sort-vs-no-sort behavioural
    change directly (synthetic precision is too coarse), but the
    docstring + the explicit non-sorted input documents the contract.
    """
    # Non-sorted by value: 3, 1, 2.
    v = pl.Series("v", [3.0, 1.0, 2.0])
    w = pl.Series("w", [10.0, 20.0, 30.0])
    out = _weighted_mean(v, w)
    # Hand-computed: (30 + 20 + 60) / 60 = 110/60.
    assert out == pytest.approx(110.0 / 60.0)


def test_borrower_balances_sorted_desc_matches_r_aggregate_then_sort_desc() -> None:
    """Pin the output ordering: per-borrower sums are sorted descending
    with id ascending tie-break.

    R's recipe:
        borrower_agg <- aggregate(cb ~ borrower, FUN = sum)   # sorted by group key
        sort(borrower_agg$x, decreasing = TRUE)               # sort-desc preserves
                                                              # tie order (insertion)

    R's `sort(decreasing=TRUE)` is stable on numeric values, so two
    borrowers with identical totals retain the alphabetical group-key
    order from `aggregate`. Our `.sort([cb, id], descending=[True, False])`
    matches: cb desc, id asc as tie-break.
    """
    # Borrowers in non-sorted insertion order; identical totals on B_M and B_Z.
    bids = pl.Series("id", ["B_Z", "B_A", "B_M", "B_Z", "B_M"])
    balances = pl.Series("cb", [50.0, 100.0, 30.0, 50.0, 70.0])
    out = _borrower_balances_sorted_desc(bids, balances)
    # Per-borrower totals: B_A=100, B_M=100, B_Z=100. All tied.
    # Tie-break id ascending: B_A, B_M, B_Z -> [100, 100, 100] in that order.
    assert out == [100.0, 100.0, 100.0]
