"""Tests for Stage-9 / B-2 MILAN ranking + cumulative-sum CB fields.

Mirrors r_reference/R/milan_mapping.R:478-555. The B-2 helpers compute
the numeric staging columns; the string-formatted MILAN fields
(`Ranking`, `External Prior Ranks CB`, `Pari Passu Ranking Loans (Not
In Pool)`) land in B-5's transmute.

Coverage:
  - Three dense_rank tie cases (all-distinct, partial-tie, all-tied).
  - Shuffle invariance: input row order does not change ranks.
  - ND in prior_principal_balances collapses to 0 before ranking.
  - Cross-property bleed: cum_sum-over-property does not leak across
    properties.
  - External Prior calc: max(0, ppb - cumsum_below); negative clamps to 0.
  - Pari Passu calc: max(0, ppue - sum_at_rank); ND ppue -> 0 float.
  - Single property, single loan: rank=1, cumsum_below=0, sum_at_rank=cpb.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import polars as pl

from esma_milan.pipeline.milan_map import (
    _attach_external_prior_and_pari_passu,
    _attach_ranking,
)


def _frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """Build a Polars frame with the columns B-2 expects, keyed by row dicts."""
    return pl.DataFrame(
        rows,
        schema={
            "calc_loan_id": pl.Utf8,
            "calc_main_property_id": pl.Utf8,
            "prior_principal_balances": pl.Utf8,
            "current_principal_balance": pl.Utf8,
            "pari_passu_underlying_exposures": pl.Utf8,
        },
    )


def _ranks_by_loan(df: pl.DataFrame) -> dict[str, int]:
    out = _attach_ranking(df)
    return {
        row["calc_loan_id"]: int(row["_milan_ranking"])
        for row in out.iter_rows(named=True)
    }


# ---------------------------------------------------------------------------
# Three tie cases for dense_rank (determinism anchor)
# ---------------------------------------------------------------------------


def test_dense_rank_partial_tie() -> None:
    """(100, 100, 200) -> (1, 1, 2): two loans tied at rank 1, third at 2."""
    df = _frame([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L2", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L3", "calc_main_property_id": "P1",
         "prior_principal_balances": "200", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
    ])
    assert _ranks_by_loan(df) == {"L1": 1, "L2": 1, "L3": 2}


def test_dense_rank_all_distinct() -> None:
    """(10, 20, 30) -> (1, 2, 3): no ties, gap-free dense ranks."""
    df = _frame([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "10", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L2", "calc_main_property_id": "P1",
         "prior_principal_balances": "20", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L3", "calc_main_property_id": "P1",
         "prior_principal_balances": "30", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
    ])
    assert _ranks_by_loan(df) == {"L1": 1, "L2": 2, "L3": 3}


def test_dense_rank_all_tied() -> None:
    """(100, 100, 100) -> (1, 1, 1): all loans tied at rank 1."""
    df = _frame([
        {"calc_loan_id": f"L{i}", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"}
        for i in (1, 2, 3)
    ])
    assert _ranks_by_loan(df) == {"L1": 1, "L2": 1, "L3": 1}


# ---------------------------------------------------------------------------
# Shuffle invariance
# ---------------------------------------------------------------------------


def test_dense_rank_is_stable_under_input_shuffle() -> None:
    """Ranks depend on the values, not the input row order."""
    base_rows = [
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L2", "calc_main_property_id": "P1",
         "prior_principal_balances": "200", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L3", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
    ]
    rng = random.Random(7)
    for _ in range(2):
        shuffled = list(base_rows)
        rng.shuffle(shuffled)
        assert _ranks_by_loan(_frame(shuffled)) == {"L1": 1, "L2": 2, "L3": 1}


# ---------------------------------------------------------------------------
# ND in prior_principal_balances
# ---------------------------------------------------------------------------


def test_nd_prior_principal_balances_collapses_to_zero_before_rank() -> None:
    """ND ppb -> 0 -> rank 1 (lowest, since other ppb values are positive)."""
    df = _frame([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "ND", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L2", "calc_main_property_id": "P1",
         "prior_principal_balances": "50", "current_principal_balance": "0",
         "pari_passu_underlying_exposures": "0"},
    ])
    out = _attach_ranking(df)
    by_loan = {row["calc_loan_id"]: row for row in out.iter_rows(named=True)}
    assert by_loan["L1"]["_ppb_num"] == 0.0
    assert by_loan["L1"]["_milan_ranking"] == 1
    assert by_loan["L2"]["_milan_ranking"] == 2


# ---------------------------------------------------------------------------
# External Prior + Pari Passu calculations
# ---------------------------------------------------------------------------


def _full_pipeline(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    df = _attach_external_prior_and_pari_passu(_attach_ranking(_frame(rows)))
    return {
        row["calc_loan_id"]: {
            "ranking": row["_milan_ranking"],
            "ppb_num": row["_ppb_num"],
            "cpb_num": row["_cpb_num"],
            "ppue_num": row["_ppue_num"],
            "sum_at_rank": row["_cpb_sum_at_rank"],
            "cumsum_below": row["_cpb_cumsum_below"],
            "ext_prior": row["_milan_ext_prior_ranks_cb"],
            "pari_passu": row["_milan_pari_passu_not_in_pool"],
        }
        for row in df.iter_rows(named=True)
    }


def test_external_prior_basic_calc() -> None:
    """Single rank-2 loan: ext_prior = max(0, ppb=300 - cumsum_below=120) = 180."""
    out = _full_pipeline([
        # Rank 1: a single loan with cpb 80.
        {"calc_loan_id": "L_low_a", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "80",
         "pari_passu_underlying_exposures": "0"},
        # Rank 1: a second loan with cpb 40 (same ppb tier, sums to 120 below).
        {"calc_loan_id": "L_low_b", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "40",
         "pari_passu_underlying_exposures": "0"},
        # Rank 2: target row with ppb=300.
        {"calc_loan_id": "L_target", "calc_main_property_id": "P1",
         "prior_principal_balances": "300", "current_principal_balance": "200",
         "pari_passu_underlying_exposures": "0"},
    ])
    target = out["L_target"]
    assert target["ranking"] == 2
    assert target["cumsum_below"] == 120.0   # 80 + 40 from rank 1
    assert target["ext_prior"] == 180.0


def test_external_prior_clamps_negative_to_zero() -> None:
    """ppb < cumsum_below -> ext_prior clamps to 0, never negative."""
    out = _full_pipeline([
        {"calc_loan_id": "L_low", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "500",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "L_target", "calc_main_property_id": "P1",
         "prior_principal_balances": "200", "current_principal_balance": "50",
         "pari_passu_underlying_exposures": "0"},
    ])
    target = out["L_target"]
    assert target["cumsum_below"] == 500.0
    assert target["ext_prior"] == 0.0


def test_pari_passu_basic_calc() -> None:
    """pari_passu = max(0, ppue - sum_at_rank). Two loans at same rank: subtract their CB total."""
    out = _full_pipeline([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "30",
         "pari_passu_underlying_exposures": "100"},
        {"calc_loan_id": "L2", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "20",
         "pari_passu_underlying_exposures": "100"},
    ])
    # Both at rank 1; sum_at_rank = 30 + 20 = 50; pari_passu = max(0, 100 - 50) = 50.
    for loan in ("L1", "L2"):
        assert out[loan]["sum_at_rank"] == 50.0
        assert out[loan]["pari_passu"] == 50.0


def test_pari_passu_nd_input_collapses_to_zero_float() -> None:
    """ND ppue -> 0 float (asymmetric vs External Prior; B-5 transmutes to "0" string)."""
    out = _full_pipeline([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "30",
         "pari_passu_underlying_exposures": "ND"},
    ])
    assert out["L1"]["ppue_num"] == 0.0
    # ppue=0, sum_at_rank=30, so pari_passu = max(0, 0 - 30) = 0.
    assert out["L1"]["pari_passu"] == 0.0


def test_pari_passu_clamps_negative_to_zero() -> None:
    """ppue < sum_at_rank -> clamp to 0."""
    out = _full_pipeline([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "200",
         "pari_passu_underlying_exposures": "50"},
    ])
    # sum_at_rank=200, ppue=50 -> max(0, 50-200) = 0.
    assert out["L1"]["pari_passu"] == 0.0


# ---------------------------------------------------------------------------
# Single-property single-loan edge case
# ---------------------------------------------------------------------------


def test_single_property_single_loan() -> None:
    """rank=1, cumsum_below=0, sum_at_rank=cpb, ext_prior=ppb, pari_passu=max(0, ppue-cpb)."""
    out = _full_pipeline([
        {"calc_loan_id": "L1", "calc_main_property_id": "P1",
         "prior_principal_balances": "100", "current_principal_balance": "60",
         "pari_passu_underlying_exposures": "200"},
    ])
    row = out["L1"]
    assert row["ranking"] == 1
    assert row["cumsum_below"] == 0.0
    assert row["sum_at_rank"] == 60.0
    assert row["ext_prior"] == 100.0      # ppb - 0 = 100
    assert row["pari_passu"] == 140.0     # max(0, 200 - 60) = 140


# ---------------------------------------------------------------------------
# Cross-property bleed (the §Risks anchor)
# ---------------------------------------------------------------------------


def test_cumsum_below_does_not_bleed_across_properties() -> None:
    """Property A's cumulative sum must not leak into property B's rank-1.

    Two properties, two ranks each. If `cum_sum().over("calc_main_property_id")`
    were misplaced (e.g. global cumsum, or partition-by missing), property B's
    rank-1 row would inherit A's totals, inflating cumsum_below.

    Expected: each property's rank-1 row sees cumsum_below=0; rank-2 sees the
    OWN-property rank-1 sum.
    """
    out = _full_pipeline([
        # Property A: rank-1 cb=10, rank-2 ppb=200.
        {"calc_loan_id": "A_low", "calc_main_property_id": "PA",
         "prior_principal_balances": "100", "current_principal_balance": "10",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "A_high", "calc_main_property_id": "PA",
         "prior_principal_balances": "200", "current_principal_balance": "5",
         "pari_passu_underlying_exposures": "0"},
        # Property B: rank-1 cb=300, rank-2 ppb=999.
        {"calc_loan_id": "B_low", "calc_main_property_id": "PB",
         "prior_principal_balances": "100", "current_principal_balance": "300",
         "pari_passu_underlying_exposures": "0"},
        {"calc_loan_id": "B_high", "calc_main_property_id": "PB",
         "prior_principal_balances": "999", "current_principal_balance": "7",
         "pari_passu_underlying_exposures": "0"},
    ])
    # Property A: A_low sees 0 below (rank 1); A_high sees 10 below (only PA's rank 1).
    assert out["A_low"]["cumsum_below"] == 0.0
    assert out["A_high"]["cumsum_below"] == 10.0
    # Property B: B_low sees 0 below (rank 1, NOT PA's 10 + 5); B_high sees 300.
    assert out["B_low"]["cumsum_below"] == 0.0
    assert out["B_high"]["cumsum_below"] == 300.0
    # Confirm no cross-contamination of sum_at_rank either.
    assert out["A_low"]["sum_at_rank"] == 10.0
    assert out["B_low"]["sum_at_rank"] == 300.0
