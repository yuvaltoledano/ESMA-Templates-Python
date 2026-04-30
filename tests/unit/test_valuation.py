"""Tests for esma_milan.pipeline.valuation.

Mirrors r_reference/tests/testthat/test-detect-aggregation.R one-for-one
(test names track the R `test_that()` strings) plus boundary tests for
the strict 0.9 threshold and unit tests for `select_valuation_amount`
covering the Stage-1/Stage-2 selection rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from esma_milan.pipeline.valuation import (
    Stage6Output,
    detect_aggregation_method,
    run_stage6,
    select_valuation_amount,
)


def _loans(
    group_ids: Sequence[int],
    loan_ids: Sequence[str],
    balances: Sequence[float | int | None],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "collateral_group_id": pl.Series(list(group_ids), dtype=pl.Int64),
            "calc_loan_id": pl.Series(list(loan_ids), dtype=pl.String),
            "current_principal_balance": pl.Series(
                list(balances), dtype=pl.Float64
            ),
        }
    )


def _props(
    group_ids: Sequence[int],
    loan_refs: Sequence[str],
    property_ids: Sequence[str],
    current_methods: Sequence[str],
    current_amounts: Sequence[float | int | None],
    original_amounts: Sequence[float | int | None],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "collateral_group_id": pl.Series(list(group_ids), dtype=pl.Int64),
            "underlying_exposure_identifier": pl.Series(
                list(loan_refs), dtype=pl.String
            ),
            "calc_property_id": pl.Series(list(property_ids), dtype=pl.String),
            "current_valuation_method": pl.Series(
                list(current_methods), dtype=pl.String
            ),
            "current_valuation_amount": pl.Series(
                list(current_amounts), dtype=pl.Float64
            ),
            "original_valuation_amount": pl.Series(
                list(original_amounts), dtype=pl.Float64
            ),
        }
    )


# ---------------------------------------------------------------------------
# select_valuation_amount: Stage-1 / Stage-2 selection rule
# ---------------------------------------------------------------------------


def test_select_valuation_prefers_current_when_method_is_full_inspection() -> None:
    """current_valuation_method ∈ {FIEI, FOEI} -> use current amount."""
    props = _props(
        [1, 1],
        ["L1", "L2"],
        ["P1", "P2"],
        ["FIEI", "FOEI"],
        [200.0, 300.0],
        [100.0, 100.0],
    )
    out = select_valuation_amount(props)
    assert out["valuation_amount"].to_list() == [200.0, 300.0]


def test_select_valuation_falls_back_to_original_for_non_inspection_methods() -> None:
    """current_valuation_method NOT in {FIEI, FOEI} -> use original amount."""
    props = _props(
        [1], ["L1"], ["P1"], ["DRVB"], [200.0], [150.0]
    )
    out = select_valuation_amount(props)
    assert out["valuation_amount"].to_list() == [150.0]


def test_select_valuation_stage2_flips_when_chosen_amount_is_null() -> None:
    """Stage 2: chosen current is null -> flip to original."""
    props = _props([1], ["L1"], ["P1"], ["FIEI"], [None], [180.0])
    assert select_valuation_amount(props)["valuation_amount"].to_list() == [180.0]


def test_select_valuation_stage2_flips_when_chosen_amount_below_min() -> None:
    """Stage 2: chosen current <= MIN_VALID_PROPERTY_VALUE (10) -> flip."""
    props = _props([1, 1], ["L1", "L2"], ["P1", "P2"], ["FIEI", "FIEI"], [5.0, 10.0], [180.0, 200.0])
    out = select_valuation_amount(props)
    # 5 <= 10 -> flip; 10 <= 10 -> flip too (boundary inclusive).
    assert out["valuation_amount"].to_list() == [180.0, 200.0]


def test_select_valuation_stage2_no_flip_when_chosen_amount_just_above_min() -> None:
    """Boundary: 11 > 10 -> no flip."""
    props = _props([1], ["L1"], ["P1"], ["FIEI"], [11.0], [180.0])
    assert select_valuation_amount(props)["valuation_amount"].to_list() == [11.0]


def test_select_valuation_drops_internal_helper_columns() -> None:
    """The helper must not leak its `_initial_*` working columns."""
    props = _props([1], ["L1"], ["P1"], ["FIEI"], [200.0], [150.0])
    out = select_valuation_amount(props)
    assert not any(c.startswith("_") for c in out.columns)
    assert "valuation_amount" in out.columns


# ---------------------------------------------------------------------------
# detect_aggregation_method - mirrors test-detect-aggregation.R one-for-one
# ---------------------------------------------------------------------------


def test_returns_ambiguous_when_no_multi_loan_groups_exist() -> None:
    """Mirrors 'returns ambiguous when no multi-loan groups exist'."""
    loans = _loans([1, 2], ["L1", "L2"], [100000.0, 200000.0])
    properties = _props(
        [1, 2],
        ["L1", "L2"],
        ["P1", "P2"],
        ["FIEI", "FIEI"],
        [150000.0, 250000.0],
        [140000.0, 240000.0],
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "ambiguous"
    assert "No multi-loan groups" in out.message


def test_detects_by_loan_when_values_are_duplicated() -> None:
    """Mirrors 'detects by_loan when values are duplicated'."""
    loans = _loans([1, 1], ["L1", "L2"], [100000.0, 80000.0])
    properties = _props(
        [1, 1],
        ["L1", "L2"],
        ["P1", "P1"],
        ["FIEI", "FIEI"],
        [200000.0, 200000.0],  # same -> by_loan
        [180000.0, 180000.0],
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "by_loan"


def test_detects_by_group_when_values_differ() -> None:
    """Mirrors 'detects by_group when values differ'."""
    loans = _loans([1, 1], ["L1", "L2"], [100000.0, 80000.0])
    properties = _props(
        [1, 1],
        ["L1", "L2"],
        ["P1", "P1"],
        ["FIEI", "FIEI"],
        [120000.0, 80000.0],  # different -> by_group
        [100000.0, 80000.0],
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "by_group"


def test_returns_ambiguous_when_join_produces_no_matches() -> None:
    """Mirrors 'returns ambiguous when join produces no matches'."""
    loans = _loans([1, 1], ["L1", "L2"], [100000.0, 80000.0])
    properties = _props(
        [1, 1],
        ["L99", "L98"],  # disjoint loan refs -> empty inner-join
        ["P1", "P1"],
        ["FIEI", "FIEI"],
        [200000.0, 200000.0],
        [180000.0, 180000.0],
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "ambiguous"


def test_ignores_a_single_na_when_classifying_by_loan() -> None:
    """Mirrors 'ignores a single NA when classifying by_loan'.

    Three loans share one property; valuations are (500000, 500000, NA).
    The single NA must NOT flip classification - n_distinct ignores NA,
    so distinct count = 1 (just 500000) -> by_loan.
    """
    loans = _loans([1, 1, 1], ["L1", "L2", "L3"], [100000.0, 80000.0, 60000.0])
    properties = _props(
        [1, 1, 1],
        ["L1", "L2", "L3"],
        ["P1", "P1", "P1"],
        ["FIEI", "FIEI", "FIEI"],
        [500000.0, 500000.0, None],
        [480000.0, 480000.0, None],
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "by_loan"


def test_treats_single_observation_properties_as_no_evidence() -> None:
    """Mirrors 'treats single-observation properties as no evidence'.

    Valuations (500000, NA, NA) - only one observed value, dropped at
    `non_na_observations >= 2` filter. Total evidence = 0 -> ambiguous.
    """
    loans = _loans([1, 1, 1], ["L1", "L2", "L3"], [100000.0, 80000.0, 60000.0])
    properties = _props(
        [1, 1, 1],
        ["L1", "L2", "L3"],
        ["P1", "P1", "P1"],
        ["FIEI", "FIEI", "FIEI"],
        [500000.0, None, None],
        [500000.0, None, None],
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "ambiguous"


def test_tolerates_a_single_na_at_the_90_percent_threshold() -> None:
    """Mirrors 'tolerates a single NA at the 90% threshold'.

    9 by_loan properties (each with 2 valid duplicated valuations) plus
    1 property with (500k, NA) - the latter has only 1 non-NA observation
    so it's dropped. Evidence = 9 by_loan, 0 by_group. 9/9 > 0.9 -> by_loan.
    """
    group_ids = [g for g in range(1, 11) for _ in (0, 1)]  # 1,1,2,2,...,10,10
    loan_ids = [f"L{i}" for i in range(1, 21)]
    property_ids = [f"P{g}" for g in group_ids]
    # First 9 properties: duplicated valid valuations. Last property: (500k, NA).
    current_amounts: list[float | None] = []
    for g in range(1, 11):
        if g <= 9:
            current_amounts += [500000.0, 500000.0]
        else:
            current_amounts += [500000.0, None]

    loans = _loans(
        group_ids,
        loan_ids,
        [100000.0 if i % 2 == 0 else 80000.0 for i in range(20)],
    )
    properties = _props(
        group_ids,
        loan_ids,
        property_ids,
        ["FIEI"] * 20,
        current_amounts,
        current_amounts,  # mirror so Stage-2 fallback doesn't change behaviour
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "by_loan"


def test_uses_original_valuation_when_current_method_is_not_fiei_foei() -> None:
    """Mirrors 'uses original valuation when current method is not FIEI/FOEI'.

    current_valuation_method = DRVB (not full inspection) -> Stage 1 picks
    original. originals are (120000, 80000) which differ -> by_group.
    """
    loans = _loans([1, 1], ["L1", "L2"], [100000.0, 80000.0])
    properties = _props(
        [1, 1],
        ["L1", "L2"],
        ["P1", "P1"],
        ["DRVB", "DRVB"],
        [200000.0, 200000.0],  # currents identical but DRVB -> not used
        [120000.0, 80000.0],   # originals differ
    )
    out = detect_aggregation_method(loans, properties)
    assert out.detected_method == "by_group"


# ---------------------------------------------------------------------------
# Threshold boundary checks (not directly mirrored from R but pinning the
# strict >0.9 contract that the R code relies on)
# ---------------------------------------------------------------------------


def test_threshold_strictly_greater_than_90pct_at_9_of_10() -> None:
    """9/10 = 0.9 exactly; NOT > 0.9 -> ambiguous. Pinning the strict
    inequality (R's `n_by_loan / total > 0.9`)."""
    # Build 10 (group, property) buckets: 9 by_loan-shaped, 1 by_group-shaped.
    group_ids = [g for g in range(1, 11) for _ in (0, 1)]  # 20 rows total
    loan_ids = [f"L{i}" for i in range(1, 21)]
    property_ids = [f"P{g}" for g in group_ids]

    current_amounts: list[float | None] = []
    for g in range(1, 11):
        if g <= 9:
            current_amounts += [500000.0, 500000.0]  # by_loan: identical
        else:
            current_amounts += [500000.0, 400000.0]  # by_group: differ

    loans = _loans(
        group_ids,
        loan_ids,
        [100000.0 if i % 2 == 0 else 80000.0 for i in range(20)],
    )
    properties = _props(
        group_ids,
        loan_ids,
        property_ids,
        ["FIEI"] * 20,
        current_amounts,
        current_amounts,
    )
    out = detect_aggregation_method(loans, properties)
    # 9 by_loan + 1 by_group = 10 total. 9/10 = 0.9, NOT > 0.9. Ambiguous.
    assert out.detected_method == "ambiguous"


# ---------------------------------------------------------------------------
# run_stage6 driver
# ---------------------------------------------------------------------------


def test_run_stage6_wraps_detect_aggregation_method() -> None:
    """run_stage6 is a thin wrapper - confirm same output as direct call."""
    loans = _loans([1, 1], ["L1", "L2"], [100.0, 200.0])
    properties = _props(
        [1, 1],
        ["L1", "L2"],
        ["P1", "P1"],
        ["FIEI", "FIEI"],
        [500.0, 500.0],
        [500.0, 500.0],
    )
    out = run_stage6(loans, properties)
    assert isinstance(out, Stage6Output)
    assert out.detected_method == "by_loan"


# Unused-import guard: keep `date` imported to make it cheap to add date-
# bearing fixtures in future without a re-import churn.
_ = date
