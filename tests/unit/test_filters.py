"""Tests for Stage 2 filters.

Two layers:

  1. Inline unit tests (this file) build small Polars DataFrames in-memory
     and exercise every branch of the loan filter and property filter,
     including drop cases that the staged synthetic fixture does NOT
     exercise.

  2. The synthetic-fixture happy-path lives in
     tests/integration/test_stage2.py - it asserts that all 8 loans and
     all 12 properties survive on the staged fixture (no drops), proving
     Stage 2 is a no-op there.

The 9-loan inline test below is the one the user explicitly requested:
exactly 8 of the 9 input loans must survive Stage 2, with a comment
identifying the dropped loan and the reason.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from esma_milan.pipeline.filters import (
    Stage2Output,
    collect_active_loan_ids,
    filter_active_loans,
    filter_residential_properties,
    run_stage2,
)

# ---------------------------------------------------------------------------
# filter_active_loans
# ---------------------------------------------------------------------------


def test_nine_loans_one_dropped_for_dflt_status() -> None:
    """The user-requested branch coverage for Stage 2.

    Construct a 9-loan inline DataFrame where exactly one loan is
    inactive: LOAN_009 has account_status = "DFLT" (defaulted, not in
    ACTIVE_LOAN_STATUSES = {PERF, ARRE, RARR, RNAR}). The filter must
    drop that loan, leaving 8 survivors.

    The other 8 loans cover all four active statuses to guarantee none
    of them are spuriously dropped.
    """
    df = pl.DataFrame(
        {
            "calc_loan_id": [f"LOAN_{i:03d}" for i in range(1, 10)],
            "account_status": [
                "PERF", "PERF", "PERF", "ARRE", "RARR", "RNAR", "PERF", "PERF",
                # LOAN_009 dropped: DFLT is not in ACTIVE_LOAN_STATUSES.
                "DFLT",
            ],
            "current_principal_balance": [100000] * 9,
        }
    )
    out = filter_active_loans(df)
    assert out.height == 8
    survivors = sorted(out["calc_loan_id"].to_list())
    assert survivors == [f"LOAN_{i:03d}" for i in range(1, 9)]
    assert "LOAN_009" not in survivors


def test_zero_balance_loan_dropped() -> None:
    """current_principal_balance > 0 condition: a loan with CPB == 0
    must be dropped even if its account_status is active."""
    df = pl.DataFrame(
        {
            "calc_loan_id": ["L1", "L2"],
            "account_status": ["PERF", "PERF"],
            "current_principal_balance": [100000, 0],
        }
    )
    out = filter_active_loans(df)
    assert out["calc_loan_id"].to_list() == ["L1"]


def test_negative_balance_loan_dropped() -> None:
    df = pl.DataFrame(
        {
            "calc_loan_id": ["L1", "L2"],
            "account_status": ["PERF", "PERF"],
            "current_principal_balance": [100, -50],
        }
    )
    out = filter_active_loans(df)
    assert out["calc_loan_id"].to_list() == ["L1"]


@pytest.mark.parametrize("status", ["PERF", "ARRE", "RARR", "RNAR"])
def test_all_four_active_statuses_kept(status: str) -> None:
    """Pin the canonical four-element ACTIVE_LOAN_STATUSES."""
    df = pl.DataFrame(
        {
            "calc_loan_id": ["L1"],
            "account_status": [status],
            "current_principal_balance": [100],
        }
    )
    assert filter_active_loans(df).height == 1


@pytest.mark.parametrize("status", ["DFLT", "DFLT_X", "RDMD", "OTHR", "REBR", "REDF", "RERE"])
def test_inactive_statuses_dropped(status: str) -> None:
    df = pl.DataFrame(
        {
            "calc_loan_id": ["L1"],
            "account_status": [status],
            "current_principal_balance": [100],
        }
    )
    assert filter_active_loans(df).height == 0


# ---------------------------------------------------------------------------
# collect_active_loan_ids
# ---------------------------------------------------------------------------


def test_collect_active_loan_ids_unions_both_id_columns() -> None:
    df = pl.DataFrame(
        {
            "original_underlying_exposure_identifier": ["O1", "O2", "O3"],
            "new_underlying_exposure_identifier": ["N1", "N2", "N3"],
        }
    )
    assert collect_active_loan_ids(df) == {"O1", "O2", "O3", "N1", "N2", "N3"}


def test_collect_active_loan_ids_drops_nulls() -> None:
    df = pl.DataFrame(
        {
            "original_underlying_exposure_identifier": ["O1", None, "O3"],
            "new_underlying_exposure_identifier": [None, "N2", "N3"],
        }
    )
    assert collect_active_loan_ids(df) == {"O1", "O3", "N2", "N3"}


def test_collect_active_loan_ids_handles_missing_column() -> None:
    """If only one of the two ID columns is present, take what's there."""
    df = pl.DataFrame({"original_underlying_exposure_identifier": ["O1", "O2"]})
    assert collect_active_loan_ids(df) == {"O1", "O2"}


# ---------------------------------------------------------------------------
# filter_residential_properties
# ---------------------------------------------------------------------------


def _properties_skeleton(**overrides: object) -> pl.DataFrame:
    """Build a single-row properties DataFrame with sensible defaults
    for every column the filter inspects. Any overrides win."""
    base: dict[str, list[object]] = {
        "underlying_exposure_identifier": ["L1"],
        "calc_property_id": ["P1"],
        "property_type": ["RHOS"],
        "collateral_type": ["RES"],
        "original_valuation_amount": [100000.0],
        "current_valuation_amount": [120000.0],
    }
    for k, v in overrides.items():
        base[k] = v if isinstance(v, list) else [v]
    return pl.DataFrame(base)


def test_property_with_null_property_type_dropped() -> None:
    df = _properties_skeleton(property_type=[None])
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


@pytest.mark.parametrize(
    "code",
    ["GUAR", "CARX", "MCHT", "SECU", "OTGI", "INDV", "AERO", "MDEQ"],
)
def test_excluded_collateral_types_dropped(code: str) -> None:
    """A handful of EXCLUDED_COLLATERAL_TYPES codes must drop the row."""
    df = _properties_skeleton(collateral_type=[code])
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0, f"collateral_type={code!r} should be dropped"


def test_property_referencing_unknown_loan_dropped() -> None:
    df = _properties_skeleton(underlying_exposure_identifier=["L_DOES_NOT_EXIST"])
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_both_valuations_zero_dropped() -> None:
    df = _properties_skeleton(
        original_valuation_amount=[0.0],
        current_valuation_amount=[0.0],
    )
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_both_valuations_negative_dropped() -> None:
    df = _properties_skeleton(
        original_valuation_amount=[-1.0],
        current_valuation_amount=[-100.0],
    )
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_both_valuations_null_dropped() -> None:
    """coalesce(NA, 0) = 0 -> both <= 0 -> dropped. Mirrors R's
    `coalesce(original_valuation_amount, 0)` treatment."""
    df = _properties_skeleton(
        original_valuation_amount=[None],
        current_valuation_amount=[None],
    )
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_one_positive_valuation_kept() -> None:
    """If at least one of the two valuations is positive, the row stays.
    Two cases: (positive, null) and (null, positive)."""
    df1 = _properties_skeleton(
        original_valuation_amount=[150000.0],
        current_valuation_amount=[None],
    )
    assert filter_residential_properties(df1, active_loan_ids={"L1"}).height == 1

    df2 = _properties_skeleton(
        original_valuation_amount=[None],
        current_valuation_amount=[200000.0],
    )
    assert filter_residential_properties(df2, active_loan_ids={"L1"}).height == 1


def test_property_with_one_positive_one_zero_kept() -> None:
    """Edge: one valuation == 0, the other > 0 -> kept (only DROPS when
    BOTH are <= 0)."""
    df = _properties_skeleton(
        original_valuation_amount=[0.0],
        current_valuation_amount=[200000.0],
    )
    assert filter_residential_properties(df, active_loan_ids={"L1"}).height == 1


def test_residential_property_with_active_loan_kept() -> None:
    df = _properties_skeleton()
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 1


# ---------------------------------------------------------------------------
# run_stage2 driver
# ---------------------------------------------------------------------------


def test_run_stage2_hard_errors_when_no_properties_remain() -> None:
    """Mirrors `stop("ERROR: No properties remaining after filtering.")`
    in r_reference/R/pipeline.R:257."""
    loans = pl.DataFrame(
        {
            "calc_loan_id": ["L1"],
            "original_underlying_exposure_identifier": ["L1"],
            "new_underlying_exposure_identifier": ["L1"],
            "account_status": ["PERF"],
            "current_principal_balance": [100000.0],
        }
    )
    # Property references an unknown loan so the filter empties the
    # table; Stage 2 must hard-error.
    properties = _properties_skeleton(
        underlying_exposure_identifier=["L_NOT_PRESENT"]
    )
    with pytest.raises(ValueError, match="No properties remaining"):
        run_stage2(loans, properties)


def test_run_stage2_returns_filtered_pair() -> None:
    """Happy path: one active loan, one matching property, both survive."""
    loans = pl.DataFrame(
        {
            "calc_loan_id": ["L1", "L2"],
            "original_underlying_exposure_identifier": ["L1", "L2"],
            "new_underlying_exposure_identifier": ["L1", "L2"],
            "account_status": ["PERF", "DFLT"],  # L2 dropped
            "current_principal_balance": [100000, 50000],
            "origination_date": [date(2020, 1, 1), date(2020, 1, 1)],
        }
    )
    properties = pl.DataFrame(
        {
            "underlying_exposure_identifier": ["L1", "L2"],  # L2's prop dropped
            "calc_property_id": ["P1", "P2"],
            "property_type": ["RHOS", "RHOS"],
            "collateral_type": ["RES", "RES"],
            "original_valuation_amount": [100000.0, 100000.0],
            "current_valuation_amount": [120000.0, 120000.0],
        }
    )
    out = run_stage2(loans, properties)
    assert isinstance(out, Stage2Output)
    assert out.loans["calc_loan_id"].to_list() == ["L1"]
    assert out.properties["underlying_exposure_identifier"].to_list() == ["L1"]


def test_run_stage2_keeps_property_under_either_id_convention() -> None:
    """Stage 2 unions both ID columns when computing active_loan_ids,
    so a property referencing either side survives."""
    loans = pl.DataFrame(
        {
            "calc_loan_id": ["L1", "L2"],
            "original_underlying_exposure_identifier": ["O1", "O2"],
            "new_underlying_exposure_identifier": ["N1", "N2"],
            "account_status": ["PERF", "PERF"],
            "current_principal_balance": [100000, 100000],
        }
    )
    properties = pl.DataFrame(
        {
            "underlying_exposure_identifier": ["O1", "N2"],  # one each
            "calc_property_id": ["P1", "P2"],
            "property_type": ["RHOS", "RHOS"],
            "collateral_type": ["RES", "RES"],
            "original_valuation_amount": [100000.0, 100000.0],
            "current_valuation_amount": [120000.0, 120000.0],
        }
    )
    out = run_stage2(loans, properties)
    assert out.properties.height == 2
