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

The 9-loan inline test below pins the drop path: exactly 8 of the 9
input loans must survive Stage 2, with a comment identifying the dropped
loan and the reason.
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
    """Branch coverage for the Stage 2 drop path.

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


def _property_row(**overrides: object) -> pl.DataFrame:
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
    df = _property_row(property_type=[None])
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


@pytest.mark.parametrize(
    "code",
    ["GUAR", "CARX", "MCHT", "SECU", "OTGI", "INDV", "AERO", "MDEQ"],
)
def test_excluded_collateral_types_dropped(code: str) -> None:
    """A handful of EXCLUDED_COLLATERAL_TYPES codes must drop the row."""
    df = _property_row(collateral_type=[code])
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0, f"collateral_type={code!r} should be dropped"


def test_property_referencing_unknown_loan_dropped() -> None:
    df = _property_row(underlying_exposure_identifier=["L_DOES_NOT_EXIST"])
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_both_valuations_zero_dropped() -> None:
    df = _property_row(
        original_valuation_amount=[0.0],
        current_valuation_amount=[0.0],
    )
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_both_valuations_negative_dropped() -> None:
    df = _property_row(
        original_valuation_amount=[-1.0],
        current_valuation_amount=[-100.0],
    )
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_both_valuations_null_dropped() -> None:
    """coalesce(NA, 0) = 0 -> both <= 0 -> dropped. Mirrors R's
    `coalesce(original_valuation_amount, 0)` treatment."""
    df = _property_row(
        original_valuation_amount=[None],
        current_valuation_amount=[None],
    )
    out = filter_residential_properties(df, active_loan_ids={"L1"})
    assert out.height == 0


def test_property_with_one_positive_valuation_kept() -> None:
    """If at least one of the two valuations is positive, the row stays.
    Two cases: (positive, null) and (null, positive)."""
    df1 = _property_row(
        original_valuation_amount=[150000.0],
        current_valuation_amount=[None],
    )
    assert filter_residential_properties(df1, active_loan_ids={"L1"}).height == 1

    df2 = _property_row(
        original_valuation_amount=[None],
        current_valuation_amount=[200000.0],
    )
    assert filter_residential_properties(df2, active_loan_ids={"L1"}).height == 1


def test_property_with_one_positive_one_zero_kept() -> None:
    """Edge: one valuation == 0, the other > 0 -> kept (only DROPS when
    BOTH are <= 0)."""
    df = _property_row(
        original_valuation_amount=[0.0],
        current_valuation_amount=[200000.0],
    )
    assert filter_residential_properties(df, active_loan_ids={"L1"}).height == 1


def test_residential_property_with_active_loan_kept() -> None:
    df = _property_row()
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
    properties = _property_row(
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


# ---------------------------------------------------------------------------
# The drop reasons as named expressions
# ---------------------------------------------------------------------------

from esma_milan.pipeline.filters import (  # noqa: E402
    collateral_drop_reasons,
    loan_drop_reasons,
)

_NAN = float("nan")

# (account_status, current_principal_balance) -> the reasons that are true.
# NaN follows Polars: NaN > 0 is true, so a NaN balance is "positive".
_LOAN_TRUTH: list[tuple[str | None, float | None, set[str]]] = [
    ("PERF", 100.0, set()),
    ("PERF", 0.0, {"balance_not_positive"}),
    ("PERF", -1.0, {"balance_not_positive"}),
    ("PERF", None, {"balance_not_positive"}),
    ("PERF", _NAN, set()),
    ("DFLT", 100.0, {"status_not_active"}),
    ("DFLT", 0.0, {"status_not_active", "balance_not_positive"}),
    ("DFLT", -1.0, {"status_not_active", "balance_not_positive"}),
    ("DFLT", None, {"status_not_active", "balance_not_positive"}),
    ("DFLT", _NAN, {"status_not_active"}),
    (None, 100.0, {"status_not_active"}),
    (None, 0.0, {"status_not_active", "balance_not_positive"}),
    (None, -1.0, {"status_not_active", "balance_not_positive"}),
    (None, None, {"status_not_active", "balance_not_positive"}),
    (None, _NAN, {"status_not_active"}),
]


def _true_reasons(frame: pl.DataFrame, reasons: dict[str, pl.Expr]) -> list[set[str]]:
    evaluated = frame.select(**reasons)
    assert sum(evaluated.null_count().row(0)) == 0, "a reason expression yielded null"
    assert evaluated.dtypes == [pl.Boolean] * len(reasons)
    return [{name for name, value in row.items() if value} for row in evaluated.to_dicts()]


def test_loan_reasons_truth_table_row_by_row() -> None:
    frame = pl.DataFrame(
        {
            "account_status": pl.Series([r[0] for r in _LOAN_TRUTH], dtype=pl.String),
            "current_principal_balance": pl.Series(
                [r[1] for r in _LOAN_TRUTH], dtype=pl.Float64
            ),
        }
    ).with_row_index("i")
    reasons = loan_drop_reasons()
    assert list(reasons) == ["status_not_active", "balance_not_positive"]
    assert _true_reasons(frame, reasons) == [r[2] for r in _LOAN_TRUTH]
    # Kept exactly when no reason is true, and that is what the filter keeps.
    expected_kept = [i for i, r in enumerate(_LOAN_TRUTH) if not r[2]]
    assert expected_kept == [0, 4]
    assert filter_active_loans(frame)["i"].to_list() == expected_kept


_VALUATION_VALUES: tuple[float | None, ...] = (100.0, 0.0, -5.0, None, _NAN)


def _not_positive(value: float | None) -> bool:
    """Missing, zero or negative. NaN is not: NaN <= 0 is false."""
    return value is None or value <= 0


def _collateral_truth() -> list[tuple[dict[str, object], set[str]]]:
    """Every combination of the five operands with the reasons expected,
    derived here from the rule, not from the expressions under test."""
    table: list[tuple[dict[str, object], set[str]]] = []
    for property_type in ("RHOS", None):
        for collateral_type in ("RBLD", "GUAR", None):
            for loan_id in ("L1", "ZZ", None):
                for original in _VALUATION_VALUES:
                    for current in _VALUATION_VALUES:
                        expected: set[str] = set()
                        if property_type is None:
                            expected.add("property_type_missing")
                        if collateral_type is None:
                            expected.add("collateral_type_missing")
                        if collateral_type == "GUAR":
                            expected.add("collateral_type_excluded")
                        if loan_id != "L1":
                            expected.add("no_active_loan")
                        if _not_positive(original) and _not_positive(current):
                            expected.add("no_positive_valuation")
                        table.append(
                            (
                                {
                                    "property_type": property_type,
                                    "collateral_type": collateral_type,
                                    "underlying_exposure_identifier": loan_id,
                                    "original_valuation_amount": original,
                                    "current_valuation_amount": current,
                                },
                                expected,
                            )
                        )
    return table


def _collateral_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "property_type": pl.String,
            "collateral_type": pl.String,
            "underlying_exposure_identifier": pl.String,
            "original_valuation_amount": pl.Float64,
            "current_valuation_amount": pl.Float64,
        },
    ).with_row_index("i")


def test_collateral_reasons_truth_table_row_by_row() -> None:
    table = _collateral_truth()
    assert len(table) == 2 * 3 * 3 * 5 * 5
    frame = _collateral_frame([row for row, _ in table])
    reasons = collateral_drop_reasons({"L1"})
    assert list(reasons) == [
        "property_type_missing",
        "collateral_type_missing",
        "collateral_type_excluded",
        "no_active_loan",
        "no_positive_valuation",
    ]
    assert _true_reasons(frame, reasons) == [expected for _, expected in table]
    expected_kept = [i for i, (_, expected) in enumerate(table) if not expected]
    assert len(expected_kept) == 16  # 25 valuation pairs minus the 9 with no positive amount
    assert (
        filter_residential_properties(frame, {"L1"})["i"].to_list() == expected_kept
    )


def test_collateral_reasons_with_no_active_loan_drop_every_row() -> None:
    table = _collateral_truth()
    frame = _collateral_frame([row for row, _ in table])
    true_reasons = _true_reasons(frame, collateral_drop_reasons(set()))
    assert all("no_active_loan" in reasons for reasons in true_reasons)
    assert filter_residential_properties(frame, set()).height == 0


def test_valuation_pair_table_including_nan() -> None:
    """The 5 x 5 valuation table of the research note, spelled out:
    dropped only when both amounts are missing, zero or negative."""
    dropped = {
        (o, c)
        for o in (0.0, -5.0, None)
        for c in (0.0, -5.0, None)
    }
    rows: list[dict[str, object]] = [
        {
            "property_type": "RHOS",
            "collateral_type": "RBLD",
            "underlying_exposure_identifier": "L1",
            "original_valuation_amount": o,
            "current_valuation_amount": c,
        }
        for o in _VALUATION_VALUES
        for c in _VALUATION_VALUES
    ]
    kept = filter_residential_properties(_collateral_frame(rows), {"L1"})["i"].to_list()
    expected = [
        i
        for i, row in enumerate(rows)
        if (row["original_valuation_amount"], row["current_valuation_amount"])
        not in dropped
    ]
    assert kept == expected and len(kept) == 16


def test_a_row_failing_two_conditions_reports_both() -> None:
    loans = pl.DataFrame(
        {
            "original_underlying_exposure_identifier": ["L1", "L2"],
            "new_underlying_exposure_identifier": ["L1", "L2"],
            "account_status": ["PERF", "DFLT"],
            "current_principal_balance": [100.0, 0.0],
        }
    )
    properties = pl.DataFrame(
        {
            "underlying_exposure_identifier": ["L1", "L2", "L1"],
            "new_collateral_identifier": ["P1", "P2", "P3"],
            "property_type": ["RHOS", None, "RHOS"],
            "collateral_type": ["RBLD", "GUAR", None],
            "original_valuation_amount": [1.0, 0.0, 1.0],
            "current_valuation_amount": [1.0, None, 1.0],
        }
    )
    out = run_stage2(loans, properties)
    assert out.loan_drop_reasons is not None and out.property_drop_reasons is not None
    assert out.loan_drop_reasons.to_dicts() == [
        {"status_not_active": False, "balance_not_positive": False},
        {"status_not_active": True, "balance_not_positive": True},
    ]
    assert out.property_drop_reasons.to_dicts() == [
        {
            "property_type_missing": False,
            "collateral_type_missing": False,
            "collateral_type_excluded": False,
            "no_active_loan": False,
            "no_positive_valuation": False,
        },
        {
            "property_type_missing": True,
            "collateral_type_missing": False,
            "collateral_type_excluded": True,
            "no_active_loan": True,  # L2 was dropped as a loan
            "no_positive_valuation": True,
        },
        {
            "property_type_missing": False,
            "collateral_type_missing": True,
            "collateral_type_excluded": False,
            "no_active_loan": False,
            "no_positive_valuation": False,
        },
    ]
    assert out.loan_rows is not None and out.property_rows is not None
    assert out.loan_rows.to_list() == [0]
    assert out.property_rows.to_list() == [0]
    assert out.loans.equals(loans[[0]])
    assert out.properties.equals(properties[[0]])


def test_no_properties_error_carries_a_count_per_reason() -> None:
    from esma_milan.errors import PipelineDataError

    loans = pl.DataFrame(
        {
            "original_underlying_exposure_identifier": ["L1", "L2"],
            "new_underlying_exposure_identifier": ["L1", "L2"],
            "account_status": ["PERF", "DFLT"],
            "current_principal_balance": [100.0, 5.0],
        }
    )
    properties = _property_row(
        underlying_exposure_identifier=["L2", "L1"],
        calc_property_id=["P1", "P2"],
        property_type=["RHOS", "RHOS"],
        collateral_type=["RBLD", "GUAR"],
        original_valuation_amount=[1.0, 1.0],
        current_valuation_amount=[1.0, 1.0],
    )
    with pytest.raises(PipelineDataError) as excinfo:
        run_stage2(loans, properties)
    assert excinfo.value.context["loan_reason_counts"] == {
        "status_not_active": 1,
        "balance_not_positive": 0,
    }
    assert excinfo.value.context["collateral_reason_counts"] == {
        "property_type_missing": 0,
        "collateral_type_missing": 0,
        "collateral_type_excluded": 1,
        "no_active_loan": 1,
        "no_positive_valuation": 0,
    }


# ---------------------------------------------------------------------------
# Polars errors with a data meaning keep their type and message. These
# sites are not wrapped; the first non-empty line is what the API shows as
# `details.reason`. The literals are what `origin/main` raised before the
# reasons were evaluated in their own step.
# ---------------------------------------------------------------------------


def _first_line(exc: BaseException) -> str:
    return next(line.strip() for line in str(exc).splitlines() if line.strip())


def _stage2_loans(status: pl.Series | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "original_underlying_exposure_identifier": ["L1"],
            "new_underlying_exposure_identifier": ["L1"],
            "account_status": status if status is not None else pl.Series(["PERF"]),
            "current_principal_balance": [1.0],
        }
    )


def _stage2_properties(**overrides: object) -> pl.DataFrame:
    columns: dict[str, object] = {
        "underlying_exposure_identifier": ["L1"],
        "new_collateral_identifier": ["P1"],
        "property_type": ["RHOS"],
        "collateral_type": ["RBLD"],
        "original_valuation_amount": [1.0],
        "current_valuation_amount": [1.0],
    }
    columns.update(overrides)
    return pl.DataFrame({k: v for k, v in columns.items() if v is not None})


_IS_IN_INT64 = "'is_in' cannot check for List(String) values in Int64 data"


def test_untyped_site_status_not_text_keeps_type_and_message() -> None:
    loans = _stage2_loans(pl.Series([1], dtype=pl.Int64))
    with pytest.raises(pl.exceptions.InvalidOperationError) as excinfo:
        run_stage2(loans, _stage2_properties())
    assert _first_line(excinfo.value) == _IS_IN_INT64


def test_untyped_site_original_valuation_amount_absent_keeps_type_and_message() -> None:
    with pytest.raises(pl.exceptions.ColumnNotFoundError) as excinfo:
        run_stage2(_stage2_loans(), _stage2_properties(original_valuation_amount=None))
    assert _first_line(excinfo.value) == (
        'unable to find column "original_valuation_amount"; valid columns: '
        '["underlying_exposure_identifier", "new_collateral_identifier", '
        '"property_type", "collateral_type", "current_valuation_amount"]'
    )


@pytest.mark.parametrize(
    "overrides",
    [{"collateral_type": [7]}, {"underlying_exposure_identifier": [100]}],
    ids=["collateral_type", "underlying_exposure_identifier"],
)
def test_untyped_site_collateral_column_not_text_keeps_type_and_message(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError) as excinfo:
        run_stage2(_stage2_loans(), _stage2_properties(**overrides))
    assert _first_line(excinfo.value) == _IS_IN_INT64
