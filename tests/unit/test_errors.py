"""Tests for `esma_milan.errors`: the typed data-fault error, the reason
codes and the catalogue of expected fault sites.

Each data-fault site is exercised with a constructed input. The expected
message is written out as a literal: it is the message the site raised as
a plain `ValueError` before `PipelineDataError` existed, and it must not
change.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest

from esma_milan.errors import (
    EXPECTED_FAULT_SITES,
    REASON_CODES,
    PipelineDataError,
)
from esma_milan.io_layer.read_csv import read_and_clean
from esma_milan.pipeline.filters import run_stage2
from esma_milan.pipeline.identifiers import (
    generate_id_column,
    run_stage3,
    select_calc_loan_id,
)
from esma_milan.pipeline.validate import validate_required_columns
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"

_TAX = {
    "RREL2": "Original Underlying Exposure Identifier",
    "RREL30": "Current Principal Balance",
}


def _check(
    excinfo: pytest.ExceptionInfo[PipelineDataError],
    *,
    message: str,
    reason: str,
    stage: str,
) -> PipelineDataError:
    err = excinfo.value
    assert type(err) is PipelineDataError
    assert isinstance(err, ValueError)
    assert str(err) == message
    assert err.reason == reason
    assert err.reason in REASON_CODES
    assert err.stage == stage
    return err


# ---------------------------------------------------------------------------
# The class
# ---------------------------------------------------------------------------


def test_pipeline_data_error_is_a_value_error_with_defaults() -> None:
    err = PipelineDataError("boom", stage="stage1", reason="no_data_rows")
    assert isinstance(err, ValueError)
    assert str(err) == "boom"
    assert err.args == ("boom",)
    assert err.items == ()
    assert err.context == {}


def test_reason_codes_are_the_closed_list() -> None:
    assert frozenset({
        "duplicate_columns",
        "no_data_rows",
        "non_numeric_value",
        "missing_required_columns",
        "missing_pool_cutoff_date",
        "no_properties_after_filter",
        "no_usable_loan_id_column",
        "loan_id_coverage_below_minimum",
        "missing_identifier_column",
        "no_loans_after_intersection",
        "loan_identifier_mismatch",
    }) == REASON_CODES


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_every_reason_code_has_exactly_one_typed_site() -> None:
    typed = [s for s in EXPECTED_FAULT_SITES if s.kind == "typed"]
    assert sorted(s.reason or "" for s in typed) == sorted(REASON_CODES)
    assert all(s.exception == "PipelineDataError" for s in typed)
    for site in EXPECTED_FAULT_SITES:
        if site.kind != "typed":
            assert site.reason is None, site


def test_every_site_names_an_importable_function() -> None:
    for site in EXPECTED_FAULT_SITES:
        module_name, _, function_name = site.where.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name)), site.where
        assert site.trigger and site.meaning


def test_sites_not_reachable_from_data_say_why() -> None:
    not_faults = [s for s in EXPECTED_FAULT_SITES if s.kind == "not_a_data_fault"]
    assert len(not_faults) == 2
    for site in not_faults:
        assert "Not reachable from data, because" in site.trigger


def test_the_two_type_error_sites_say_they_are_not_caught_as_input_errors() -> None:
    type_errors = [s for s in EXPECTED_FAULT_SITES if s.exception == "TypeError"]
    assert len(type_errors) == 2
    for site in type_errors:
        assert site.kind == "untyped"
        assert "do not catch it as one today" in site.meaning


# ---------------------------------------------------------------------------
# Stage 1 sites
# ---------------------------------------------------------------------------


def test_duplicate_columns(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("Loan Identifier,loan_identifier,Other,other\nA,B,C,D\n")
    with pytest.raises(PipelineDataError) as excinfo:
        read_and_clean(p, taxonomy={})
    err = _check(
        excinfo,
        message=(
            "CRITICAL ERROR: Duplicated column names found after renaming: "
            "loan_identifier, other. This indicates a problem with the "
            "taxonomy mapping. Multiple ESMA field codes may be mapping to "
            "the same field name."
        ),
        reason="duplicate_columns",
        stage="stage1",
    )
    assert err.items == ("loan_identifier", "other")
    assert err.context == {"file": "loans.csv"}


def test_no_data_rows(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")
    with pytest.raises(PipelineDataError) as excinfo:
        read_and_clean(p, taxonomy={})
    err = _check(
        excinfo,
        message="ERROR: No data rows found in file: empty.csv",
        reason="no_data_rows",
        stage="stage1",
    )
    assert err.items == ()
    assert err.context == {"file": "empty.csv"}


def test_non_numeric_value_carries_every_bad_row_while_the_message_samples(
    tmp_path: Path,
) -> None:
    p = tmp_path / "loans.csv"
    values = ["100", *[f"x{i}" for i in range(12)]]
    p.write_text("RREL2,RREL30\n" + "".join(f"L{i},{v}\n" for i, v in enumerate(values)))
    with pytest.raises(PipelineDataError) as excinfo:
        read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    samples = "".join(f"\n  data row {i + 2}: 'x{i}'" for i in range(10))
    err = _check(
        excinfo,
        message=(
            "ERROR: 12 non-numeric value(s) in numeric field RREL30 "
            "(current_principal_balance) of file loans.csv; the taxonomy types "
            "this field as numeric and every value must parse as a number or "
            "be an ND token." + samples + "\n  ... and 2 more"
        ),
        reason="non_numeric_value",
        stage="stage1",
    )
    # 12 bad rows on the error, 10 in the message; record index is 0-based.
    assert err.items == tuple((i + 1, f"x{i}") for i in range(12))
    assert err.context == {
        "file": "loans.csv",
        "field_code": "RREL30",
        "column": "current_principal_balance",
        "n_bad": 12,
    }


def test_missing_required_columns() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(PipelineDataError) as excinfo:
        validate_required_columns(df, ("a", "b", "c"), "loans_cleaned")
    err = _check(
        excinfo,
        message="ERROR: loans_cleaned is missing required columns: b, c",
        reason="missing_required_columns",
        stage="stage1",
    )
    assert err.items == ("b", "c")
    assert err.context == {"table": "loans_cleaned"}


def test_missing_pool_cutoff_date(tmp_path: Path) -> None:
    text = (SYNTHETIC / "loans.csv").read_text()
    loans = tmp_path / "loans.csv"
    loans.write_text(text.replace("2024-06-30", "ND"))
    with pytest.raises(PipelineDataError) as excinfo:
        run_pipeline(
            loans_file_path=loans,
            collaterals_file_path=SYNTHETIC / "collaterals.csv",
            deal_name="D",
            dry_run=True,
            verbose=False,
        )
    err = _check(
        excinfo,
        message="pool_cutoff_date is missing in loans file",
        reason="missing_pool_cutoff_date",
        stage="runner",
    )
    assert err.items == ()


# ---------------------------------------------------------------------------
# Stage 2 site
# ---------------------------------------------------------------------------


def _loans(
    orig: Sequence[str | None],
    new: Sequence[str | None],
    *,
    status: str = "PERF",
) -> pl.DataFrame:
    n = len(orig)
    return pl.DataFrame(
        {
            "original_underlying_exposure_identifier": pl.Series(list(orig), dtype=pl.String),
            "new_underlying_exposure_identifier": pl.Series(list(new), dtype=pl.String),
            "original_obligor_identifier": [f"B{i}" for i in range(n)],
            "new_obligor_identifier": [f"B{i}" for i in range(n)],
            "account_status": [status] * n,
            "current_principal_balance": [100.0] * n,
        }
    )


def _collateral(loan_ids: Sequence[str | None]) -> pl.DataFrame:
    n = len(loan_ids)
    return pl.DataFrame(
        {
            "underlying_exposure_identifier": pl.Series(list(loan_ids), dtype=pl.String),
            "original_collateral_identifier": [f"P{i}" for i in range(n)],
            "new_collateral_identifier": [f"P{i}" for i in range(n)],
            "property_type": ["RHOS"] * n,
            "collateral_type": ["RBLD"] * n,
            "original_valuation_amount": [100000.0] * n,
            "current_valuation_amount": [100000.0] * n,
        }
    )


def test_no_properties_after_filter() -> None:
    with pytest.raises(PipelineDataError) as excinfo:
        run_stage2(_loans(["O1", "O2"], ["N1", "N2"]), _collateral(["ZZ"]))
    err = _check(
        excinfo,
        message="ERROR: No properties remaining after filtering.",
        reason="no_properties_after_filter",
        stage="stage2",
    )
    assert err.items == ()
    assert err.context["n_loans"] == 2
    assert err.context["n_loans_kept"] == 2
    assert err.context["n_collateral"] == 1


# ---------------------------------------------------------------------------
# Stage 3 sites
# ---------------------------------------------------------------------------


def test_no_usable_loan_id_column_carries_every_duplicated_value() -> None:
    orig = [f"O{i}" for i in range(12)] * 2
    new = [None] * 24
    with pytest.raises(PipelineDataError) as excinfo:
        select_calc_loan_id(_loans(orig, new), _collateral(["O1"]))
    sample = ", ".join(f"O{i}" for i in range(10))
    err = _check(
        excinfo,
        message=(
            "select_calc_loan_id: neither candidate loan-ID column is usable. "
            f"'original_underlying_exposure_identifier': 12 duplicated values [{sample}]. "
            "'new_underlying_exposure_identifier': all NA. "
            "This is a data-quality problem that requires manual investigation."
        ),
        reason="no_usable_loan_id_column",
        stage="stage3",
    )
    assert err.items == tuple(
        ("original_underlying_exposure_identifier", f"O{i}") for i in range(12)
    )
    assert err.context["all_missing"] == {
        "original_underlying_exposure_identifier": False,
        "new_underlying_exposure_identifier": True,
    }


def test_loan_id_coverage_below_minimum_carries_every_unmatched_identifier() -> None:
    ids = [f"L{i:02d}" for i in range(20)]
    with pytest.raises(PipelineDataError) as excinfo:
        select_calc_loan_id(
            _loans(ids, [None] * 20), _collateral([f"L{i:02d}" for i in range(8)])
        )
    sample = ", ".join(f"L{i:02d}" for i in range(8, 18))
    err = _check(
        excinfo,
        message=(
            "select_calc_loan_id: chosen column "
            "'original_underlying_exposure_identifier' has coverage 8/20 = 40.00%, "
            f"below min_coverage = 85.00%. Unmatched sample: [{sample}]."
        ),
        reason="loan_id_coverage_below_minimum",
        stage="stage3",
    )
    # 12 unmatched on the error, 10 in the message.
    assert err.items == tuple(f"L{i:02d}" for i in range(8, 20))
    assert err.context["column"] == "original_underlying_exposure_identifier"
    assert err.context["matched"] == 8
    assert err.context["total"] == 20
    assert err.context["coverage"] == 0.4
    assert err.context["min_coverage"] == 0.85


def test_missing_identifier_column() -> None:
    df = pl.DataFrame({"new_obligor_identifier": ["B1"]})
    with pytest.raises(PipelineDataError) as excinfo:
        generate_id_column(
            df,
            original_id_col="original_obligor_identifier",
            new_id_col="new_obligor_identifier",
            output_col_name="calc_borrower_id",
        )
    err = _check(
        excinfo,
        message="generate_id_column: df is missing column 'original_obligor_identifier'",
        reason="missing_identifier_column",
        stage="stage3",
    )
    assert err.items == ("original_obligor_identifier",)
    assert err.context == {"output_column": "calc_borrower_id"}


def test_missing_new_identifier_column_stays_a_plain_value_error() -> None:
    """The new-identifier columns are required by Stage 1, so this check
    guards a caller's mistake, not the data."""
    df = pl.DataFrame({"original_obligor_identifier": ["B1"]})
    with pytest.raises(ValueError, match="new_obligor_identifier") as excinfo:
        generate_id_column(
            df,
            original_id_col="original_obligor_identifier",
            new_id_col="new_obligor_identifier",
            output_col_name="calc_borrower_id",
        )
    assert type(excinfo.value) is ValueError


def test_no_loans_after_intersection() -> None:
    # Original has duplicates, so new is chosen; no new id has collateral.
    # Only a minimum coverage of 0 lets the run reach the intersection.
    loans = _loans(["O1", "O1", "O3"], ["N1", "N2", "N3"])
    with (
        warnings.catch_warnings(),
        pytest.raises(PipelineDataError) as excinfo,
    ):
        warnings.simplefilter("ignore", UserWarning)
        run_stage3(loans, _collateral(["O1", "O3"]), min_coverage=0.0)
    _check(
        excinfo,
        message="ERROR: No active loans remaining after filtering.",
        reason="no_loans_after_intersection",
        stage="stage3",
    )


def test_loan_identifier_mismatch_carries_every_identifier() -> None:
    """A collateral row that references a loan under the identifier column
    that was not chosen. Twelve such rows: ten in the message, all on the
    error."""
    n = 40
    loans = _loans([f"O{i:02d}" for i in range(n)], [f"N{i:02d}" for i in range(n)])
    stray = [f"O{i:02d}" for i in range(12)]
    collateral = _collateral([*(f"N{i:02d}" for i in range(n)), *stray])
    with pytest.raises(PipelineDataError) as excinfo:
        run_stage3(loans, collateral)
    err = _check(
        excinfo,
        message=(
            "ERROR: Loan identifier mismatch between calc_loan_id and "
            "properties$underlying_exposure_identifier. Missing in properties: 0"
            f"; missing in loans: 12 [{', '.join(stray[:10])}]."
        ),
        reason="loan_identifier_mismatch",
        stage="stage3",
    )
    assert err.items == tuple(("missing_in_loans", v) for v in stray)
    assert err.context == {"n_missing_in_properties": 0, "n_missing_in_loans": 12}
