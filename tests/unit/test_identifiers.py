"""Tests for select_calc_loan_id and generate_id_column.

Mirrors r_reference/tests/testthat/test-select-calc-loan-id.R (20 cases)
and the generate_id_column block in test-utils.R::"generate_id_column
selects the correct column for borrower and property IDs" one-for-one.
Test names track the R `test_that()` strings where applicable so the
correspondence is auditable.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import polars as pl
import pytest

from esma_milan.config import DEFAULT_MIN_LOAN_ID_COVERAGE
from esma_milan.pipeline.identifiers import (
    generate_id_column,
    select_calc_loan_id,
)


def _make_loans(
    orig: Sequence[str | None], new: Sequence[str | None]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "original_underlying_exposure_identifier": pl.Series(
                list(orig), dtype=pl.String
            ),
            "new_underlying_exposure_identifier": pl.Series(
                list(new), dtype=pl.String
            ),
        }
    )


def _make_properties(ids: Sequence[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"underlying_exposure_identifier": pl.Series(list(ids), dtype=pl.String)}
    )


# ---------------------------------------------------------------------------
# Gate-3 selection branches
# ---------------------------------------------------------------------------


def test_both_clean_and_fully_covered_picks_original() -> None:
    """Mirrors 'both clean and fully covered -> picks original (tie-break)'."""
    loans = _make_loans(["O1", "O2", "O3"], ["N1", "N2", "N3"])
    properties = _make_properties(["O1", "O2", "O3", "N1", "N2", "N3"])
    result = select_calc_loan_id(loans, properties)
    assert result["calc_loan_id"].to_list() == ["O1", "O2", "O3"]


def test_both_clean_only_new_fully_covered_picks_new() -> None:
    """Mirrors 'both clean, only new_* fully covered -> picks new_*'."""
    loans = _make_loans(["O1", "O2", "O3"], ["N1", "N2", "N3"])
    properties = _make_properties(["N1", "N2", "N3"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = select_calc_loan_id(loans, properties, min_coverage=0)
    assert result["calc_loan_id"].to_list() == ["N1", "N2", "N3"]


def test_original_has_duplicates_new_clean_and_covered_picks_new() -> None:
    """Mirrors 'original_* has duplicates, new_* clean and covered -> picks new_*'."""
    loans = _make_loans(["O1", "O1", "O2"], ["N1", "N2", "N3"])
    properties = _make_properties(["N1", "N2", "N3"])
    result = select_calc_loan_id(loans, properties)
    assert result["calc_loan_id"].to_list() == ["N1", "N2", "N3"]


def test_new_has_duplicates_original_clean_picks_original() -> None:
    """Mirrors 'new_* has duplicates, original_* clean -> picks original_*'."""
    loans = _make_loans(["O1", "O2", "O3"], ["N1", "N1", "N2"])
    properties = _make_properties(["O1", "O2", "O3"])
    result = select_calc_loan_id(loans, properties)
    assert result["calc_loan_id"].to_list() == ["O1", "O2", "O3"]


def test_both_columns_have_duplicates_errors_with_diagnostics() -> None:
    """Mirrors 'both columns have duplicates -> errors with duplicate info'."""
    loans = _make_loans(["O1", "O1", "O2"], ["N1", "N1", "N2"])
    properties = _make_properties(["O1", "N1"])
    with pytest.raises(ValueError) as excinfo:
        select_calc_loan_id(loans, properties)
    msg = str(excinfo.value)
    assert "neither candidate loan-ID column is usable" in msg
    assert "original_underlying_exposure_identifier" in msg
    assert "new_underlying_exposure_identifier" in msg


def test_both_partial_new_higher_picks_new_with_warning() -> None:
    """Mirrors 'both clean, partial coverage, new_* higher -> picks new_* with warning'.

    original: O1..O4, properties match O1 only      = 25%
    new:      N1..N4, properties match N1..N3      = 75%
    """
    loans = _make_loans(["O1", "O2", "O3", "O4"], ["N1", "N2", "N3", "N4"])
    properties = _make_properties(["O1", "N1", "N2", "N3"])
    with pytest.warns(UserWarning, match="partial coverage"):
        result = select_calc_loan_id(loans, properties, min_coverage=0.5)
    assert result["calc_loan_id"].to_list() == ["N1", "N2", "N3", "N4"]


def test_equal_partial_coverage_defaults_to_original_with_warning() -> None:
    """Mirrors 'both clean, equal partial coverage -> defaults to original with warning'."""
    loans = _make_loans(["O1", "O2"], ["N1", "N2"])
    properties = _make_properties(["O1", "N1"])
    with pytest.warns(UserWarning, match="equal partial coverage"):
        result = select_calc_loan_id(loans, properties, min_coverage=0.4)
    assert result["calc_loan_id"].to_list() == ["O1", "O2"]


# ---------------------------------------------------------------------------
# min_coverage threshold check
# ---------------------------------------------------------------------------


def test_chosen_coverage_below_min_coverage_errors() -> None:
    """Mirrors 'chosen column coverage below min_coverage -> errors'."""
    loans = _make_loans(["O1", "O2", "O3"], ["N1", "N2", "N3"])
    properties = _make_properties(["N1", "N2"])  # 2/3 = 66.7%
    with pytest.raises(ValueError, match="below min_coverage"):
        select_calc_loan_id(loans, properties, min_coverage=0.99)


def test_coverage_at_min_coverage_below_one_succeeds_with_warning() -> None:
    """Mirrors 'coverage at min_coverage (below 1.0) -> succeeds with warning'."""
    loans = _make_loans(["O1", "O2", "O3"], ["N1", "N2", "N3"])
    properties = _make_properties(["N1", "N2"])  # 2/3
    with pytest.warns(UserWarning, match="partial coverage"):
        result = select_calc_loan_id(loans, properties, min_coverage=0.5)
    assert result["calc_loan_id"].to_list() == ["N1", "N2", "N3"]


def test_chosen_fully_covered_min_coverage_one_emits_no_warning() -> None:
    """Mirrors 'chosen column fully covered, min_coverage=1.0 -> no warning'."""
    loans = _make_loans(["O1", "O2"], ["N1", "N2"])
    properties = _make_properties(["O1", "O2"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = select_calc_loan_id(loans, properties, min_coverage=1.0)
    assert result["calc_loan_id"].to_list() == ["O1", "O2"]


# ---------------------------------------------------------------------------
# all-NA handling
# ---------------------------------------------------------------------------


def test_one_column_all_na_picks_clean_one() -> None:
    """Mirrors 'one column all-NA, other clean and covered -> picks clean one'."""
    loans = _make_loans([None, None, None], ["N1", "N2", "N3"])
    properties = _make_properties(["N1", "N2", "N3"])
    result = select_calc_loan_id(loans, properties)
    assert result["calc_loan_id"].to_list() == ["N1", "N2", "N3"]


def test_both_columns_all_na_errors_at_gate_1() -> None:
    """Mirrors 'both columns all-NA -> errors at Gate 1'."""
    loans = _make_loans([None, None], [None, None])
    properties = _make_properties(["X1", "X2"])
    with pytest.raises(ValueError, match="neither candidate loan-ID column"):
        select_calc_loan_id(loans, properties)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [1.5, -0.1, "0.9", [0.5, 0.9]])
def test_min_coverage_out_of_range_errors_on_param_validation(bad: object) -> None:
    """Mirrors 'min_coverage out of range -> errors on parameter validation'."""
    loans = _make_loans(["O1"], ["N1"])
    properties = _make_properties(["O1"])
    with pytest.raises(ValueError, match=r"min_coverage.*\[0, 1\]"):
        select_calc_loan_id(loans, properties, min_coverage=bad)  # type: ignore[arg-type]


def test_missing_original_id_col_errors() -> None:
    """Mirrors 'missing expected columns -> errors naming the missing column' (orig)."""
    loans = pl.DataFrame({"new_underlying_exposure_identifier": ["N1"]})
    properties = _make_properties(["N1"])
    with pytest.raises(ValueError, match="original_underlying_exposure_identifier"):
        select_calc_loan_id(loans, properties)


def test_missing_new_id_col_errors() -> None:
    loans = pl.DataFrame({"original_underlying_exposure_identifier": ["O1"]})
    properties = _make_properties(["O1"])
    with pytest.raises(ValueError, match="new_underlying_exposure_identifier"):
        select_calc_loan_id(loans, properties)


def test_missing_property_id_col_errors() -> None:
    loans = _make_loans(["O1"], ["N1"])
    properties = pl.DataFrame({"some_other_col": ["N1"]})
    with pytest.raises(ValueError, match="underlying_exposure_identifier"):
        select_calc_loan_id(loans, properties)


# ---------------------------------------------------------------------------
# Default min_coverage and the loosened tie-break (equal partial coverage)
# ---------------------------------------------------------------------------


def test_default_min_loan_id_coverage_is_85_percent() -> None:
    """Mirrors 'DEFAULT_MIN_LOAN_ID_COVERAGE exists and is 0.85'."""
    assert DEFAULT_MIN_LOAN_ID_COVERAGE == 0.85


def test_tied_99pct_partial_coverage_defaults_to_original_warns_with_unmatched() -> None:
    """Mirrors 'tied ~99% partial coverage -> defaults to original, warns with unmatched sample'.

    Both candidate columns identical, 99/100 covered. Gate 3 emits an
    'equal partial coverage' warning plus a post-selection 'partial
    coverage 99/100' warning, both naming the unmatched ID.
    """
    ids = [f"L{i:03d}" for i in range(1, 101)]
    loans = _make_loans(ids, ids)
    properties = _make_properties(ids[:-1])  # drop L100

    seen: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = select_calc_loan_id(loans, properties)
        seen = [str(w.message) for w in caught]

    assert result["calc_loan_id"].to_list() == ids
    assert any("equal partial coverage" in m for m in seen)
    assert any("partial coverage" in m and "99/100" in m for m in seen)
    assert any("L100" in m for m in seen)


def test_partial_coverage_90pct_above_default_threshold_warns() -> None:
    """Mirrors 'partial coverage ~90% (above default threshold) -> warns
    with coverage and unmatched sample'."""
    loans = _make_loans([f"O{i}" for i in range(1, 11)], [f"N{i}" for i in range(1, 11)])
    properties = _make_properties([f"O{i}" for i in range(1, 10)])  # 9/10 = 90%

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = select_calc_loan_id(loans, properties)
        seen = [str(w.message) for w in caught]

    assert result["calc_loan_id"].to_list() == [f"O{i}" for i in range(1, 11)]
    assert any("90.00%" in m or "9/10" in m for m in seen)
    assert any("Unmatched sample" in m and "O10" in m for m in seen)


def test_partial_coverage_70pct_below_default_threshold_errors() -> None:
    """Mirrors 'partial coverage ~70% (below default threshold) -> errors
    with unmatched sample'."""
    loans = _make_loans([f"O{i}" for i in range(1, 11)], [f"N{i}" for i in range(1, 11)])
    properties = _make_properties([f"O{i}" for i in range(1, 8)])  # 7/10 = 70%

    with pytest.raises(ValueError) as excinfo:
        select_calc_loan_id(loans, properties)
    msg = str(excinfo.value)
    assert "below min_coverage" in msg
    assert "Unmatched sample" in msg
    # At least one of the unmatched ids appears in the error message.
    assert any(uid in msg for uid in ("O8", "O9", "O10"))


def test_tied_partial_coverage_70pct_ultimately_errors() -> None:
    """Mirrors 'tied partial coverage ~70% -> ultimately errors via threshold check'."""
    ids = [f"L{i}" for i in range(1, 11)]
    loans = _make_loans(ids, ids)
    properties = _make_properties(ids[:7])
    with (
        pytest.raises(ValueError, match="below min_coverage"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", UserWarning)
        select_calc_loan_id(loans, properties)


def test_full_coverage_on_both_candidates_no_warnings() -> None:
    """Mirrors 'full coverage on both candidates -> defaults to original
    with no warnings'."""
    loans = _make_loans(["L1", "L2", "L3"], ["L1", "L2", "L3"])
    properties = _make_properties(["L1", "L2", "L3"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = select_calc_loan_id(loans, properties)
    assert result["calc_loan_id"].to_list() == ["L1", "L2", "L3"]


# ---------------------------------------------------------------------------
# generate_id_column — mirrors test-utils.R
# ---------------------------------------------------------------------------


def _two_col(a_name: str, a: Sequence[str | None], b_name: str, b: Sequence[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            a_name: pl.Series(list(a), dtype=pl.String),
            b_name: pl.Series(list(b), dtype=pl.String),
        }
    )


def test_generate_id_identical_columns_picks_original() -> None:
    """Mirrors test-utils.R: 'Case 1: identical -> original'."""
    df = _two_col(
        "original_obligor_identifier", ["B1", "B2"],
        "new_obligor_identifier", ["B1", "B2"],
    )
    result = generate_id_column(
        df,
        original_id_col="original_obligor_identifier",
        new_id_col="new_obligor_identifier",
        output_col_name="calc_borrower_id",
    )
    assert result["calc_borrower_id"].to_list() == ["B1", "B2"]


def test_generate_id_new_more_unique_picks_new() -> None:
    """Mirrors 'Case 2: new has more unique values -> new'."""
    df = _two_col(
        "original_collateral_identifier", ["P1", "P1"],
        "new_collateral_identifier", ["P1", "P2"],
    )
    result = generate_id_column(
        df,
        original_id_col="original_collateral_identifier",
        new_id_col="new_collateral_identifier",
        output_col_name="calc_property_id",
    )
    assert result["calc_property_id"].to_list() == ["P1", "P2"]


def test_generate_id_original_more_unique_picks_original() -> None:
    """Mirrors 'Case 3: original has more unique values -> original'."""
    df = _two_col(
        "original_obligor_identifier", ["B1", "B2"],
        "new_obligor_identifier", ["B1", "B1"],
    )
    result = generate_id_column(
        df,
        original_id_col="original_obligor_identifier",
        new_id_col="new_obligor_identifier",
        output_col_name="calc_borrower_id",
    )
    assert result["calc_borrower_id"].to_list() == ["B1", "B2"]


def test_generate_id_tied_unique_count_picks_original() -> None:
    """Both columns have the same unique count but are NOT identical
    (different values). Tie-break: original wins."""
    df = _two_col(
        "original_obligor_identifier", ["B1", "B2"],
        "new_obligor_identifier", ["X1", "X2"],
    )
    result = generate_id_column(
        df,
        original_id_col="original_obligor_identifier",
        new_id_col="new_obligor_identifier",
        output_col_name="calc_borrower_id",
    )
    assert result["calc_borrower_id"].to_list() == ["B1", "B2"]


def test_generate_id_nulls_excluded_from_unique_count() -> None:
    """is.na values must NOT count toward uniqueness - matches R's
    `length(unique(x[!is.na(x)]))`. Original has 2 unique non-null
    values, new has 1; original wins."""
    df = _two_col(
        "original_obligor_identifier", ["B1", "B2", None],
        "new_obligor_identifier", ["X1", None, None],
    )
    result = generate_id_column(
        df,
        original_id_col="original_obligor_identifier",
        new_id_col="new_obligor_identifier",
        output_col_name="calc_borrower_id",
    )
    assert result["calc_borrower_id"].to_list() == ["B1", "B2", None]


def test_generate_id_identical_with_matching_nulls_picks_original() -> None:
    """Two columns with NAs in the same positions are still identical.
    Mirrors R's `identical(original_ids, new_ids)` returning TRUE for
    identical NA patterns."""
    df = _two_col(
        "original_obligor_identifier", ["B1", None, "B3"],
        "new_obligor_identifier", ["B1", None, "B3"],
    )
    result = generate_id_column(
        df,
        original_id_col="original_obligor_identifier",
        new_id_col="new_obligor_identifier",
        output_col_name="calc_borrower_id",
    )
    assert result["calc_borrower_id"].to_list() == ["B1", None, "B3"]


def test_generate_id_missing_original_col_errors() -> None:
    df = pl.DataFrame({"new_obligor_identifier": ["B1"]})
    with pytest.raises(ValueError, match="original_obligor_identifier"):
        generate_id_column(
            df,
            original_id_col="original_obligor_identifier",
            new_id_col="new_obligor_identifier",
            output_col_name="calc_borrower_id",
        )


def test_generate_id_missing_new_col_errors() -> None:
    df = pl.DataFrame({"original_obligor_identifier": ["B1"]})
    with pytest.raises(ValueError, match="new_obligor_identifier"):
        generate_id_column(
            df,
            original_id_col="original_obligor_identifier",
            new_id_col="new_obligor_identifier",
            output_col_name="calc_borrower_id",
        )


# ---------------------------------------------------------------------------
# Stage 3 decisions as data
# ---------------------------------------------------------------------------

import structlog  # noqa: E402

from esma_milan.errors import PipelineDataError  # noqa: E402
from esma_milan.pipeline.filters import run_stage2  # noqa: E402
from esma_milan.pipeline.identifiers import (  # noqa: E402
    LoanIdSelection,
    _generate_id_column,
    _select_calc_loan_id,
    run_stage3,
)

_ORIG = "original_underlying_exposure_identifier"
_NEW = "new_underlying_exposure_identifier"


def _stage3_loans(
    orig: Sequence[str | None],
    new: Sequence[str | None],
    *,
    statuses: Sequence[str] | None = None,
) -> pl.DataFrame:
    n = len(orig)
    return pl.DataFrame(
        {
            _ORIG: pl.Series(list(orig), dtype=pl.String),
            _NEW: pl.Series(list(new), dtype=pl.String),
            "original_obligor_identifier": [f"BO{i}" for i in range(n)],
            "new_obligor_identifier": [f"BN{i}" for i in range(n)],
            "account_status": list(statuses) if statuses else ["PERF"] * n,
            "current_principal_balance": [100.0] * n,
        }
    )


def _stage3_properties(loan_ids: Sequence[str | None]) -> pl.DataFrame:
    n = len(loan_ids)
    return pl.DataFrame(
        {
            "underlying_exposure_identifier": pl.Series(list(loan_ids), dtype=pl.String),
            "original_collateral_identifier": [f"PO{i}" for i in range(n)],
            "new_collateral_identifier": [f"PN{i}" for i in range(n)],
            "property_type": ["RHOS"] * n,
            "collateral_type": ["RBLD"] * n,
            "original_valuation_amount": [100000.0] * n,
            "current_valuation_amount": [100000.0] * n,
        }
    )


def test_selection_reports_both_candidates_and_the_choice() -> None:
    loans = _make_loans(["O1", "O2", "O3", "O4"], ["N1", "N2", "N3", "N4"])
    properties = _make_properties(["O1", "N1", "N2", "N3"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        frame, selection = _select_calc_loan_id(loans, properties, min_coverage=0.5)
    assert frame["calc_loan_id"].to_list() == ["N1", "N2", "N3", "N4"]
    assert selection.chosen_column == _NEW
    assert selection.reason == "higher coverage than alternative"
    assert selection.min_coverage == 0.5
    original, new = selection.candidates
    assert (original.column, original.total, original.matched, original.coverage) == (
        _ORIG, 4, 1, 0.25,
    )
    assert original.unmatched == ("O2", "O3", "O4")
    assert (new.column, new.total, new.matched, new.coverage) == (_NEW, 4, 3, 0.75)
    assert new.unmatched == ("N4",)
    assert not original.disqualified and not new.disqualified
    assert original.duplicated_values == () and new.duplicated_values == ()


def test_selection_duplicate_disqualification_lists_every_duplicated_value() -> None:
    loans = _make_loans(["O1", "O1", "O2", "O2", "O3"], ["N1", "N2", "N3", "N4", "N5"])
    properties = _make_properties(["N1", "N2", "N3", "N4", "N5"])
    _, selection = _select_calc_loan_id(loans, properties)
    original, new = selection.candidates
    assert original.has_duplicates and original.disqualified and not original.all_missing
    assert original.duplicated_values == ("O1", "O2")
    # Total and matched are not computed for a disqualified column.
    assert (original.total, original.matched, original.coverage) == (0, 0, 0.0)
    assert original.unmatched == ()
    assert selection.chosen_column == _NEW
    assert selection.reason == "only duplicate-free candidate"
    assert (new.total, new.matched, new.coverage) == (5, 5, 1.0)


def test_selection_all_missing_disqualification() -> None:
    loans = _make_loans([None, None], ["N1", "N2"])
    _, selection = _select_calc_loan_id(loans, _make_properties(["N1", "N2"]))
    original, _new = selection.candidates
    assert original.all_missing and original.disqualified and not original.has_duplicates
    assert selection.chosen_column == _NEW


def test_partial_coverage_unmatched_list_equals_the_warning_sample() -> None:
    """Ten or fewer unmatched: the complete list is what the warning shows."""
    loans = _make_loans([f"O{i}" for i in range(1, 11)], [f"N{i}" for i in range(1, 11)])
    properties = _make_properties([f"O{i}" for i in range(1, 10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        _, selection = _select_calc_loan_id(loans, properties)
    chosen = selection.candidates[0]
    assert chosen.unmatched == ("O10",)
    assert any(
        f"partial coverage {chosen.matched}/{chosen.total} = 90.00%. "
        f"Unmatched sample: [{', '.join(chosen.unmatched)}]." in str(w.message)
        for w in caught
    )


def test_coverage_failure_carries_the_selection_with_both_candidates() -> None:
    loans = _make_loans([f"O{i}" for i in range(1, 11)], [f"N{i}" for i in range(1, 11)])
    properties = _make_properties([f"O{i}" for i in range(1, 8)])
    with pytest.raises(PipelineDataError) as excinfo:
        _select_calc_loan_id(loans, properties)
    selection = excinfo.value.context["loan_id_selection"]
    assert isinstance(selection, LoanIdSelection)
    assert selection.chosen_column == _ORIG
    assert selection.reason == "higher coverage than alternative"
    assert selection.min_coverage == DEFAULT_MIN_LOAN_ID_COVERAGE
    original, new = selection.candidates
    assert (original.total, original.matched, original.coverage) == (10, 7, 0.7)
    assert original.unmatched == ("O10", "O8", "O9")  # sorted as strings
    assert (new.total, new.matched, new.coverage) == (10, 0, 0.0)
    assert len(new.unmatched) == 10
    assert excinfo.value.items == original.unmatched


def test_no_usable_column_failure_carries_the_selection_without_a_choice() -> None:
    loans = _make_loans(["O1", "O1", "O2"], [None, None, None])
    with pytest.raises(PipelineDataError) as excinfo:
        _select_calc_loan_id(loans, _make_properties(["O1"]), min_coverage=0.9)
    selection = excinfo.value.context["loan_id_selection"]
    assert isinstance(selection, LoanIdSelection)
    assert selection.chosen_column is None and selection.reason is None
    assert selection.min_coverage == 0.9
    original, new = selection.candidates
    assert original.disqualified and original.duplicated_values == ("O1",)
    assert new.disqualified and new.all_missing and new.duplicated_values == ()


@pytest.mark.parametrize(
    ("original", "new", "decision", "chosen_is_new", "identical"),
    [
        (["B1", "B2"], ["B1", "B2"], "identical, using original", False, True),
        (["B1", "B2"], ["B1", "B1"], "original has more unique values", False, False),
        (["B1", "B1"], ["B1", "B2"], "new has more unique values", True, False),
        (["B1", "B2"], ["X1", "X2"], "tied, defaulting to original", False, False),
    ],
)
def test_id_column_choice_for_each_decision(
    original: list[str], new: list[str], decision: str, chosen_is_new: bool, identical: bool
) -> None:
    df = _two_col("original_obligor_identifier", original, "new_obligor_identifier", new)
    with structlog.testing.capture_logs() as logs:
        frame, choice = _generate_id_column(
            df,
            original_id_col="original_obligor_identifier",
            new_id_col="new_obligor_identifier",
            output_col_name="calc_borrower_id",
        )
    assert choice.decision == decision
    assert choice.identical is identical
    assert choice.output_column == "calc_borrower_id"
    assert choice.original_column == "original_obligor_identifier"
    assert choice.new_column == "new_obligor_identifier"
    expected_column = "new_obligor_identifier" if chosen_is_new else "original_obligor_identifier"
    assert choice.chosen_column == expected_column
    assert frame["calc_borrower_id"].to_list() == df[expected_column].to_list()
    (event,) = [e for e in logs if e["event"] == "generate_id_column"]
    assert (choice.n_unique_original, choice.n_unique_new, choice.decision) == (
        event["n_unique_original"], event["n_unique_new"], event["decision"],
    )


def test_stage3_reports_the_loans_it_dropped_as_positions_in_its_input() -> None:
    """Six loans reach Stage 3. The new column is chosen (the original has
    a duplicate). One loan has no chosen identifier, two have no
    collateral row; three are kept."""
    loans = _stage3_loans(
        ["O1", "O1", "O3", "O4", "O5", "O6"],
        ["N1", "N2", None, "N4", "N5", "N6"],
    )
    properties = _stage3_properties(["N1", "N1", "N4", "N6"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        out = run_stage3(loans, properties, min_coverage=0.5)
    decisions = out.decisions
    assert decisions is not None
    assert decisions.loan_id.chosen_column == _NEW
    assert decisions.loans_missing_calc_loan_id.to_list() == [2]
    assert decisions.loans_without_collateral.to_list() == [1, 4]
    assert decisions.loan_rows.to_list() == [0, 3, 5]
    assert decisions.property_rows.to_list() == [0, 1, 2, 3]
    assert decisions.unmatched_loan_ids == ("N2", "N5")
    # Empty on every successful run today (issue #45).
    assert decisions.properties_removed.to_list() == []
    assert decisions.unmatched_collateral_ids == ()
    for series in (
        decisions.loan_rows,
        decisions.property_rows,
        decisions.loans_missing_calc_loan_id,
        decisions.loans_without_collateral,
        decisions.properties_removed,
    ):
        assert series.dtype == pl.UInt32
    assert out.loans.drop("calc_loan_id", "calc_borrower_id").equals(
        loans[decisions.loan_rows]
    )
    assert out.properties.drop("calc_property_id").equals(
        properties[decisions.property_rows]
    )
    assert any("Dropped 1 loans with missing calc_loan_id" in str(w.message) for w in caught)
    assert decisions.borrower_id.output_column == "calc_borrower_id"
    assert decisions.borrower_id.decision == "tied, defaulting to original"
    assert decisions.property_id.output_column == "calc_property_id"
    assert decisions.property_id.chosen_column == "original_collateral_identifier"


def test_stage3_positions_translate_to_file_records_through_stage2_rows() -> None:
    """Every position in Stage3Decisions is a position in the Stage 3
    input, which is the Stage 2 output; Stage2Output.loan_rows turns it
    into the record's position in the file. Here Stage 2 drops file
    records 1 and 4, so the two numberings differ."""
    loans = _stage3_loans(
        ["O0", "O1", "O0", "O3", "O4", "O5", "O6", "O7"],
        ["N0", "N1", "N2", None, "N4", "N5", "N6", "N7"],
        statuses=["PERF", "DFLT", "PERF", "PERF", "RDMD", "PERF", "PERF", "PERF"],
    )
    properties = _stage3_properties(["N0", "N2", "N1", "N6", "N7"])
    stage2 = run_stage2(loans, properties)
    assert stage2.loan_rows is not None and stage2.property_rows is not None
    assert stage2.loan_rows.to_list() == [0, 2, 3, 5, 6, 7]
    assert stage2.property_rows.to_list() == [0, 1, 3, 4]  # N1's loan is not active

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stage3 = run_stage3(stage2.loans, stage2.properties, min_coverage=0.5)
    decisions = stage3.decisions
    assert decisions is not None

    # In the Stage 3 input (six loans): position 2 has no new identifier,
    # position 3 (N5) has no collateral row.
    assert decisions.loans_missing_calc_loan_id.to_list() == [2]
    assert decisions.loans_without_collateral.to_list() == [3]
    # In the file: records 3 and 5.
    missing_in_file = stage2.loan_rows.gather(decisions.loans_missing_calc_loan_id)
    without_collateral_in_file = stage2.loan_rows.gather(decisions.loans_without_collateral)
    assert missing_in_file.to_list() == [3]
    assert without_collateral_in_file.to_list() == [5]
    assert loans[_NEW].gather(missing_in_file).to_list() == [None]
    assert loans[_NEW].gather(without_collateral_in_file).to_list() == ["N5"]
    # The loans kept, as file records.
    kept_in_file = stage2.loan_rows.gather(decisions.loan_rows)
    assert kept_in_file.to_list() == [0, 2, 6, 7]
    assert stage3.loans["calc_loan_id"].to_list() == loans[_NEW].gather(kept_in_file).to_list()
    assert stage2.property_rows.gather(decisions.property_rows).to_list() == [0, 1, 3, 4]
