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
