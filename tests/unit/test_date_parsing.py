"""Tests for esma_milan.io_layer.date_parsing.

Mirrors r_reference/tests/testthat/test-parse-iso-or-excel-date.R
(11 test cases) and test-collapse-multi-date-restructuring.R (9 cases),
one-for-one. Test names match the R `test_that()` strings so the
correspondence is auditable.
"""

from __future__ import annotations

import warnings
from datetime import date

import polars as pl
import pytest
import structlog

from esma_milan.io_layer.date_parsing import (
    collapse_multi_date_restructuring,
    parse_iso_or_excel_date,
    parse_iso_or_excel_date_values,
)

# ---------------------------------------------------------------------------
# parse_iso_or_excel_date — mirrors test-parse-iso-or-excel-date.R
# ---------------------------------------------------------------------------


def test_parses_iso_strings() -> None:
    """Mirrors 'parse_iso_or_excel_date parses ISO strings'."""
    result = parse_iso_or_excel_date_values(["2023-12-31", "2024-06-15"], "test")
    assert result == [date(2023, 12, 31), date(2024, 6, 15)]


def test_parses_yyyy_slash_mm_slash_dd_strings() -> None:
    """Mirrors 'parse_iso_or_excel_date parses YYYY/MM/DD strings'."""
    result = parse_iso_or_excel_date_values(["2023/12/31", "2024/06/15"], "test")
    assert result == [date(2023, 12, 31), date(2024, 6, 15)]


def test_parses_excel_serials_from_user_sample() -> None:
    """Mirrors 'parse_iso_or_excel_date parses Excel serials from the user's sample'.

    NOTE - R-test discrepancy flagged per project brief §10:
    The R test's comment claims 42642 = 2016-10-01 and 43264 = 2018-06-15,
    but R's actual `as.Date(numeric, origin="1899-12-30")` math (which is
    `as.Date("1899-12-30") + n`, i.e. proleptic Gregorian addition)
    produces 2016-09-29 and 2018-06-13 respectively for those inputs.
    Verified independently with Python's `datetime` arithmetic, which
    uses the same proleptic-Gregorian calendar as R.

    The R test as written should be failing assertion. This Python port
    matches the **mathematics** R is performing (which is also what the
    pipeline does for round-tripping serials in Cleaned ESMA loans /
    properties / Combined flattened pool sheets - serial 43539 in the
    synthetic fixture's expected output corresponds to 2019-03-15, the
    origination_date for ORIG_001, with origin=1899-12-30).
    """
    result = parse_iso_or_excel_date_values(["42642", "43264"], "test")
    assert result == [date(2016, 9, 29), date(2018, 6, 13)]


def test_parses_a_mixed_vector() -> None:
    """Mirrors 'parse_iso_or_excel_date parses a mixed vector'.

    Same R-test-comment caveat as test_parses_excel_serials_from_user_sample.
    """
    inputs: list[object] = ["2023-12-31", "42642", "ND", "", "2024/06/15", "43264", None]
    result = parse_iso_or_excel_date_values(inputs, "test")
    assert result[0] == date(2023, 12, 31)
    assert result[1] == date(2016, 9, 29)
    assert result[2] is None
    assert result[3] is None
    assert result[4] == date(2024, 6, 15)
    assert result[5] == date(2018, 6, 13)
    assert result[6] is None


def test_accepts_numeric_input() -> None:
    """Mirrors 'parse_iso_or_excel_date accepts numeric input'.

    Same R-test-comment caveat as test_parses_excel_serials_from_user_sample.
    """
    result = parse_iso_or_excel_date_values([42642, 43264], "test")
    assert result == [date(2016, 9, 29), date(2018, 6, 13)]


def test_round_trip_with_synthetic_fixture_origination_date() -> None:
    """Anchor: the synthetic fixture writes 2019-03-15 as Excel serial
    43539. Reading 43539 must round-trip back to 2019-03-15 - this
    confirms my implementation matches the pipeline's actual write side
    regardless of the R test-comment discrepancy."""
    result = parse_iso_or_excel_date_values([43539], "origination_date")
    assert result == [date(2019, 3, 15)]


def test_passes_date_input_through() -> None:
    """Mirrors 'parse_iso_or_excel_date passes Date input through'."""
    inputs = [date(2023, 1, 1), date(2024, 6, 15)]
    result = parse_iso_or_excel_date_values(inputs, "test")
    assert result == inputs


def test_treats_all_esma_nd_codes_as_na() -> None:
    """Mirrors 'parse_iso_or_excel_date treats all ESMA ND codes as NA'."""
    nd = ["ND", "ND1", "ND2", "ND3", "ND4", "ND5", "NA"]
    result = parse_iso_or_excel_date_values(nd, "test")
    assert all(v is None for v in result), result


def test_warns_on_unparseable_values_and_returns_na() -> None:
    """Mirrors 'parse_iso_or_excel_date warns on unparseable values and returns NA'."""
    with pytest.warns(UserWarning, match="origination_date"):
        result = parse_iso_or_excel_date_values(
            ["2023-12-31", "31/12/2023", "not-a-date"],
            "origination_date",
        )
    assert result[0] == date(2023, 12, 31)
    assert result[1] is None
    assert result[2] is None


def test_warning_truncates_examples_to_first_five() -> None:
    """Spot-check that the warning shows up to 5 distinct bad values."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_iso_or_excel_date_values(["a", "b", "c", "d", "e", "f", "g"], "col1")
    assert any("a, b, c, d, e" in str(w.message) for w in caught)
    assert all(", f," not in str(w.message) for w in caught)


def test_rejects_serials_outside_plausible_range() -> None:
    """Mirrors 'parse_iso_or_excel_date rejects serials outside the plausible range'."""
    with pytest.warns(UserWarning, match="test"):
        result = parse_iso_or_excel_date_values(["100", "-999999"], "test")
    # 100 -> 1900-04-09, inside default range -> kept.
    assert result[0] == date(1900, 4, 9)
    assert result[1] is None


def test_respects_a_custom_serial_range() -> None:
    """Mirrors 'parse_iso_or_excel_date respects a custom serial_range'."""
    with pytest.warns(UserWarning, match="test"):
        result = parse_iso_or_excel_date_values(
            ["42642", "2023-12-31"],
            "test",
            serial_range=(date(2020, 1, 1), date(2100, 1, 1)),
        )
    # 42642 -> 2016-10-01, outside the custom range -> rejected.
    assert result[0] is None
    assert result[1] == date(2023, 12, 31)


def test_handles_empty_input() -> None:
    """Mirrors 'parse_iso_or_excel_date handles empty input'."""
    result = parse_iso_or_excel_date_values([], "test")
    assert result == []


# ---------------------------------------------------------------------------
# parse_iso_or_excel_date — Polars Series wrapper
# ---------------------------------------------------------------------------


def test_polars_wrapper_returns_date_series() -> None:
    series = pl.Series("origination_date", ["2023-12-31", "42642", "ND"])
    result = parse_iso_or_excel_date(series, "origination_date")
    assert result.dtype == pl.Date
    assert result.name == "origination_date"
    # serial 42642 -> 2016-09-29; see test_parses_excel_serials_from_user_sample.
    assert result.to_list() == [date(2023, 12, 31), date(2016, 9, 29), None]


def test_polars_wrapper_passes_date_dtype_through() -> None:
    series = pl.Series(
        "origination_date",
        [date(2023, 12, 31), date(2024, 6, 15)],
        dtype=pl.Date,
    )
    result = parse_iso_or_excel_date(series, "origination_date")
    # Same identity (passthrough) - check dtype and values.
    assert result.dtype == pl.Date
    assert result.to_list() == [date(2023, 12, 31), date(2024, 6, 15)]


# ---------------------------------------------------------------------------
# collapse_multi_date_restructuring — mirrors test-collapse-multi-date-restructuring.R
# ---------------------------------------------------------------------------


def test_no_delimiters_present_input_returned_unchanged() -> None:
    """Mirrors 'no delimiters present: input returned unchanged and no message fires'."""
    input_vals: list[object] = ["2020-01-01", "2021-06-15", None, ""]
    with structlog.testing.capture_logs() as logs:
        result = collapse_multi_date_restructuring(input_vals)
    assert result == ["2020-01-01", "2021-06-15", None, ""]
    assert all(
        e.get("event") != "collapse_multi_date_restructuring" for e in logs
    ), "no log should fire when there are no delimited cells"


def test_comma_separated_two_date_cell_collapses_to_max() -> None:
    """Mirrors 'comma-separated two-date cell collapses to the max'."""
    assert collapse_multi_date_restructuring(["2015-05-26,2015-05-27"]) == ["2015-05-27"]


def test_semicolon_separated_two_date_cell_collapses_to_max() -> None:
    """Mirrors 'semicolon-separated two-date cell collapses to the max'."""
    assert collapse_multi_date_restructuring(["2020-04-28;2026-01-16"]) == ["2026-01-16"]


def test_three_date_cell_with_mixed_whitespace_collapses_to_max() -> None:
    """Mirrors 'three-date cell with mixed whitespace collapses to the max'."""
    assert collapse_multi_date_restructuring(
        ["2019-02-04, 2019-12-27 ,2024-04-17"]
    ) == ["2024-04-17"]


def test_mixed_vector_scalars_pass_through_multidates_collapse_log_reports_2() -> None:
    """Mirrors 'mixed vector: ... message reports 2 collapsed'."""
    inputs: list[object] = [
        "2020-01-01",
        "2015-05-26,2015-05-27",
        None,
        "2019-02-04,2019-12-27",
    ]
    with structlog.testing.capture_logs() as logs:
        result = collapse_multi_date_restructuring(inputs)
    assert result == ["2020-01-01", "2015-05-27", None, "2019-12-27"]
    collapse_logs = [e for e in logs if e.get("event") == "collapse_multi_date_restructuring"]
    assert len(collapse_logs) == 1
    assert collapse_logs[0]["n_collapsed"] == 2
    assert collapse_logs[0]["n_unparseable"] == 0


def test_all_garbage_delimited_cell_becomes_na_and_message_reports_unparseable() -> None:
    """Mirrors 'all-garbage delimited cell becomes NA and message reports 1 unparseable'."""
    with structlog.testing.capture_logs() as logs:
        result = collapse_multi_date_restructuring(["not-a-date,also-not-a-date"])
    assert result == [None]
    collapse_logs = [e for e in logs if e.get("event") == "collapse_multi_date_restructuring"]
    assert collapse_logs[0]["n_unparseable"] == 1


def test_partial_garbage_delimited_cell_keeps_parseable_token() -> None:
    """Mirrors 'partial-garbage delimited cell keeps the parseable token and no unparseable count'."""
    with structlog.testing.capture_logs() as logs:
        result = collapse_multi_date_restructuring(["2020-01-01,not-a-date"])
    assert result == ["2020-01-01"]
    collapse_logs = [e for e in logs if e.get("event") == "collapse_multi_date_restructuring"]
    assert collapse_logs[0]["n_collapsed"] == 1
    assert collapse_logs[0]["n_unparseable"] == 0


def test_empty_tokens_are_dropped_by_nzchar_filter() -> None:
    """Mirrors 'empty tokens are dropped by the nzchar filter'."""
    assert collapse_multi_date_restructuring(["2020-01-01,,2021-01-01"]) == ["2021-01-01"]


def test_non_character_input_is_coerced_to_character() -> None:
    """Mirrors 'non-character input is coerced to character'.

    R's input was `as.factor(...)`. In Python the closest analogue is any
    non-string scalar that has a meaningful str() — here, an enum-like
    object is overkill; we just test str-coercion of a non-str input.
    """
    class _StrRepr:
        def __str__(self) -> str:
            return "2020-01-01,2021-01-01"

    result = collapse_multi_date_restructuring([_StrRepr()])
    assert result == ["2021-01-01"]
