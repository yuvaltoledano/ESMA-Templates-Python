"""Tests for Stage-9 / B-4 INFER + remaining CALC field helpers.

Mirrors r_reference/R/milan_mapping.R:670-675 (Flexible Loan Amount),
:732-739 (Social Programme Type), :751-756 (Borrower Type), :759-763
(Borrower Residency), :803-808 (Total Income), :859-863 (Recourse),
:978-983 (Restructured Loan), :1015-1021 (MIG Provider).

Pinning strategy: branches exercised by the synthetic fixture are
pinned against fixture-extracted values. Branches NOT exercised by
the fixture are tested against the R source contract; those tests
are explicitly marked "branch not exercised by synthetic fixture" so
the coverage gap is visible. Layer-2 parity (real_anonymised CI
nightly) is the broader-coverage backstop.

Fixture-pinned values:
  Social Programme Type: "6"   (only branch the fixture hits)
  Borrower Type:         "1"   (only branch the fixture hits)
  Borrower Residency:    "Y"   (only branch the fixture hits)
  Recourse:              "Y"   (only branch the fixture hits)
  Restructured Loan:     "N"   (only branch the fixture hits)
  MIG Provider:          "NHG / Waarborgfonds Eigen Woningen", "No Guarantor"
  Total Income:          integer-valued floats formatted as plain integers
                         (52000, 100000, 45000, 38000, 83000, 55000,
                          71000, 48000)
  Flexible Loan Amount:  integer-valued floats (40000, 60000, 35000,
                         30000, 45000, 25000)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl

from esma_milan.pipeline.milan_map import (
    _milan_borrower_residency_expr,
    _milan_borrower_type_expr,
    _milan_flexible_loan_amount_expr,
    _milan_mig_provider_expr,
    _milan_recourse_expr,
    _milan_restructured_loan_expr,
    _milan_social_programme_expr,
    _milan_total_income_expr,
)


def _apply1(
    expr_factory: Callable[[pl.Expr], pl.Expr],
    values: Sequence[str | None],
) -> list[str]:
    df = pl.DataFrame({"x": list(values)}, schema={"x": pl.Utf8})
    return df.select(expr_factory(pl.col("x")).alias("out"))["out"].to_list()


# ---------------------------------------------------------------------------
# Social Programme Type
# ---------------------------------------------------------------------------


def test_social_programme_nd_branch_pinned_to_fixture() -> None:
    """ND special_scheme -> "6". Fixture has all 8 rows hitting this branch."""
    out = _apply1(_milan_social_programme_expr, ["ND", "ND3", "", "  ", None])
    assert out == ["6"] * 5


def test_social_programme_rtb_branch() -> None:
    """RTB | Right to Buy (case-insensitive) -> "1". Branch not exercised by synthetic fixture."""
    out = _apply1(
        _milan_social_programme_expr,
        ["RTB scheme", "right to buy programme", "Right to Buy", "RTB"],
    )
    assert out == ["1", "1", "1", "1"]


def test_social_programme_tenant_purchase_branch() -> None:
    """Tenant Purchase -> "2". Branch not exercised by synthetic fixture."""
    out = _apply1(
        _milan_social_programme_expr,
        ["Tenant Purchase", "TENANT PURCHASE Scheme", "tenant purchase 2024"],
    )
    assert out == ["2", "2", "2"]


def test_social_programme_htb_branch() -> None:
    """HTB | Help to Buy | Equity Loan -> "3". Branch not exercised by synthetic fixture."""
    out = _apply1(
        _milan_social_programme_expr,
        ["HTB", "Help to Buy", "Equity Loan", "help to buy"],
    )
    assert out == ["3", "3", "3", "3"]


def test_social_programme_shared_ownership_branch() -> None:
    """Shared Ownership | Part Buy Part Rent -> "4". Branch not exercised by synthetic fixture."""
    out = _apply1(
        _milan_social_programme_expr,
        ["Shared Ownership", "Part Buy Part Rent", "shared ownership 2024"],
    )
    assert out == ["4", "4", "4"]


def test_social_programme_else_branch_unknown_value() -> None:
    """Non-matching, non-ND -> "5". Branch not exercised by synthetic fixture."""
    out = _apply1(
        _milan_social_programme_expr,
        ["Unknown Scheme", "Random Text", "Foo Bar"],
    )
    assert out == ["5", "5", "5"]


def test_social_programme_short_circuit_first_match_wins() -> None:
    """case_when short-circuits at first match. "RTB Tenant Purchase" -> "1", not "2"."""
    out = _apply1(_milan_social_programme_expr, ["RTB and Tenant Purchase"])
    assert out == ["1"]


# ---------------------------------------------------------------------------
# Borrower Type
# ---------------------------------------------------------------------------


def _apply_borrower_type(
    emp: Sequence[str | None],
    inc: Sequence[str | None],
) -> list[str]:
    df = pl.DataFrame(
        {"e": list(emp), "i": list(inc)},
        schema={"e": pl.Utf8, "i": pl.Utf8},
    )
    return df.select(
        _milan_borrower_type_expr(pl.col("e"), pl.col("i")).alias("out")
    )["out"].to_list()


def test_borrower_type_branch_1_pinned_to_fixture() -> None:
    """Employed (non-NOEM) + non-CORP income -> "1". All 8 fixture rows hit this."""
    out = _apply_borrower_type(["EMRS", "SFEM", "EMUK"], ["", None, "ND"])
    assert out == ["1", "1", "1"]


def test_borrower_type_branch_2_noem_or_corp() -> None:
    """employment=NOEM OR primary_income_type=CORP -> "2". Not exercised by synthetic fixture."""
    out = _apply_borrower_type(
        ["NOEM", "EMRS", "ND"],
        ["IFXX", "CORP", "CORP"],
    )
    assert out == ["2", "2", "2"]


def test_borrower_type_branch_2_noem_short_circuits_corp() -> None:
    """NOEM employment + non-CORP income still -> "2" (OR semantic)."""
    out = _apply_borrower_type(["NOEM"], ["IFXX"])
    assert out == ["2"]


def test_borrower_type_fallback_when_both_nd() -> None:
    """Both employment and income ND/null -> "ND" fallback."""
    out = _apply_borrower_type(
        [None, "ND", ""],
        [None, "ND", ""],
    )
    assert out == ["ND", "ND", "ND"]


def test_borrower_type_corp_with_null_employment_still_branch_2() -> None:
    """primary_income=CORP overrides null employment via OR -> "2"."""
    out = _apply_borrower_type([None, "ND"], ["CORP", "CORP"])
    assert out == ["2", "2"]


# ---------------------------------------------------------------------------
# Borrower Residency
# ---------------------------------------------------------------------------


def test_borrower_residency_branch_y_pinned_to_fixture() -> None:
    """resident="TRUE" -> "Y". All 8 fixture rows hit this."""
    out = _apply1(_milan_borrower_residency_expr, ["TRUE"])
    assert out == ["Y"]


def test_borrower_residency_branch_n_not_in_fixture() -> None:
    """resident="FALSE" -> "N". Branch not exercised by synthetic fixture."""
    out = _apply1(_milan_borrower_residency_expr, ["FALSE"])
    assert out == ["N"]


def test_borrower_residency_case_sensitive_lowercase_falls_to_nd() -> None:
    """R uses plain `==` on character (case-sensitive). "True"/"true" fall to "ND"."""
    out = _apply1(
        _milan_borrower_residency_expr,
        ["True", "true", "Y", "anything", "ND", None, ""],
    )
    assert out == ["ND"] * 7


# ---------------------------------------------------------------------------
# Recourse
# ---------------------------------------------------------------------------


def test_recourse_branch_y_pinned_to_fixture() -> None:
    """recourse="Y" -> "Y" pass-through. All 8 fixture rows hit this."""
    out = _apply1(_milan_recourse_expr, ["Y"])
    assert out == ["Y"]


def test_recourse_branch_n_not_in_fixture() -> None:
    """recourse="N" -> "N" pass-through. Branch not exercised by synthetic fixture."""
    out = _apply1(_milan_recourse_expr, ["N"])
    assert out == ["N"]


def test_recourse_unknown_or_nd_emits_nd() -> None:
    """Anything other than Y/N -> "ND"."""
    out = _apply1(
        _milan_recourse_expr,
        ["Yes", "n", "ND", "", None, "anything"],
    )
    assert out == ["ND"] * 6


# ---------------------------------------------------------------------------
# Restructured Loan
# ---------------------------------------------------------------------------


def _apply_restructured(
    date_of_restructuring: Sequence[str | None],
    account_status: Sequence[str | None],
) -> list[str]:
    df = pl.DataFrame(
        {"d": list(date_of_restructuring), "s": list(account_status)},
        schema={"d": pl.Utf8, "s": pl.Utf8},
    )
    return df.select(
        _milan_restructured_loan_expr(pl.col("d"), pl.col("s")).alias("out")
    )["out"].to_list()


def test_restructured_loan_branch_n_pinned_to_fixture() -> None:
    """ND date + non-RNAR/RARR status -> "N". All 8 fixture rows hit this."""
    out = _apply_restructured(["ND", "ND", ""], ["PERF", "DFLT", "ARRE"])
    assert out == ["N", "N", "N"]


def test_restructured_loan_branch_y_via_date_not_in_fixture() -> None:
    """Non-ND date -> "Y" regardless of status. Not exercised by synthetic fixture."""
    out = _apply_restructured(
        ["2024-05-15", "2023-01-01"],
        ["PERF", "ND"],
    )
    assert out == ["Y", "Y"]


def test_restructured_loan_branch_y_via_status_not_in_fixture() -> None:
    """ND date + status in {RNAR, RARR} -> "Y". Not exercised by synthetic fixture."""
    out = _apply_restructured(["ND", "ND"], ["RNAR", "RARR"])
    assert out == ["Y", "Y"]


def test_restructured_loan_branch_nd_when_both_nd_not_in_fixture() -> None:
    """Both date and status ND -> "ND". Not exercised by synthetic fixture."""
    out = _apply_restructured([None, "ND", ""], [None, "ND", ""])
    assert out == ["ND", "ND", "ND"]


# ---------------------------------------------------------------------------
# MIG Provider
# ---------------------------------------------------------------------------


def test_mig_provider_nhgx_pinned_to_fixture() -> None:
    """guarantor_type=NHGX -> exact fixture string."""
    out = _apply1(_milan_mig_provider_expr, ["NHGX"])
    assert out == ["NHG / Waarborgfonds Eigen Woningen"]


def test_mig_provider_no_guarantor_pinned_to_fixture() -> None:
    """ND/null guarantor -> "No Guarantor" (fixture-pinned)."""
    out = _apply1(_milan_mig_provider_expr, [None, "ND", ""])
    assert out == ["No Guarantor", "No Guarantor", "No Guarantor"]


def test_mig_provider_other_branches_not_in_fixture() -> None:
    """FGAS/CATN/OTHR -> their respective strings. Not exercised by synthetic fixture."""
    out = _apply1(_milan_mig_provider_expr, ["FGAS", "CATN", "OTHR"])
    assert out == ["SGFGAS", "Caution", "Other"]


def test_mig_provider_unknown_falls_to_no_guarantor() -> None:
    """Unknown guarantor codes -> "No Guarantor"."""
    out = _apply1(_milan_mig_provider_expr, ["XXXX", "BANK", "INSU"])
    assert out == ["No Guarantor", "No Guarantor", "No Guarantor"]


# ---------------------------------------------------------------------------
# Total Income
# ---------------------------------------------------------------------------


def _apply_total_income(
    primary: Sequence[float | None],
    secondary: Sequence[float | None],
) -> list[str]:
    df = pl.DataFrame(
        {"p": list(primary), "s": list(secondary)},
        schema={"p": pl.Float64, "s": pl.Float64},
    )
    return df.select(
        _milan_total_income_expr(pl.col("p"), pl.col("s")).alias("out")
    )["out"].to_list()


def test_total_income_sum_pinned_to_fixture_strings() -> None:
    """Both incomes present -> sum formatted as integer-string (fixture pattern).

    Fixture has 8 rows where Total Income emits plain-integer strings
    (52000, 100000, 45000, 38000, 83000, 55000, 71000, 48000).
    """
    out = _apply_total_income(
        [30000.0, 60000.0, 25000.0],
        [22000.0, 40000.0, 20000.0],
    )
    assert out == ["52000", "100000", "45000"]


def test_total_income_both_null_emits_nd() -> None:
    """Both null -> "ND"."""
    out = _apply_total_income([None], [None])
    assert out == ["ND"]


def test_total_income_one_null_returns_the_other() -> None:
    """Primary null -> secondary; secondary null -> primary."""
    out = _apply_total_income([None, 38000.0], [38000.0, None])
    assert out == ["38000", "38000"]


def test_total_income_decimal_value_uses_15_sig_digits() -> None:
    """Non-integer sum uses %.15g (matches R's as.character)."""
    out = _apply_total_income([30000.5], [22000.25])
    assert out == ["52000.75"]


# ---------------------------------------------------------------------------
# Flexible Loan Amount
# ---------------------------------------------------------------------------


def _apply_flex(
    tcl: Sequence[float | None],
    cpb: Sequence[float | None],
) -> list[str]:
    df = pl.DataFrame(
        {"t": list(tcl), "c": list(cpb)},
        schema={"t": pl.Float64, "c": pl.Float64},
    )
    return df.select(
        _milan_flexible_loan_amount_expr(pl.col("t"), pl.col("c")).alias("out")
    )["out"].to_list()


def test_flexible_loan_amount_positive_diff_pinned_to_fixture_strings() -> None:
    """tcl > cpb -> diff formatted as integer-string (fixture pattern).

    Fixture has 8 rows where Flexible Loan Amount emits plain-integer
    strings (40000, 60000, 35000, 30000, 45000, 25000).
    """
    out = _apply_flex(
        [100000.0, 200000.0, 80000.0],
        [60000.0, 140000.0, 45000.0],
    )
    assert out == ["40000", "60000", "35000"]


def test_flexible_loan_amount_null_tcl_emits_zero() -> None:
    """tcl null -> "0"."""
    out = _apply_flex([None, None], [50000.0, None])
    assert out == ["0", "0"]


def test_flexible_loan_amount_negative_or_zero_diff_emits_zero() -> None:
    """tcl - cpb <= 0 -> "0"."""
    out = _apply_flex(
        [50000.0, 30000.0, 100000.0],
        [50000.0, 60000.0, 200000.0],
    )
    assert out == ["0", "0", "0"]


def test_flexible_loan_amount_decimal_diff_uses_15_sig_digits() -> None:
    """Non-integer diff uses %.15g."""
    out = _apply_flex([100000.5], [60000.25])
    assert out == ["40000.25"]
