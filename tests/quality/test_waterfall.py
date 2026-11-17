"""The reconciliation waterfall, the dropped-loans list, G8 and A17."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from esma_milan import runner
from esma_milan.errors import PipelineDataError
from esma_milan.quality import assess_files, complete_report
from esma_milan.quality.assess import QualityReport
from esma_milan.quality.catalogue import RULES
from esma_milan.quality.waterfall import CAUSES, DROPPED_SCHEMA, build_waterfall
from esma_milan.runner import PipelineResult, run_pipeline
from tests.quality.builders import Pair
from tests.quality.test_report import FIXTURES, STARTED, summary_dict

# The pipeline's own warnings about the pools built here (partial coverage, a
# missing source column) are what those pools are for.
pytestmark = pytest.mark.filterwarnings("ignore:select_calc_loan_id", "ignore:MILAN mapping")

LOANS = [
    ("L01", "PERF", "100"), ("L02", "PERF", "0"), ("L03", "PERF", "-5"), ("L04", "PERF", ""),
    ("L05", " PERF", "100"), ("L06", "RDMD", "0"), ("L07", "", "100"), ("L08", "PERF", "100"),
    ("L09", "PERF", "100"), ("L10", "PERF", "100"), ("L11", "PERF", "100"), ("L12", "PERF", "100"),
    ("L13", "ARRE", "100"), ("L14", "PERF", "100"),
]
COLLATERAL: list[tuple[str, dict[str, str]]] = [
    ("L01", {}), ("L09", {"RREC5": "GUAR"}), ("L10", {"RREC9": "ND5"}),
    ("L11", {"RREC13": "0", "RREC17": "0"}), ("L12", {"RREC5": "GUAR"}), ("L12", {"RREC9": ""}),
    ("L13", {}), ("L14", {"RREC5": ""}), ("L06", {}),
]


def constructed_pool() -> Pair:
    """14 loans and 9 collateral records that lose a loan at every line of
    the waterfall a completed run can show: 14 = 2 + 4 + 6 + 2."""
    pair = Pair()
    loan, record = pair.loans[0], pair.collateral[0]
    pair.loans = [
        {**loan, "RREL2": i, "RREL3": i, "RREL4": "B" + i, "RREL5": "B" + i, "RREL69": status, "RREL30": balance}
        for i, status, balance in LOANS
    ]
    pair.collateral = [
        {**record, "RREC2": i, "RREC3": f"P{n}", "RREC4": f"P{n}", **changes}
        for n, (i, changes) in enumerate(COLLATERAL)
    ]
    return pair


def _report(pair: Pair, folder: Path, *, min_coverage: float | None = None) -> QualityReport:
    loans, collaterals = pair.write(folder)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    try:
        result = run_pipeline(
            loans_file_path=loans, collaterals_file_path=collaterals, deal_name="T",
            dry_run=True, verbose=False, min_coverage=min_coverage,
        )
    except Exception as exc:
        return complete_report(assessment, pipeline=None, error=exc)
    return complete_report(assessment, pipeline=result, error=None)


def _lines(report: QualityReport) -> dict[int, dict[str, Any]]:
    return {line["line"]: line for line in summary_dict(report)["waterfall"]["lines"]}


def _parts(line: dict[str, Any]) -> list[Any]:
    return [part["records"] for part in line["parts"]]


def _a17(report: QualityReport) -> list[dict[str, object]]:
    return report.assessment.results.filter(pl.col("rule_id") == "A17").to_dicts()


# -- the lines ----------------------------------------------------------------


def test_the_constructed_pool_line_by_line(tmp_path: Path) -> None:
    report = _report(constructed_pool(), tmp_path, min_coverage=0.0)
    lines = _lines(report)  # the sums are asserted for every summary a test builds
    assert [lines[n]["records"] for n in range(1, 8)] == [14, 2, 12, 4, 8, 6, 2]
    assert {lines[n]["basis"] for n in range(1, 8)} == {"observed"}
    assert _parts(lines[2]) == [1, 1]  # eligibility unknown; inactive
    assert _parts(lines[4]) == [1, 1, 1, 0, 1]  # zero, negative, missing, unparseable, " PERF"
    assert _parts(lines[6]) == [1, 1, 1, 2, 1, 0, 0, 0]
    assert [part["label"] for part in lines[6]["parts"]] == [label for _, label in CAUSES]
    assert "the R reference keeps it" in lines[6]["parts"][2]["label"]
    assert lines[5]["reconciliation_with_stage2"] == {"audit": 8, "stage2": 8, "state": "agrees"}
    assert lines[7]["reconciliation_with_line_5_less_line_6"] == {
        "line_5_less_line_6": 2, "output": 2, "state": "agrees",
    }
    # Balances as the loader reads them, per currency, with the records that have none apart.
    assert lines[1]["balance_by_currency"] == {"EUR": {"balance": 995.0, "records": 13}}
    assert lines[1]["records_without_a_balance"] == 1
    assert lines[6]["balance_by_currency"] == {"EUR": {"balance": 600.0, "records": 6}}
    assert lines[7]["balance_by_currency"] == {"EUR": {"balance": 200.0, "records": 2}}
    assert report.waterfall is not None
    assert report.waterfall.collateral_counts == {
        "raw collateral records": 9,
        "distinct collateral identifiers (RREC4, as written)": 9,
        "property rows retained by Stage 2": 2,
        "property rows linked to a retained loan (Stage 3)": 2,
        "graph groups": 2,
    }


def test_the_dropped_loans_list_of_the_constructed_pool(tmp_path: Path) -> None:
    report = _report(constructed_pool(), tmp_path, min_coverage=0.0)
    assert report.waterfall is not None
    dropped = report.waterfall.dropped
    assert dropped.schema == DROPPED_SCHEMA
    causes = dict(CAUSES)
    assert [
        (r["record"], r["waterfall_step"], r["assigned_line"], r["collateral_records"])
        for r in dropped.to_dicts()
    ] == [
        (2, 4, "zero balance", ""), (3, 4, "negative balance", ""), (4, 4, "balance missing", ""),
        (5, 4, "active status not recognised", ""),
        (8, 6, causes["no_collateral_record"], ""),
        (9, 6, causes["only_excluded_types"], "2"),
        (10, 6, causes["property_type_missing"], "3"),
        (11, 6, causes["no_positive_valuation"], "4"),
        (12, 6, causes["property_type_missing"], "5;6"),
        (14, 6, causes["collateral_type_missing"], "8"),
    ]
    by_record = {r["record"]: r for r in dropped.to_dicts()}
    # The two-cause loan: assigned by the priority, with the other cause shown.
    assert by_record[12]["other_causes"] == "a collateral record of an excluded type"
    assert by_record[9]["other_causes"] == ""
    assert {r["observed_or_predicted"] for r in dropped.to_dicts()} == {"observed"}
    assert set(dropped["collateral_records_kept_by_stage2"].to_list()) == {""}
    # Identifiers and the balance are the text of the file; a missing balance is empty.
    assert (by_record[3]["loan_identifier_new"], by_record[3]["balance"], by_record[3]["currency"]) == (
        "L03", "-5", "EUR",
    )
    assert by_record[4]["balance"] is None and by_record[5]["status"] == " PERF"
    # Lost at line 2 (inactive, eligibility unknown) is not a dropped loan.
    assert not {6, 7} & set(by_record)
    assert summary_dict(report)["dropped_loans"]["loans"] == 10


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("synthetic", [8, 0, 8, 0, 8, 0, 8]), ("real_anonymised", [1295, 169, 1126, 0, 1126, 0, 1126])],
)
def test_the_fixtures(fixture: str, expected: list[int]) -> None:
    folder = FIXTURES / fixture
    assessment = assess_files(folder / "loans.csv", folder / "collaterals.csv", started_at=STARTED)
    result = run_pipeline(
        loans_file_path=folder / "loans.csv", collaterals_file_path=folder / "collaterals.csv",
        deal_name="T", dry_run=True, verbose=False,
    )
    report = complete_report(assessment, pipeline=result, error=None)
    lines = _lines(report)
    assert [lines[n]["records"] for n in range(1, 8)] == expected
    assert {lines[n]["basis"] for n in range(1, 8)} == {"observed"}
    assert report.waterfall is not None and report.waterfall.dropped.height == 0
    assert report.stage3_comparison is not None and report.stage3_comparison.agrees
    g8 = next(o for o in report.waterfall.outcomes if o.rule_id == "G8")
    assert (g8.applicable, g8.evaluated, g8.failed, g8.not_evaluable) == (2, 2, 0, 0)


def test_an_unparseable_balance_is_always_predicted(tmp_path: Path) -> None:
    report = _report(Pair().set("RREL30", "abc", record=1), tmp_path)  # the loader stops at it
    assert report.waterfall is not None
    (row,) = report.waterfall.dropped.to_dicts()
    assert (row["record"], row["assigned_line"], row["observed_or_predicted"], row["balance"]) == (
        1, "balance unparseable", "predicted", "abc",
    )


# -- parts 6 and 7 of the priority, on constructed decisions -------------------


def _frames(pair: Pair, folder: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    loans, collaterals = pair.write(folder)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    loan_records, collateral = assessment.records("loans"), assessment.records("collateral")
    assert loan_records is not None and collateral is not None and assessment.links is not None
    return loan_records, collateral, assessment.links


def test_parts_6_and_7_of_the_priority_on_constructed_decisions(tmp_path: Path) -> None:
    # Four retained loans. Stage 3 is taken to have chosen the original
    # identifier (RREL2). Loan 2 has none; loan 3's collateral record refers
    # to it by its new identifier only; loan 4 has no collateral record.
    pair = Pair()
    loan, record = pair.loans[0], pair.collateral[0]
    pair.loans = [
        {**loan, "RREL2": "A1", "RREL3": "N1"}, {**loan, "RREL2": "", "RREL3": "N2"},
        {**loan, "RREL2": "A3", "RREL3": "N3"}, {**loan, "RREL2": "A4", "RREL3": "N4"},
    ]
    pair.collateral = [
        {**record, "RREC2": "A1", "RREC4": "P1"}, {**record, "RREC2": "N2", "RREC4": "P2"},
        {**record, "RREC2": "N3", "RREC4": "P3"},
    ]
    loans, collateral, links = _frames(pair, tmp_path)
    positions = pl.Series([0, 1, 2, 3], dtype=pl.UInt32)
    decisions = SimpleNamespace(
        loan_id=SimpleNamespace(chosen_column="original_underlying_exposure_identifier"),
        loans_missing_calc_loan_id=pl.Series([1], dtype=pl.UInt32),
        loans_without_collateral=pl.Series([2, 3], dtype=pl.UInt32),
    )
    result: Any = SimpleNamespace(
        stage2=SimpleNamespace(
            loan_rows=positions, property_rows=pl.Series([0, 1, 2], dtype=pl.UInt32),
            property_drop_reasons=None, properties=collateral,
        ),
        stage3=SimpleNamespace(decisions=decisions, properties=collateral.head(1)),
        stage4=None, stage7=None, loan_record_index=None, collateral_record_index=None,
    )
    waterfall = build_waterfall(loans, collateral, links, result, completed=False)
    line6 = waterfall.lines[5]
    assert line6["basis"] == "observed" and line6["selected_loan_identifier"] == "RREL2"
    parts = line6["parts"]
    assert isinstance(parts, list)
    assert [part["records"] for part in parts] == [1, 0, 0, 0, 0, 1, 1, None]
    causes = dict(CAUSES)
    assert [(r["record"], r["assigned_line"]) for r in waterfall.dropped.to_dicts()] == [
        (2, causes["selected_identifier_missing"]),
        (3, causes["no_match_under_selected_convention"]),
        (4, causes["no_collateral_record"]),
    ]
    # A loan whose identifier is missing also has no match under the convention: shown as another cause.
    assert waterfall.dropped["other_causes"].to_list() == [
        causes["no_match_under_selected_convention"], "", "",
    ]
    assert waterfall.lines[6]["basis"] == "not available: the pipeline did not complete"


def test_a_dropped_loan_the_links_cannot_explain_is_reconciliation_incomplete(tmp_path: Path) -> None:
    loans, collateral, links = _frames(Pair(), tmp_path)
    result: Any = SimpleNamespace(
        stage2=SimpleNamespace(
            loan_rows=pl.Series([0, 1], dtype=pl.UInt32), property_rows=pl.Series([0, 1], dtype=pl.UInt32),
            property_drop_reasons=None, properties=collateral,
        ),
        stage3=SimpleNamespace(
            decisions=SimpleNamespace(
                loan_id=SimpleNamespace(chosen_column="new_underlying_exposure_identifier"),
                loans_missing_calc_loan_id=pl.Series([], dtype=pl.UInt32),
                loans_without_collateral=pl.Series([0], dtype=pl.UInt32),  # yet its record was kept
            ),
            properties=collateral,
        ),
        stage4=None, stage7=None, loan_record_index=None, collateral_record_index=None,
    )
    waterfall = build_waterfall(loans, collateral, links, result, completed=False)
    parts = waterfall.lines[5]["parts"]
    assert isinstance(parts, list)
    assert parts[-1]["label"] == "cause not established (reconciliation incomplete)"
    assert parts[-1]["records"] == 1
    assert waterfall.dropped["collateral_records_kept_by_stage2"].to_list() == ["1"]


# -- a failed run, with and without the stages that completed ------------------


def _failing(monkeypatch: pytest.MonkeyPatch, stage: int) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise PipelineDataError("m", stage="stage3", reason="no_data_rows")

    monkeypatch.setattr(runner, f"run_stage{stage}", fail)


@pytest.mark.parametrize(
    ("failing", "bases"),
    [
        # Stage 1 failed: nothing was observed but the files.
        (1, ["observed", "predicted", "predicted", "predicted", "predicted", "predicted", "n/a"]),
        # Stage 3 failed: Stage 2's own result is there, Stage 3's choice is not.
        (3, ["observed", "observed", "observed", "observed", "observed", "predicted", "n/a"]),
        # Stage 4 failed: every loan Stage 3 dropped is named.
        (4, ["observed", "observed", "observed", "observed", "observed", "observed", "n/a"]),
    ],
)
def test_a_failed_run_labels_each_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: int, bases: list[str]
) -> None:
    _failing(monkeypatch, failing)
    report = _report(constructed_pool(), tmp_path, min_coverage=0.0)
    assert not report.pipeline.completed
    lines = _lines(report)
    seen = [
        "n/a" if str(lines[n]["basis"]).startswith("not available")
        else str(lines[n]["basis"]).split(":")[0]
        for n in range(1, 8)
    ]
    assert seen == bases
    assert [lines[n]["records"] for n in range(1, 6)] == [14, 2, 12, 4, 8]
    assert lines[7]["records"] is None
    assert lines[7]["basis"] == "not available: the pipeline did not complete"
    parts = lines[6]["parts"]
    if failing == 4:
        assert _parts(lines[6]) == [1, 1, 1, 2, 1, 0, 0, None]
    else:
        # The causes that need no selection, and nothing that needs Stage 3's choice.
        assert _parts(lines[6]) == [1, 1, 1, 2, 1, None, None, None]
        assert {p["basis"] for p in parts[5:7]} == {"not available: Stage 3 did not complete"}
        assert lines[6]["basis"].startswith("predicted: the loans that would be lost under either")
    assert lines[6]["collateral_reasons"] == ("predicted" if failing == 1 else "observed")
    assert parts[7]["basis"] == "not available: the pipeline did not complete"
    assert report.waterfall is not None
    lost = report.waterfall.dropped.filter(pl.col("waterfall_step") == 6)
    assert set(lost["observed_or_predicted"].to_list()) == {"observed" if failing == 4 else "predicted"}
    step4 = report.waterfall.dropped.filter(pl.col("waterfall_step") == 4)
    assert set(step4["observed_or_predicted"].to_list()) == {"predicted" if failing == 1 else "observed"}
    g8 = next(o for o in report.waterfall.outcomes if o.rule_id == "G8")
    assert (g8.evaluated, g8.failed, g8.not_evaluable) == ((0, 0, 2) if failing == 1 else (1, 0, 1))


def test_a_failure_without_a_partial_result_predicts_everything(tmp_path: Path) -> None:
    loans, collaterals = constructed_pool().write(tmp_path)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    report = complete_report(assessment, pipeline=None, error=RuntimeError("no stages on this one"))
    lines = _lines(report)
    assert [lines[n]["basis"].split(":")[0] for n in range(2, 7)] == ["predicted"] * 5
    assert [lines[n]["records"] for n in range(1, 7)] == [14, 2, 12, 4, 8, 6]
    assert lines[5]["reconciliation_with_stage2"]["state"].startswith("not evaluated")


# -- G8 -----------------------------------------------------------------------


def _completed(pair: Pair, folder: Path) -> tuple[Any, PipelineResult]:
    loans, collaterals = pair.write(folder)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    result = run_pipeline(
        loans_file_path=loans, collaterals_file_path=collaterals, deal_name="T",
        dry_run=True, verbose=False, min_coverage=0.0,
    )
    return assessment, result


def test_g8_a_difference_at_line_5_is_reconciliation_incomplete_with_both_figures(tmp_path: Path) -> None:
    assessment, result = _completed(constructed_pool(), tmp_path)
    assert result.stage2 is not None and result.stage2.loan_rows is not None
    # The stage is made to say it kept record 2 (zero balance) and not record 1.
    rows = pl.Series([1, *result.stage2.loan_rows.to_list()[1:]], dtype=pl.UInt32)
    tampered = replace(result, stage2=replace(result.stage2, loan_rows=rows))
    report = complete_report(assessment, pipeline=tampered, error=None)
    lines = _lines(report)
    step5 = lines[5]["reconciliation_with_stage2"]
    assert (step5["audit"], step5["stage2"], step5["state"]) == (8, 8, "reconciliation incomplete")
    assert (step5["records_that_differ"], step5["first_differing_records"]) == (2, [1, 2])
    assert lines[5]["basis"] == "predicted"  # the labels are not confirmed by the stage
    found = report.assessment.results.filter(pl.col("rule_id") == "G8").to_dicts()
    assert [(r["file"], r["result_kind"], r["reason"]) for r in found] == [
        ("pool", "limitation", "line_5_reconciliation_incomplete"),
        ("pool", "limitation", "line_7_reconciliation_incomplete"),
    ]
    assert "the audit retains 8 loan record(s) and Stage 2 8, and 2 record(s) differ" in str(
        found[0]["expected"]
    )
    assert "not a finding about the data" in str(found[0]["expected"])
    finding = next(f for f in summary_dict(report)["findings"] if f["rule_id"] == "G8")
    assert finding["sentence"].startswith("the pool: could not be assessed in full: two figures of the waterfall")
    assert not finding["enters_compliance_judgment"]


def test_g8_a_difference_at_line_7(tmp_path: Path) -> None:
    assessment, result = _completed(constructed_pool(), tmp_path)
    assert result.loan_record_index is not None
    tampered = replace(result, loan_record_index=result.loan_record_index.head(1))
    report = complete_report(assessment, pipeline=tampered, error=None)
    step7 = _lines(report)[7]["reconciliation_with_line_5_less_line_6"]
    assert step7 == {"line_5_less_line_6": 2, "output": 2, "state": "reconciliation incomplete"}
    assert RULES["G8"].status == "implemented" and RULES["G8"].result_kinds == ("limitation",)


# -- A17 ----------------------------------------------------------------------


def test_a17_nothing_on_a_pool_in_one_currency(tmp_path: Path) -> None:
    report = _report(Pair(), tmp_path)
    assert _a17(report) == []
    outcomes = [o for o in report.assessment.outcomes if o.rule_id == "A17"]
    assert [(o.field, o.applicable, o.evaluated, o.failed, o.not_evaluable) for o in outcomes] == [
        ("RREL30", 2, 2, 0, 0), ("RREL30", 2, 2, 0, 0), ("RREL16", 2, 2, 0, 0), ("RREL29", 2, 2, 0, 0),
    ]


def test_a17_pool_balances_in_two_currencies_and_an_invalid_code(tmp_path: Path) -> None:
    pair = Pair()
    pair.loans.append({**pair.loans[0], "RREL2": "LOAN003", "RREL3": "LOAN003"})
    pair.collateral.append({**pair.collateral[0], "RREC2": "LOAN003", "RREC4": "PROPERTY003"})
    for code in ("RREL30", "RREL29", "RREL31", "RREL32", "RREL33"):
        pair.set(f"{code}_CURRENCY", "USD", record=2)
        pair.set(f"{code}_CURRENCY", "ZZZ", record=3)
    pair.set("RREL28", "ND5", record=2).set("RREL28", "ND5", record=3)
    for code in ("RREC13", "RREC17"):
        pair.collateral[1][f"{code}_CURRENCY"] = "USD"
        pair.collateral[2][f"{code}_CURRENCY"] = "ZZZ"
    pair.collateral[1]["RREC22"] = "USD"
    pair.collateral[2]["RREC22"] = "ND5"
    report = _report(pair, tmp_path)
    assert report.pipeline.completed  # the pool is processed; this is a warning
    rows = _a17(report)
    assert [(r["file"], r["record"], r["field"], r["reason"], r["note"]) for r in rows] == [
        ("pool", None, None, "pool_balances_in_more_than_one_currency", None),
        ("pool", None, None, "selected_valuations_in_more_than_one_currency", None),
        ("loans", 2, "RREL30", "balance_currency_differs_from_the_pool", "currency USD"),
        ("loans", 3, "RREL30", "balance_currency_not_a_valid_code", "currency ZZZ"),
    ]
    assert "EUR: 1 record(s), USD: 1 record(s), ZZZ: 1 record(s)" in str(rows[0]["expected"])
    assert {r["severity"] for r in rows} == {"Warn"} and {r["result_kind"] for r in rows} == {"violation"}


def test_a17_amounts_combined_on_one_loan(tmp_path: Path) -> None:
    pair = Pair().set("RREL31_CURRENCY", "USD", record=1).set("RREL20_CURRENCY", "GBP", record=2)
    rows = _a17(_report(pair, tmp_path))
    assert [(r["record"], r["field"], r["reason"], r["note"]) for r in rows] == [
        (1, "RREL30", "ranking_amounts_in_more_than_one_currency", "currencies EUR, USD"),
        (2, "RREL16", "total_income_in_more_than_one_currency", "currencies EUR, GBP"),
    ]
    assert "Total Income" in str(rows[1]["expected"])


def test_a17_the_valuation_selected_against_the_loan(tmp_path: Path) -> None:
    pair = Pair()
    for code in ("RREC13", "RREC17"):  # whichever valuation Stage 7 selects
        pair.collateral[0][f"{code}_CURRENCY"] = "USD"
    pair.collateral[0]["RREC22"] = "USD"
    report = _report(pair, tmp_path)
    rows = _a17(report)
    assert [(r["file"], r["record"], r["field"], r["reason"], r["note"]) for r in rows] == [
        ("pool", None, None, "selected_valuations_in_more_than_one_currency", None),
        ("loans", 1, "RREL29", "selected_valuations_in_more_than_one_currency", "currencies EUR, USD"),
    ]
    assert "EUR: 1 record(s), USD: 1 record(s)" in str(rows[0]["expected"])


def test_a17_the_fourth_set_needs_stage_7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = Pair()
    for code in ("RREC13", "RREC17"):
        pair.collateral[0][f"{code}_CURRENCY"] = "USD"
    pair.collateral[0]["RREC22"] = "USD"
    _failing(monkeypatch, 7)
    report = _report(pair, tmp_path)
    assert _a17(report) == []
    fourth = [o for o in report.assessment.outcomes if o.rule_id == "A17"][-1]
    assert (fourth.field, fourth.applicable, fourth.evaluated, fourth.not_evaluable) == ("RREL29", 2, 0, 2)
    assert dict(fourth.not_evaluable_reasons) == {"Stage 7 did not complete": 2}
    summary = summary_dict(report)
    completeness = summary["assessment_completeness"]["rules_not_evaluable_on_these_files"]
    assert completeness["A17"] == {"records": 2, "reasons": {"Stage 7 did not complete": 2}}


def test_a17_a_currency_that_is_not_stated_is_counted_apart(tmp_path: Path) -> None:
    pair = Pair().drop("RREL30_CURRENCY").set("RREL28", "ND5")
    report = _report(pair, tmp_path)
    assert _a17(report) == []
    first = next(o for o in report.assessment.outcomes if o.rule_id == "A17")
    assert (first.applicable, first.evaluated, first.not_evaluable) == (2, 0, 2)
    assert dict(first.not_evaluable_reasons) == {"the currency of an amount is not stated": 2}
    lines = _lines(report)
    assert lines[1]["balance_by_currency"] == {"currency not stated": {"balance": 300003.0, "records": 2}}
