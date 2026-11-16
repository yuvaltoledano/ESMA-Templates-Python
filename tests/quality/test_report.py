"""The report object and the summary document."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from esma_milan.errors import PipelineDataError
from esma_milan.quality import assess_files, complete_report
from esma_milan.quality import summary_dict as _summary_dict
from esma_milan.quality.assess import QualityReport
from esma_milan.quality.catalogue import RESULT_KINDS, RULES, SENTENCES
from esma_milan.quality.export import NOT_IMPLEMENTED, finding_sentence, render_json
from esma_milan.runner import run_pipeline
from tests.quality.builders import Pair

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXPECTED = Path(__file__).resolve().parent / "expected"
STARTED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def summary_dict(report: QualityReport) -> dict[str, Any]:
    return _summary_dict(report)


def _report(pair: Pair, folder: Path) -> QualityReport:
    """Assess the pair, run the pipeline (dry run), complete the report."""
    loans, collaterals = pair.write(folder)
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    try:
        result = run_pipeline(
            loans_file_path=loans, collaterals_file_path=collaterals, deal_name="T",
            dry_run=True, verbose=False,
        )
    except Exception as exc:
        return complete_report(assessment, pipeline=None, error=exc, options={"dry_run": True})
    return complete_report(assessment, pipeline=result, error=None, options={"dry_run": True})


def test_the_compliant_pair(tmp_path: Path) -> None:
    report = _report(Pair(), tmp_path)
    results = report.assessment.results
    # No violation, no question, and nothing of Warn severity: only Info context.
    assert set(results["result_kind"].cast(pl.String).unique().to_list()) == {"context"}
    assert set(results["severity"].cast(pl.String).unique().to_list()) == {"Info"}
    summary = summary_dict(report)
    assert summary["compliance"]["violations"] == {
        "state": "none", "rules": 0, "fields": 0, "records": 0, "column_level_findings": 0,
    }
    assert summary["plausibility"] == {"observations": 0}
    assert summary["pipeline"] == {"completed": True}
    assert report.populations_observed
    assert summary["identity_confidence"] == {
        "selected_loan_identifier": "RREL2", "counts": {"unique": 2},
    }
    assert len(summary["expected_fields"]) == 107
    assert all(row["present"] for row in summary["expected_fields"])


def test_wording_follows_the_result_kind_then_the_basis() -> None:
    for rule in RULES.values():
        for kind in RESULT_KINDS:
            sentence = finding_sentence(rule.rule_id, kind, "RRELn Field")
            assert sentence.startswith("RRELn Field: ") and sentence.endswith(".")
            complies = "does not comply with" in sentence
            assert complies is (kind == "violation" and rule.basis == "E"), (rule.rule_id, kind)
            assert ("contradicts" in sentence) is (kind == "violation" and rule.basis == "X")
            if rule.basis == "P":  # a plausibility result never asserts an error
                assert "does not comply" not in sentence and "contradicts" not in sentence
            if kind == "question":
                assert "confirm" in sentence
            if kind == "limitation":
                assert "could not be assessed" in sentence
            if kind == "violation" and rule.basis == "L":
                assert "the pipeline" in sentence
            # No kind name in brackets, and no rule label where a clause belongs.
            assert not re.search(r"\((context|question|violation|limitation)\)", sentence)
    assert finding_sentence("P8", "question", "X") == (
        "X: possible placeholder; confirm whether this is a measurement."
    )


def test_every_rule_and_kind_that_can_be_emitted_has_its_own_clause() -> None:
    for rule in RULES.values():
        if rule.status == "not_implemented" or rule.rule_id == "P2":
            continue
        for emitted in rule.result_kinds:
            assert (rule.rule_id, emitted) in SENTENCES, (rule.rule_id, emitted)
    for (rule_id, result_kind), clause in SENTENCES.items():
        assert result_kind in RULES[rule_id].result_kinds, (rule_id, result_kind)
        assert clause[0].islower() and not clause.endswith(".")
        assert " (context)" not in clause and "plausibility window)" not in clause


def test_sentences_in_a_summary_read_as_sentences(tmp_path: Path) -> None:
    pair = Pair().set("RREL43", "N/A", record=1).set("RREL30", "", record=1).drop("RREL31_CURRENCY")
    pair.set("RREL40", "0.3").set("RREL81", "NA", record=1)
    sentences = {f["finding_id"]: f["sentence"] for f in summary_dict(_report(pair, tmp_path))["findings"]}
    assert sentences["P3d:violation:loans:RREL43"] == (
        "RREL43 Current Interest Rate: does not comply with the requirement to report a value or a "
        "permitted ND option: the cell holds a missing-value marker that is not one of ND1 to ND5."
    )
    assert sentences["P3c:violation:loans:RREL30"].endswith("the cell is empty.")
    assert sentences["P13:context:loans:RREL31"] == (
        "RREL31 Prior Principal Balances: the file has no currency column for this amount, so the "
        "currency is taken from the other permitted evidence; this is a note about this "
        "application's CSV layout."
    )
    assert sentences["P12:question:loans:RREL40"].startswith("RREL40 Debt To Income Ratio: unusual; please confirm")
    assert sentences["P3g:context:loans:RREL81"] == (
        "RREL81 Original Lender Establishment Country: the pipeline's loader reads the cell "
        "differently from what the file states; the reason on each row says how."
    )
    assert sentences["P9:context:loans:RREL82"] == (
        "RREL82 Originator Name: the column holds one value on every record that reports a value."
    )


def _rule(summary: dict[str, Any], rule_id: str) -> dict[str, Any]:
    found: dict[str, Any] = next(r for r in summary["rules"] if r["rule_id"] == rule_id)
    return found


def test_a_check_with_findings_and_records_it_could_not_evaluate(tmp_path: Path) -> None:
    pair = Pair().set("RREL7", "2030-01-01").set("RREL6", "2024-06-29", record=1)
    summary = summary_dict(_report(pair, tmp_path))
    rule = _rule(summary, "P6(b)")
    assert rule["label"] == "findings and partly evaluated"
    assert rule["failed"] == 1
    # Record 1 has two cut-off dates; RREL7 and RREL23 are its dated fields under the rule.
    assert rule["not_evaluable_reasons"] == {"cut-off not established (two values on the record)": 2}
    assert rule["applicable"] == rule["evaluated"] + rule["not_evaluable"]
    listed = summary["assessment_completeness"]["rules_not_evaluable_on_these_files"]
    assert listed["P6(b)"]["records"] == 2


def test_every_implemented_rule_reports_its_four_counts(tmp_path: Path) -> None:
    summary = summary_dict(_report(Pair(), tmp_path))
    rules = summary["rules"]
    assert isinstance(rules, list)
    assert {r["rule_id"] for r in rules} == {r.rule_id for r in RULES.values() if r.status != "not_implemented"}
    for rule in rules:
        for key in ("applicable", "evaluated", "failed", "not_evaluable", "label"):
            assert key in rule, (rule["rule_id"], key)
        assert rule["applicable"] == rule["evaluated"] + rule["not_evaluable"]
    assert _rule(summary, "P4(c)")["label"].startswith("reported as P5 rows")  # one defect, one row
    assert _rule(summary, "P2")["label"].startswith("reported as the value-state counts")


def test_completeness_is_always_partial_with_the_two_lists_apart(tmp_path: Path) -> None:
    completeness = summary_dict(_report(Pair(), tmp_path))["assessment_completeness"]
    assert isinstance(completeness, dict)
    assert completeness["state"] == "partial"
    not_implemented = completeness["rules_not_implemented_in_this_release"]
    listed = {rule for ids in not_implemented["rules"].values() for rule in ids}
    assert listed == {r.rule_id for r in RULES.values() if r.status == "not_implemented"}
    assert {"A1", "C12", "E17", "F3", "G8", "L1", "P10", "P6(e)"} <= listed
    assert not_implemented["count"] == len(listed)
    assert completeness["rules_not_evaluable_on_these_files"] == {}  # nothing on the compliant pair
    assert set(completeness["rules_partly_implemented"]) == {"P3f", "P11", "P13"}


def test_later_parts_say_not_implemented_and_never_eligible(tmp_path: Path) -> None:
    summary = summary_dict(_report(Pair(), tmp_path))
    for key in ("policy_outcome", "release_permitted", "override", "waterfall", "dropped_loans",
                "lineage", "detail_files"):
        assert summary[key] == NOT_IMPLEMENTED == "not implemented in this release"
    assert "eligible" not in json.dumps(summary).lower()
    assert all(row["pipeline_use_and_lineage"].startswith(NOT_IMPLEMENTED)
               for row in summary["expected_fields"])


def _failing_pair() -> Pair:
    pair = Pair()
    pair.set("RREL30", "abc", record=1)  # the loader stops here
    pair.set("RREL23", "10/05/2019", record=2).set("RREL69", "perf", record=2)
    pair.set("RREC13", "-5", record=1).set("RREC15", "2030-01-01", record=2).set("RREC5", "", record=2)
    return pair


def test_independent_defects_in_both_files_are_all_reported_when_the_pipeline_raises(
    tmp_path: Path,
) -> None:
    report = _report(_failing_pair(), tmp_path)
    assert not report.pipeline.completed
    assert (report.pipeline.error_type, report.pipeline.stage, report.pipeline.reason) == (
        "PipelineDataError", "stage1", "non_numeric_value",
    )
    summary = summary_dict(report)
    findings = summary["findings"]
    assert isinstance(findings, list)
    ids = {f["finding_id"] for f in findings}
    assert {
        "P5:violation:loans:RREL30", "P5:violation:loans:RREL23", "P5:violation:loans:RREL69",
        "P4(d):violation:loans:RREL69", "P6(a):violation:collateral:RREC13",
        "P6(b):violation:collateral:RREC15", "P3c:violation:collateral:RREC5",
    } <= ids
    assert summary["execution"] == "pipeline_failed"
    assert summary["pipeline"]["reason_code"] == "non_numeric_value"
    populations = summary["populations"]
    assert isinstance(populations, dict)
    assert populations["labels"] == "predicted from the Stage 2 predicate"
    assert populations["counts"]["Active status, balance unparseable"] == 1
    assert populations["counts"]["Active status not recognised"] == 1
    assert populations["stage2_comparison"]["state"] == "not evaluated"
    assert summary["artifacts"][0]["state"] == "failed"


def test_a_failed_run_is_reported_as_a_fact_and_never_as_an_outcome(tmp_path: Path) -> None:
    report = _report(_failing_pair(), tmp_path)
    assert isinstance(report.pipeline.error_message, str)
    text = render_json(summary_dict(report)).lower()
    # "accepted alternative" is a result kind, not a statement about the pool.
    text = text.replace("accepted alternative", "")
    for word in ("blocked", "rejected", "eligible", "accepted"):
        assert not re.search(rf"\b{word}\b", text), word
    block = summary_dict(report)["pipeline"]
    assert set(block) == {"completed", "statement", "error_type", "message", "stage", "reason_code"}
    assert block["statement"] == "the pipeline failed; no MILAN workbook was produced by this run"


def test_a_failure_that_is_not_a_typed_data_error_says_so_in_words(tmp_path: Path) -> None:
    # A status column of numbers only: Stage 2 raises a Polars error, with no reason code.
    report = _report(Pair().set("RREL69", "12345"), tmp_path)
    assert not report.pipeline.completed and report.pipeline.reason is None
    block = summary_dict(report)["pipeline"]
    assert block == {
        "completed": False,
        "statement": "the pipeline failed with an error that carries no reason code "
                     f"({report.pipeline.error_type}); no MILAN workbook was produced by this run",
        "error_type": report.pipeline.error_type,
    }
    assert report.pipeline.error_type != "PipelineDataError"
    assert "12345" not in json.dumps(block)  # no message text from the data


def test_a_file_that_cannot_be_split_into_records(tmp_path: Path) -> None:
    pair = Pair().set("RREL30", "-1", record=1)
    loans, collaterals = pair.write(tmp_path)
    collaterals.write_bytes(collaterals.read_bytes() + b"one,two\n" + b"x," * 200 + b"\n")
    assessment = assess_files(loans, collaterals, started_at=STARTED)
    with pytest.raises(pl.exceptions.ComputeError) as raised:
        run_pipeline(loans_file_path=loans, collaterals_file_path=collaterals, deal_name="T",
                     dry_run=True, verbose=False)
    summary = summary_dict(complete_report(assessment, pipeline=None, error=raised.value))
    assert summary["assessment_completeness"]["state"] == "file_unreadable"
    assert summary["assessment_completeness"]["files_unreadable"] == ["collateral"]
    loans_block, collateral_block = summary["files"]
    assert loans_block["readable"] and loans_block["records"] == 2
    assert not collateral_block["readable"]
    assert "records" not in collateral_block and "columns" not in collateral_block  # no invented counts
    unreadable = collateral_block["unreadable"]
    assert unreadable["reason"] == "rows_with_more_fields_than_the_header"
    assert [r["record"] for r in unreadable["long_rows"]] == [4]
    assert unreadable["physical_line"] == 5
    assert summary["actual_columns"]["collateral"] == []
    assert all(row["file"] == "loans" for row in summary["expected_fields"])
    # The loans file is assessed in full all the same.
    assert assessment.loans.results.filter(pl.col("rule_id") == "P5").height == 0
    assert _rule(summary, "P3c")["applicable"] == 2 * 84
    assert summary["pipeline"]["error_type"] == "ComputeError"
    assert "carries no reason code (ComputeError)" in summary["pipeline"]["statement"]
    assert "reason_code" not in summary["pipeline"] and "message" not in summary["pipeline"]


def test_complete_report_needs_exactly_one_of_result_and_error(tmp_path: Path) -> None:
    loans, collaterals = Pair().write(tmp_path)
    assessment = assess_files(loans, collaterals)
    with pytest.raises(ValueError, match="not both"):
        complete_report(assessment, pipeline=None, error=None)
    assert assessment.as_of == assessment.started_at.astimezone(UTC).date()
    error = PipelineDataError("x", stage="stage3", reason="loan_identifier_mismatch")
    assert complete_report(assessment, pipeline=None, error=error).pipeline.reason == "loan_identifier_mismatch"


def test_result_rows_keep_every_record_and_use_categoricals(tmp_path: Path) -> None:
    pair = Pair()
    template = pair.loans[0]
    pair.loans = [{**template, "RREL2": f"L{i}", "RREL3": f"L{i}", "RREL30": "abc"} for i in range(500)]
    loans, collaterals = pair.write(tmp_path)
    results = assess_files(loans, collaterals).loans.results
    rows = results.filter((pl.col("rule_id") == "P5") & (pl.col("field") == "RREL30"))
    assert rows["record"].to_list() == list(range(1, 501))  # nothing sampled
    for column in ("file", "field", "rule_id", "result_kind", "severity", "value_state", "reason", "header"):
        assert results.schema[column] == pl.Categorical, column
    assert results.schema["raw_value"] == pl.String


def test_provenance(tmp_path: Path) -> None:
    summary = summary_dict(_report(Pair(), tmp_path))
    provenance = summary["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["rule_set_version"] == "2026.11.1"
    assert provenance["commit"] is None and "not recorded" in provenance["commit_note"]
    assert provenance["assessment_started_utc"] == "2026-01-02T03:04:05+00:00"
    assert provenance["as_of_date"] == "2026-01-02"
    assert [i["file"] for i in provenance["inputs"]] == ["loans", "collateral"]
    assert all(len(i["sha256"]) == 64 for i in provenance["inputs"])
    assert len(provenance["taxonomy_sha256"]) == 64
    assert provenance["reference_lists"]["iso4217_active"]["version"] == "published 2026-09-17"
    assert provenance["reference_lists"]["nuts3_2024"]["countries"][-2:] == ["UA", "XK"]
    assert provenance["thresholds"]["DQ_DATE_MIN_YEAR"] == 1950
    assert summary["assessment_id"] == provenance["assessment_id"]
    assert len(summary["scope_statements"]) >= 6


def test_nd5_without_a_conditional_rule_is_a_scope_statement_not_a_row(tmp_path: Path) -> None:
    # The compliant pair reports ND5 in many fields where ND5 is allowed and
    # nothing in the files establishes applicability.
    report = _report(Pair(), tmp_path)
    summary = summary_dict(report)
    assert any("justification not verifiable from the files" in s for s in summary["scope_statements"])
    assert any("Links between collateral records and loan records" in s for s in summary["scope_statements"])
    field = next(f for f in summary["expected_fields"] if f["field"] == "RREL8")
    assert field["value_states"]["nd5"] == 2 and field["results"] == {}
    assert report.assessment.results.filter(pl.col("rule_id") == "P3f").height == 0


def test_render_json_is_json() -> None:
    document = {"a": [1, 2, 3], "b": {"long": ["x" * 40] * 10, "n": None, "t": True},
                "c": [{"k": "v" * 200}], "d": [], "e": {}, "f": "é\n\"quoted\""}
    assert json.loads(render_json(document)) == document


def test_the_synthetic_fixtures_summary_is_the_reviewed_one() -> None:
    """The whole summary of the synthetic fixture, as a reviewed file. The
    run is a dry run with a fixed start time, so nothing in it varies."""
    folder = FIXTURES / "synthetic"
    assessment = assess_files(folder / "loans.csv", folder / "collaterals.csv", started_at=STARTED)
    result = run_pipeline(
        loans_file_path=folder / "loans.csv", collaterals_file_path=folder / "collaterals.csv",
        deal_name="SYNTHETIC_FIXTURE", dry_run=True, verbose=False,
    )
    options = {"deal_name": "SYNTHETIC_FIXTURE", "aggregation": "auto", "min_coverage": None, "dry_run": True}
    rendered = render_json(summary_dict(complete_report(assessment, pipeline=result, error=None, options=options)))
    expected = (EXPECTED / "synthetic-summary.json").read_text(encoding="utf-8")
    assert rendered + "\n" == expected
