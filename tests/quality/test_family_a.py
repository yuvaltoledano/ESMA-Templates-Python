"""Family A: the files themselves. Each rule firing and not firing, on
pairs built by test code."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from esma_milan.errors import PipelineDataError
from esma_milan.quality import assess_files
from esma_milan.quality.assess import FileAssessment, complete_report
from esma_milan.quality.catalogue import RULES, SENTENCES
from esma_milan.runner import run_pipeline
from tests.quality.builders import Pair
from tests.quality.test_report import STARTED, summary_dict

Edit = Callable[[Path, Path], Any]
FAMILY_A = sorted(r for r in RULES if r.startswith("A") and RULES[r].status != "not_implemented")


def _assess(pair: Pair, folder: Path, edit: Edit | None = None) -> FileAssessment:
    loans, collaterals = pair.write(folder)
    if edit is not None:
        edit(loans, collaterals)
    return assess_files(loans, collaterals, started_at=STARTED)


def _rows(assessment: FileAssessment, rule_id: str | None = None) -> list[dict[str, object]]:
    """The family A rows, or one rule's, in the order they were emitted."""
    rule = pl.col("rule_id").cast(pl.String)
    wanted = rule.str.starts_with("A") if rule_id is None else rule == rule_id
    return assessment.results.filter(wanted).to_dicts()


def _keys(assessment: FileAssessment, rule_id: str) -> list[tuple[object, ...]]:
    return [(r["file"], r["record"], r["field"], r["reason"]) for r in _rows(assessment, rule_id)]


def _header(path: Path, change: Callable[[str], str]) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[0] = change(lines[0])
    path.write_text("\n".join(lines), encoding="utf-8")


def _label(assessment: FileAssessment, rule_id: str) -> str:
    rules = summary_dict(complete_report(assessment, pipeline=None, error=RuntimeError("x")))["rules"]
    return str(next(r["label"] for r in rules if r["rule_id"] == rule_id))


def test_the_compliant_pair_has_no_family_a_row_and_every_rule_reports(tmp_path: Path) -> None:
    assessment = _assess(Pair(), tmp_path)
    assert _rows(assessment) == []
    labels = {rule_id: _label(assessment, rule_id) for rule_id in FAMILY_A}
    assert labels.pop("A4") == "reported as P1 rows (one defect, one row)"
    assert labels.pop("A14") == "reported as the record counts of the summary's file block"
    # Nothing was repaired and no field is supplied twice: those rules have nothing to apply to.
    assert {r for r, label in labels.items() if label == "not applicable"} == {"A6", "A15(a)", "A15(b)", "A16"}
    assert {label for r, label in labels.items() if r not in ("A6", "A15(a)", "A15(b)", "A16")} == {"passed"}


def test_every_family_a_rule_has_a_clause_and_a_stop_is_worded_by_its_basis() -> None:
    for rule_id in FAMILY_A:
        rule = RULES[rule_id]
        for kind in rule.result_kinds:
            clause = SENTENCES[(rule_id, kind)]
            if rule.severity == "Stop" and rule.basis == "A":
                assert clause.startswith("would stop under the acceptance policy: ")
            if rule.severity == "Stop" and rule.basis == "L" and rule_id not in ("A12", "A13(a)"):
                assert clause.startswith("the pipeline cannot process this: ")
            # A Stop is a label. No clause says this run was stopped.
            for phrase in ("was stopped", "has been stopped", "rejected", "blocked", "withheld"):
                assert phrase not in clause
    assert SENTENCES[("A12", "violation")].startswith("the pipeline halts or mismatches on this today (issue #45)")
    assert SENTENCES[("A13(a)", "violation")].startswith(
        "the pipeline halts or mismatches on this today (issue #46)"
    )
    assert RULES["A12"].severity == "Stop" and RULES["A13(a)"].severity == "Stop"
    assert RULES["A13(b)"].severity == "Warn"


# -- A1, A16 ------------------------------------------------------------------


def test_a1_a_file_without_a_header(tmp_path: Path) -> None:
    assessment = _assess(Pair(), tmp_path, lambda loans, _: loans.write_text("", encoding="utf-8"))
    assert _keys(assessment, "A1(a)") == [("loans", None, None, "no_header")]
    assert _rows(assessment, "A16") == []
    assert assessment.collateral.readable  # the other file is still assessed


def test_a1_a_header_with_no_record(tmp_path: Path) -> None:
    def edit(_: Path, collaterals: Path) -> None:
        collaterals.write_text(collaterals.read_text(encoding="utf-8").split("\n")[0] + "\n", encoding="utf-8")

    assessment = _assess(Pair(), tmp_path, edit)
    assert _keys(assessment, "A1(a)") == [("collateral", None, None, "no_data_records")]
    assert _rows(assessment, "A11") == []  # no identifier in an empty file: nothing to compare


def test_a16_a_record_longer_than_the_header_is_listed(tmp_path: Path) -> None:
    def edit(_: Path, collaterals: Path) -> None:
        text = collaterals.read_text(encoding="utf-8")
        collaterals.write_text(text + "a,b\n" + ",".join(["x"] * 60) + "\n", encoding="utf-8")

    assessment = _assess(Pair(), tmp_path, edit)
    assert _keys(assessment, "A16") == [
        ("collateral", None, None, "rows_with_more_fields_than_the_header"),
        ("collateral", 4, None, "record_longer_than_the_header"),
    ]
    assert _rows(assessment, "A1(a)") == []
    assert {r["severity"] for r in _rows(assessment, "A16")} == {"Stop"}


# -- the duplicate-header policy (5.3): A1(b), A1(c), A15 ----------------------


def test_the_same_header_text_twice_with_differing_values(tmp_path: Path) -> None:
    # RREL40 (2.5) is given RREL41's header (1000.50): the header text is repeated.
    assessment = _assess(
        Pair(), tmp_path, lambda loans, _: _header(loans, lambda h: h.replace("RREL40,", "RREL41,", 1))
    )
    repeated = _rows(assessment, "A1(c)")
    assert [(r["column_position"], r["note"]) for r in repeated] == [
        (49, "also at column 50"), (50, "also at column 49"),
    ]
    assert "RREL41_duplicated_n" not in str(repeated[0]["expected"])  # the loader's cleaned name
    assert "rrel41_duplicated_n" in str(repeated[0]["expected"])
    assert _rows(assessment, "A1(b)") == []  # the loader renames the second column: no collision
    differing = _rows(assessment, "A15(b)")
    # Every differing record, two rows a record, one per column, each naming the other's value.
    assert [(r["record"], r["column_position"], r["raw_value"], r["note"]) for r in differing] == [
        (1, 49, "2.5", "column 50 holds 1000.50"), (2, 49, "2.5", "column 50 holds 1000.50"),
        (1, 50, "1000.50", "column 49 holds 2.5"), (2, 50, "1000.50", "column 49 holds 2.5"),
    ]
    assert {r["severity"] for r in differing} == {"Stop"}
    assert _rows(assessment, "A15(a)") == []


def test_the_same_header_text_twice_with_equal_values(tmp_path: Path) -> None:
    pair = Pair().set("RREL40", "1000.50")
    assessment = _assess(
        pair, tmp_path, lambda loans, _: _header(loans, lambda h: h.replace("RREL40,", "RREL41,", 1))
    )
    assert len(_rows(assessment, "A1(c)")) == 2
    assert _rows(assessment, "A15(b)") == []
    (row,) = _rows(assessment, "A15(a)")
    assert (row["record"], row["field"], row["severity"]) == (None, "RREL41", "Warn")
    assert "columns 49, 50" in str(row["expected"])


@pytest.mark.parametrize("equal", [True, False])
def test_two_different_headers_of_one_field_collide_in_the_pipeline(tmp_path: Path, equal: bool) -> None:
    # RREL39's column is given RREL40's taxonomy name: two headers, one pipeline name.
    pair = Pair().set("RREL39", "2.5") if equal else Pair()

    def edit(loans: Path, _: Path) -> None:
        _header(loans, lambda h: h.replace("RREL39,", "Debt To Income Ratio,", 1))

    assessment = _assess(pair, tmp_path, edit)
    collision = _rows(assessment, "A1(b)")
    assert [(r["column_position"], r["field"], r["severity"]) for r in collision] == [
        (47, "RREL40", "Stop"), (49, "RREL40", "Stop"),
    ]
    assert "debt_to_income_ratio" in str(collision[0]["expected"])
    assert _rows(assessment, "A1(c)") == []
    assert len(_rows(assessment, "A15(a)")) == (1 if equal else 0)
    assert len(_rows(assessment, "A15(b)")) == (0 if equal else 4)
    # The pipeline does raise on this file, with the reason the row names.
    with pytest.raises(PipelineDataError) as caught:
        run_pipeline(
            loans_file_path=tmp_path / "loans.csv", collaterals_file_path=tmp_path / "collaterals.csv",
            deal_name="T", dry_run=True, verbose=False,
        )
    assert caught.value.reason == "duplicate_columns"


# -- A2, A3 --------------------------------------------------------------------


def test_a2_a_required_column_is_absent_and_p1_reports_the_field_too(tmp_path: Path) -> None:
    pair = Pair().drop("RREL30").drop("RREC13").drop("Pool_Cutoff_Date", "loans")
    assessment = _assess(pair, tmp_path)
    assert _keys(assessment, "A2") == [
        ("loans", None, None, "required_column_absent"),  # pool_cutoff_date has no field code
        ("loans", None, "RREL30", "required_column_absent"),
        ("collateral", None, "RREC13", "required_column_absent"),
    ]
    assert {r["severity"] for r in _rows(assessment, "A2")} == {"Stop"}
    p1 = {(r["file"], r["field"]) for r in _rows(assessment, "P1")}
    assert {("loans", "RREL30"), ("collateral", "RREC13")} <= p1
    # One defect, one key: the reporting requirement (P1) and the processing one (A2).
    assert RULES["A2"].defect_key == RULES["P1"].defect_key


def test_a3_one_context_row_per_unmapped_column(tmp_path: Path) -> None:
    pair = Pair().set("Servicer note", "x")
    assessment = _assess(pair, tmp_path)
    rows = _rows(assessment, "A3")
    assert [(r["file"], r["header"], r["result_kind"], r["severity"]) for r in rows] == [
        ("loans", "Servicer note", "context", "Info"), ("collateral", "Servicer note", "context", "Info"),
    ]
    assert "not assessed for compliance" in str(rows[0]["expected"])


# -- A5 against P5 -------------------------------------------------------------


def test_a5_is_the_loaders_test_and_p5_the_format_rule(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL29", "1e3", record=1)  # the loader's cast accepts it; the format does not
    pair.set("RREL29", "abc", record=2)  # neither
    pair.set("RREL16", "ND4-2025-01-01", record=1)  # a valid ND4 the loader cannot read
    pair.set("RREL43", " 2.5 ", record=1)  # trimmed by the loader, as readr does
    for row in pair.loans:  # the audit calls it unmapped in a loans file; the pipeline
        row["lien"] = "first"  # names it a numeric field and casts it
    assessment = _assess(pair, tmp_path)
    a5 = [(r["file"], r["record"], r["field"], r["header"], r["raw_value"]) for r in _rows(assessment, "A5")]
    assert sorted(a5, key=str) == sorted([
        ("loans", 1, "RREL16", "RREL16", "ND4-2025-01-01"),
        ("loans", 2, "RREL29", "RREL29", "abc"),
        ("loans", 1, None, "lien", "first"), ("loans", 2, None, "lien", "first"),
    ], key=str)
    p5 = {(r["record"], r["raw_value"]) for r in _rows(assessment, "P5") if r["field"] == "RREL29"}
    assert p5 == {(1, "1e3"), (2, "abc")}
    # The dated ND4 is A5's row alone: P3g no longer reports it, and it is no format breach.
    on_rrel16 = {(r["rule_id"], r["result_kind"]) for r in assessment.results.filter(
        (pl.col("field") == "RREL16") & (pl.col("record") == 1)
    ).to_dicts()}
    assert on_rrel16 == {("A5", "violation")}
    assert RULES["A5"].defect_key != RULES["P5"].defect_key
    # The loader stops at the first such column; the audit lists every cell.
    with pytest.raises(PipelineDataError) as caught:
        run_pipeline(
            loans_file_path=tmp_path / "loans.csv", collaterals_file_path=tmp_path / "collaterals.csv",
            deal_name="T", dry_run=True, verbose=False,
        )
    assert "non-numeric value(s) in numeric field RREL16" in str(caught.value)


def test_a5_counts_the_cells_the_loader_reads_as_present(tmp_path: Path) -> None:
    pair = Pair().set("RREL29", "ND1", record=1).set("RREL29", "abc", record=2)
    assessment = _assess(pair, tmp_path)
    outcome = next(o for o in assessment.outcomes if o.rule_id == "A5" and o.field == "RREL29")
    assert (outcome.applicable, outcome.evaluated, outcome.failed) == (1, 1, 1)


# -- A6 ------------------------------------------------------------------------


def test_a6_repaired_bytes_are_context_on_the_cell(tmp_path: Path) -> None:
    def edit(loans: Path, _: Path) -> None:
        raw = loans.read_bytes().replace(b"EXAMPLE BANK NV", b"EXAMPLE BANK N\xe9", 1)
        loans.write_bytes(raw)

    assessment = _assess(Pair(), tmp_path, edit)
    (row,) = _rows(assessment, "A6")
    assert (row["file"], row["record"], row["field"], row["result_kind"]) == ("loans", 1, "RREL82", "context")
    assert row["raw_value"] == "EXAMPLE BANK Né"
    assert _rows(_assess(Pair(), tmp_path), "A6") == []


# -- A7, A8 --------------------------------------------------------------------


def test_a7_a8_duplicate_identifiers_and_whole_record_duplicates(tmp_path: Path) -> None:
    pair = Pair()
    pair.loans.append(dict(pair.loans[0]))  # record 3 repeats record 1 in every cell
    pair.loans.append({**pair.loans[1], "RREL2": "LOAN009", "RREL30": "7.00"})  # record 4: RREL3 of 2
    pair.collateral.append(dict(pair.collateral[1]))  # record 3 repeats record 2
    pair.collateral.append({**pair.collateral[0], "RREC13": "9.00"})  # record 4: the pair of record 1
    assessment = _assess(pair, tmp_path)
    assert [(r["record"], r["field"], r["note"]) for r in _rows(assessment, "A7(a)")] == [
        (1, "RREL2", "also on record 3"), (3, "RREL2", "also on record 1"),
        (1, "RREL3", "also on record 3"), (2, "RREL3", "also on record 4"),
        (3, "RREL3", "also on record 1"), (4, "RREL3", "also on record 2"),
    ]
    assert [(r["record"], r["column_position"], r["note"]) for r in _rows(assessment, "A7(b)")] == [
        (1, None, "also on record 3"), (3, None, "also on record 1"),
    ]
    assert [(r["record"], r["field"]) for r in _rows(assessment, "A8(a)")] == [
        (1, "RREC4"), (2, "RREC4"), (3, "RREC4"), (4, "RREC4"),
    ]
    assert [(r["record"], r["note"]) for r in _rows(assessment, "A8(b)")] == [
        (2, "also on record 3"), (3, "also on record 2"),
    ]


def test_a7_a8_ignore_cells_that_are_not_identifiers(tmp_path: Path) -> None:
    pair = Pair().set("RREL2", "ND5").set("RREL3", "").set("RREC2", "ND1").set("RREC4", "ND1")
    assessment = _assess(pair, tmp_path)
    for rule_id in ("A7(a)", "A8(a)", "A12", "A13(a)", "A13(b)"):
        assert _rows(assessment, rule_id) == [], rule_id


# -- A9, A10 -------------------------------------------------------------------


def test_a9_a_record_with_another_cutoff_and_a_record_without_one(tmp_path: Path) -> None:
    pair = Pair()
    pair.loans.append({**pair.loans[0], "RREL2": "LOAN003", "RREL3": "LOAN003"})
    pair.loans[1].update({"RREL6": "2024-05-31", "Pool_Cutoff_Date": "2024-05-31"})
    pair.loans[2].update({"RREL6": "2024-05-31"})  # two values on the record
    pair.collateral[1]["Pool_Cutoff_Date"] = ""
    assessment = _assess(pair, tmp_path)
    assert _keys(assessment, "A9") == [
        ("loans", 3, "RREL6", "cutoff_not_established_on_record"),
        ("loans", 2, "RREL6", "cutoff_differs_from_the_pool"),
        ("collateral", 2, None, "cutoff_not_established_on_record"),
        ("pool", None, None, "more_than_one_cutoff_or_missing"),
    ]
    pool_row = _rows(assessment, "A9")[-1]
    # Every distinct date with its source and record count, and no cell of a record.
    assert "2024-05-31 (RREL6: 2 record(s), Pool_Cutoff_Date (loans file): 1 record(s))" in str(pool_row["expected"])
    assert "2 record(s) without an established cut-off" in str(pool_row["expected"])
    assert {r["severity"] for r in _rows(assessment, "A9")} == {"Stop"}


def test_a9_the_later_date_is_the_reference_on_a_tie(tmp_path: Path) -> None:
    pair = Pair()
    pair.loans[0].update({"RREL6": "2024-05-31", "Pool_Cutoff_Date": "2024-05-31"})
    assessment = _assess(pair, tmp_path)
    assert ("loans", 1, "RREL6", "cutoff_differs_from_the_pool") in _keys(assessment, "A9")
    assert ("loans", 2, "RREL6", "cutoff_differs_from_the_pool") not in _keys(assessment, "A9")


@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        ("2026-01-03", [("violation", "cutoff_after_the_assessment_date")]),
        ("2026-01-02", [("question", "cutoff_equals_the_assessment_date")]),
        ("2026-01-01", []),
    ],
)
def test_a10a_the_cutoff_against_the_assessment_date(
    tmp_path: Path, cutoff: str, expected: list[tuple[str, str]]
) -> None:
    assert STARTED.date().isoformat() == "2026-01-02"
    assessment = _assess(Pair().set("RREL6", cutoff).set("Pool_Cutoff_Date", cutoff), tmp_path)
    rows = _rows(assessment, "A10(a)")
    assert [(r["result_kind"], r["reason"]) for r in rows] == expected
    assert all(r["raw_value"] == cutoff and r["field"] == "RREL6" for r in rows)
    assert RULES["A10(a)"].regulatory_source is not None  # the single owner of P6(e)'s defect
    assert RULES["A10(a)"].defect_key == RULES["P6(e)"].defect_key


def test_a10b_a_cutoff_before_every_origination_date_is_a_question(tmp_path: Path) -> None:
    assessment = _assess(Pair().set("RREL6", "2019-01-31").set("Pool_Cutoff_Date", "2019-01-31"), tmp_path)
    (row,) = _rows(assessment, "A10(b)")
    assert (row["result_kind"], row["raw_value"]) == ("question", "2019-01-31")
    assert "2019-05-10" in str(row["expected"])
    assert _rows(_assess(Pair(), tmp_path), "A10(b)") == []


# -- A11 -----------------------------------------------------------------------


def test_a11_the_unique_identifier_in_each_file_and_across_them(tmp_path: Path) -> None:
    assessment = _assess(Pair().set("RREL1", "OTHER", record=1), tmp_path)
    assert _keys(assessment, "A11") == [
        ("loans", None, "RREL1", "unique_identifier_not_constant"),
        ("pool", None, None, "unique_identifiers_differ_between_the_files"),
    ]
    assessment = _assess(Pair().set("RREC1", "OTHER"), tmp_path)
    assert _keys(assessment, "A11") == [("pool", None, None, "unique_identifiers_differ_between_the_files")]


# -- A12, A13 ------------------------------------------------------------------


def test_a12_a_value_that_is_a_different_records_other_identifier(tmp_path: Path) -> None:
    pair = Pair().set("RREL2", "LOAN002", record=1)  # equals RREL3 (and RREL2) of record 2
    assessment = _assess(pair, tmp_path)
    assert [(r["record"], r["field"], r["severity"], r["note"]) for r in _rows(assessment, "A12")] == [
        (1, "RREL2", "Stop", "equals RREL3 of record 2"),
        (2, "RREL3", "Stop", "equals RREL2 of record 1"),
    ]
    assert "issue #45" in str(_rows(assessment, "A12")[0]["expected"])
    # The same value in both columns of one record is not a collision.
    assert _rows(_assess(Pair(), tmp_path), "A12") == []


def test_a13_padding_and_identifiers_equal_after_trimming(tmp_path: Path) -> None:
    pair = Pair().set("RREL4", " OBLIGOR002", record=1).set("RREC3", "PROPERTY001 ", record=1)
    assessment = _assess(pair, tmp_path)
    assert [(r["file"], r["record"], r["field"], r["severity"]) for r in _rows(assessment, "A13(a)")] == [
        ("loans", 1, "RREL4", "Stop"), ("collateral", 1, "RREC3", "Stop"),
    ]
    assert "issue #46" in str(_rows(assessment, "A13(a)")[0]["expected"])
    # " OBLIGOR002" and "OBLIGOR002" are two raw values that trimming makes one, within RREL4.
    assert [(r["file"], r["record"], r["field"], r["severity"]) for r in _rows(assessment, "A13(b)")] == [
        ("loans", 1, "RREL4", "Warn"), ("loans", 2, "RREL4", "Warn"),
    ]


# -- the summary ---------------------------------------------------------------


def test_family_a_in_the_summary(tmp_path: Path) -> None:
    pair = Pair().set("RREL2", "LOAN002", record=1).set("RREL6", "2024-05-31", record=1)
    assessment = _assess(pair, tmp_path)
    summary = summary_dict(complete_report(assessment, pipeline=None, error=RuntimeError("x")))
    findings = {f["finding_id"]: f for f in summary["findings"]}
    collision = findings["A12:violation:loans:RREL2"]
    assert collision["sentence"] == (
        "RREL2 Original Underlying Exposure Identifier: the pipeline halts or mismatches on this "
        "today (issue #45): the value equals a different loan record's identifier in the other "
        "identifier column."
    )
    assert (collision["basis"], collision["severities"], collision["defect_key"]) == (
        "L", ["Stop"], "identifier_collision",
    )
    assert not collision["enters_compliance_judgment"]
    pool = findings["A9:violation:pool:None"]
    assert pool["sentence"].startswith("the pool: would stop under the acceptance policy: ")
    assert pool["records"] is None
    duplicate = findings["A7(a):violation:loans:RREL2"]
    assert duplicate["sentence"] == (
        "RREL2 Original Underlying Exposure Identifier: contradicts the requirement that the "
        "identifier is unique: the same value is on more than one loan record."
    )
    assert (duplicate["rows"], duplicate["records"]) == (2, 2)
