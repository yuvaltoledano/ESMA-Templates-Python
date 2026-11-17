"""P1 and P3a to P3g, each firing and not firing."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.quality.builders import Pair, assess


def test_the_compliant_pair_has_no_row(tmp_path: Path) -> None:
    assert assess(Pair(), tmp_path, "loans").rows_beyond_cardinality() == []
    assert assess(Pair(), tmp_path, "collateral").rows_beyond_cardinality() == []


def test_p1_absent_field(tmp_path: Path) -> None:
    assessed = assess(Pair().drop("RREL18"), tmp_path, "loans")
    rows = assessed.found("P1")
    assert [(r["field"], r["result_kind"], r["record"], r["column_position"]) for r in rows] == [
        ("RREL18", "violation", None, None)
    ]
    assert assessed.outcome("P1", "RREL18") == (1, 1, 1, 0)
    assert assessed.outcome("P1", "RREL28") == (1, 1, 0, 0)


def test_p3a_and_p3b_nd_not_allowed(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL30", "ND1", record=1)  # ND1-4 not allowed, ND5 allowed
    pair.set("RREL30", "ND5", record=2)
    pair.set("RREL23", "ND5", record=1)  # ND5 not allowed, ND1-4 allowed
    pair.set("RREL23", "ND2", record=2)
    pair.set("RREL69", "ND4-2030-01-01", record=1)  # dated ND4 where no ND is allowed
    assessed = assess(pair, tmp_path, "loans")
    assert [(r["field"], r["record"], r["raw_value"]) for r in assessed.found("P3a")] == [
        ("RREL30", 1, "ND1"), ("RREL69", 1, "ND4-2030-01-01"),
    ]
    assert [(r["field"], r["record"]) for r in assessed.found("P3b")] == [("RREL23", 1)]
    assert assessed.outcome("P3a", "RREL30") == (2, 2, 1, 0)
    assert assessed.found("P5", "RREL30") == []  # a valid ND is never also a failed number


def test_p3c_empty_and_p3d_marker(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL30", "", record=1)
    pair.set("RREL22", " ", record=1)  # empty in free text too
    pair.set("RREL43", "N/A", record=1)
    pair.set("RREL43", "nd5", record=2)
    pair.set("RREL22", "NULL", record=2)  # content in free text
    assessed = assess(pair, tmp_path, "loans")
    assert sorted((r["field"], r["record"]) for r in assessed.found("P3c")) == [
        ("RREL22", 1), ("RREL30", 1),
    ]
    assert [(r["field"], r["record"], r["raw_value"]) for r in assessed.found("P3d")] == [
        ("RREL43", 1, "N/A"), ("RREL43", 2, "nd5"),
    ]
    assert assessed.found("P5", "RREL43") == []  # a marker is P3d only


def test_p3e_dated_nd4(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL16", "ND4-2025-01-01", record=1)  # valid: after the cut-off
    pair.set("RREL16", "ND4-2024-06-30", record=2)  # not after the cut-off
    pair.set("RREL19", "ND4", record=1)
    pair.set("RREL19", "ND4-2025-13-01", record=2)
    assessed = assess(pair, tmp_path, "loans")
    rows = assessed.found("P3e")
    assert [(r["field"], r["record"], r["reason"]) for r in rows] == [
        ("RREL16", 2, "nd4_date_not_after_cutoff"),
        ("RREL19", 1, "nd4_malformed"),
        ("RREL19", 2, "nd4_malformed"),
    ]
    assert str(rows[0]["nd4_date"]) == "2024-06-30"
    assert assessed.outcome("P3e", "RREL16") == (2, 2, 1, 0)
    assert assessed.found("P3a", "RREL16") == []  # ND1-4 are allowed for RREL16


def test_p3e_not_evaluable_without_a_cut_off(tmp_path: Path) -> None:
    pair = Pair().set("RREL16", "ND4-2025-01-01", record=1)
    pair.set("RREL6", "2024-06-29", record=1)  # disagrees with Pool_Cutoff_Date
    assessed = assess(pair, tmp_path, "loans")
    assert assessed.found("P3e") == []
    assert assessed.outcome("P3e", "RREL16") == (1, 0, 0, 1)
    outcome = next(o for o in assessed.assessment.outcomes if o.rule_id == "P3e" and o.field == "RREL16")
    assert dict(outcome.not_evaluable_reasons) == {
        "cut-off not established (two values on the record)": 1
    }


CONDITIONAL = ("RREC7", "RREC8", "RREC9", "RREC12", "RREC16")


def test_p3f_value_on_non_property_collateral(tmp_path: Path) -> None:
    pair = Pair().set("RREC5", "CARX", record=1)  # a vehicle; record 2 stays property
    assessed = assess(pair, tmp_path, "collateral")
    rows = assessed.found("P3f")
    assert sorted((r["field"], r["record"], r["result_kind"]) for r in rows) == sorted(
        (code, 1, "violation") for code in CONDITIONAL
    )
    assert assessed.outcome("P3f", "RREC7") == (2, 2, 1, 0)


def test_p3f_nd5_on_non_property_is_the_prescribed_value(tmp_path: Path) -> None:
    pair = Pair().set("RREC5", "CARX")
    for code in CONDITIONAL:
        pair.set(code, "ND5")
    assessed = assess(pair, tmp_path, "collateral")
    assert assessed.found("P3f") == []
    assert assessed.outcome("P3f", "RREC12") == (2, 2, 0, 0)


def test_p3f_lien_on_a_guarantee_is_compliant(tmp_path: Path) -> None:
    pair = Pair().set("RREC5", "GUAR")
    for code in ("RREC7", "RREC9", "RREC12", "RREC16"):
        pair.set(code, "ND5")
    assessed = assess(pair, tmp_path, "collateral")  # RREC8 keeps its value 1
    assert assessed.found("P3f") == []


def test_p3f_nd5_on_a_property_row_is_a_limitation(tmp_path: Path) -> None:
    pair = Pair().set("RREC7", "ND5", record=1).set("RREC12", "ND5", record=1)
    assessed = assess(pair, tmp_path, "collateral")
    rows = assessed.found("P3f")
    assert sorted((r["field"], r["record"], r["result_kind"]) for r in rows) == [
        ("RREC12", 1, "limitation"), ("RREC7", 1, "limitation"),
    ]
    assert assessed.outcome("P3f", "RREC7") == (2, 1, 0, 1)
    assert not [r for r in assessed.assessment.results.to_dicts() if r["result_kind"] == "question"]


@pytest.mark.parametrize("collateral_type", ["OTHR", "MIXD", "xxxx", ""])
def test_p3f_undetermined_collateral_is_a_limitation(tmp_path: Path, collateral_type: str) -> None:
    assessed = assess(Pair().set("RREC5", collateral_type, record=2), tmp_path, "collateral")
    rows = assessed.found("P3f")
    assert sorted((r["field"], r["record"], r["result_kind"]) for r in rows) == sorted(
        (code, 2, "limitation") for code in CONDITIONAL
    )
    assert assessed.outcome("P3f", "RREC9") == (2, 1, 0, 1)


def test_p3f_without_the_collateral_type_column(tmp_path: Path) -> None:
    assessed = assess(Pair().drop("RREC5"), tmp_path, "collateral")
    assert assessed.found("P3f") == []
    assert assessed.outcome("P3f", "RREC9") == (2, 0, 0, 2)


def test_p3g_na_as_a_country_is_a_value_the_loader_loses(tmp_path: Path) -> None:
    assessed = assess(Pair().set("RREL81", "NA", record=1), tmp_path, "loans")
    assert assessed.rules_on("RREL81") == {("P3g", "context")}  # no compliance violation
    row = assessed.found("P3g")[0]
    assert (row["record"], row["reason"], row["severity"]) == (1, "valid_value_read_as_missing", "Warn")


def test_p3g_dated_nd4_by_field_type(tmp_path: Path) -> None:
    pair = Pair()
    pair.set("RREL16", "ND4-2025-01-01", record=1)  # numeric: the loader stops (A5's row)
    pair.set("RREL23", "ND4-2025-01-01", record=1)  # date: set to missing with a warning
    pair.set("RREL78", "ND4-2025-01-01", record=1)  # text: kept
    assessed = assess(pair, tmp_path, "loans")
    by_field = {r["field"]: (r["result_kind"], r["severity"], r["reason"]) for r in assessed.found("P3g")}
    assert by_field == {
        "RREL23": ("context", "Warn", "dated_nd4_set_to_missing_with_a_warning"),
        "RREL78": ("context", "Warn", "dated_nd4_kept_as_text"),
    }


def test_p3g_a_date_form_the_pipeline_parses(tmp_path: Path) -> None:
    pair = Pair().set("RREL23", "2019/05/10", record=1).set("RREL23", "43595", record=2)
    assessed = assess(pair, tmp_path, "loans")
    assert [(r["record"], r["result_kind"]) for r in assessed.found("P5", "RREL23")] == [
        (1, "violation"), (2, "violation"),
    ]
    assert [(r["record"], r["reason"]) for r in assessed.found("P3g", "RREL23")] == [
        (1, "date_parsed_by_the_pipeline"), (2, "date_parsed_by_the_pipeline"),
    ]


# --- P3f on a property row, assessed through the link to the loan record ---


def _p3f_through_the_link(tmp_path: Path, pair: Pair) -> list[tuple[int, str, str]]:
    from esma_milan.quality import assess_files

    loans, collaterals = pair.write(tmp_path)
    results = assess_files(loans, collaterals).collateral.results
    rows = results.filter((results["rule_id"] == "P3f") & (results["field"] == "RREC7")).to_dicts()
    return [(int(str(r["record"])), str(r["result_kind"]), str(r["reason"])) for r in rows]


def test_p3f_nd5_on_property_of_a_reporting_active_loan_is_a_question(tmp_path: Path) -> None:
    pair = Pair().set("RREC7", "ND5", record=1)  # LOAN001 is PERF
    assert _p3f_through_the_link(tmp_path, pair) == [
        (1, "question", "nd5_on_property_of_a_reporting_active_exposure")
    ]


def test_p3f_nd5_on_property_of_a_final_inactive_loan_is_the_prescribed_value(tmp_path: Path) -> None:
    pair = Pair().set("RREC7", "ND5", record=1)
    pair.set("RREL69", "RDMD", record=1).set("RREL9", "2024-01-31", record=1).set("RREL30", "0", record=1)
    assert _p3f_through_the_link(tmp_path, pair) == []
    repurchased = Pair().set("RREC7", "ND5", record=2).set("RREL69", "REBR", record=2)
    assert _p3f_through_the_link(tmp_path, repurchased) == []


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"RREL69": "DFLT"}, "nd5_on_property_default_recoveries_not_determinable"),
        ({"RREL69": "OTHR"}, "nd5_on_property_lifecycle_not_established"),
        ({"RREL69": ""}, "nd5_on_property_lifecycle_not_established"),
        ({"RREL2": "ELSEWHERE", "RREL3": "ELSEWHERE"}, "nd5_on_property_no_loan_record_linked"),
    ],
)
def test_p3f_where_the_link_does_not_establish_it_the_limitation_stays(
    tmp_path: Path, changes: dict[str, str], reason: str
) -> None:
    pair = Pair().set("RREC7", "ND5", record=1)
    for header, value in changes.items():
        pair.set(header, value, record=1)
    assert _p3f_through_the_link(tmp_path, pair) == [(1, "limitation", reason)]


def test_p3f_a_record_of_two_loans_whose_lifecycles_differ(tmp_path: Path) -> None:
    pair = Pair().set("RREC7", "ND5", record=1)
    pair.set("RREL3", "LOAN001").set("RREL2", "LOAN001", record=1).set("RREL2", "OTHER", record=2)
    pair.set("RREL69", "RDMD", record=2).set("RREL9", "2024-01-31", record=2)
    assert _p3f_through_the_link(tmp_path, pair) == [(1, "limitation", "nd5_on_property_linked_loans_differ")]


def test_p3f_a_value_on_property_and_nd5_on_non_property_need_no_link(tmp_path: Path) -> None:
    assert _p3f_through_the_link(tmp_path, Pair()) == []
    non_property = Pair().set("RREC5", "CARX", record=1).set("RREL69", "OTHR")
    assert _p3f_through_the_link(tmp_path, non_property) == [(1, "violation", "value_on_non_property_collateral")]
