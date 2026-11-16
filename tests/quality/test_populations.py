"""Population, identity-confidence and lifecycle labels (specification 8.1, 3.3)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import polars as pl
import pytest

from esma_milan.quality.audit_read import read_file
from esma_milan.quality.column_rules.context import pool_facts
from esma_milan.quality.populations import (
    IDENTITY_AMBIGUOUS,
    IDENTITY_OTHER_COLUMN,
    IDENTITY_UNIQUE,
    IDENTITY_UNKNOWN_SELECTION,
    POPULATIONS,
    collateral_records,
    compare_with_stage2,
    identity_confidence,
    loan_records,
)
from esma_milan.runner import run_pipeline
from tests.quality.builders import Pair

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _loans(pair: Pair, tmp_path: Path) -> pl.DataFrame:
    loans_path, _ = pair.write(tmp_path)
    raw = read_file(loans_path, "loans")
    return loan_records(raw, pool_facts(raw))


def _pair(rows: list[tuple[str, str]]) -> Pair:
    """A pair with one loan record per (status, balance)."""
    pair = Pair()
    template = pair.loans[0]
    pair.loans = [
        {**template, "RREL2": f"L{i}", "RREL3": f"L{i}", "RREL69": status, "RREL30": balance}
        for i, (status, balance) in enumerate(rows, start=1)
    ]
    return pair


@pytest.mark.parametrize(
    ("status", "balance", "population"),
    [
        ("PERF", "100", "Active, retained"),
        ("ARRE", " 100.5 ", "Active, retained"),  # the loader trims a number
        ("RNAR", "NaN", "Active, retained"),  # as Stage 2 does: NaN orders above 0
        ("PERF", "0", "Active status, zero balance"),
        ("RARR", "0.00", "Active status, zero balance"),
        ("PERF", "-5", "Active status, negative balance"),
        ("PERF", "", "Active status, balance missing"),
        ("PERF", "ND5", "Active status, balance missing"),
        ("PERF", "NA", "Active status, balance missing"),
        ("PERF", "abc", "Active status, balance unparseable"),
        ("PERF", "ND4-2030-01-01", "Active status, balance unparseable"),
        (" PERF", "100", "Active status not recognised"),
        ("perf", "0", "Active status not recognised"),  # tested before the balance
        ("RDMD", "0", "Inactive"),
        ("DFLT", "100", "Inactive"),
        ("OTHR", "100", "Inactive"),
        ("rdmd", "0", "Inactive status not recognised"),
        (" RDMD ", "0", "Inactive status not recognised"),
        ("", "100", "Eligibility unknown"),
        ("ND5", "100", "Eligibility unknown"),
        ("ACTIVE", "100", "Eligibility unknown"),
        ("N/A", "abc", "Eligibility unknown"),
    ],
)
def test_every_population(tmp_path: Path, status: str, balance: str, population: str) -> None:
    records = _loans(_pair([(status, balance)]), tmp_path)
    assert records["population"].to_list() == [population]
    assert population in POPULATIONS


def test_each_record_is_in_exactly_one_population(tmp_path: Path) -> None:
    rows = [("PERF", "100"), ("PERF", "0"), ("PERF", "-1"), ("PERF", ""), ("PERF", "x"),
            (" PERF", "1"), ("RDMD", "0"), ("rdmd", "0"), ("", "1")]
    records = _loans(_pair(rows), tmp_path)
    assert records["population"].to_list() == list(POPULATIONS)


def test_absent_status_or_balance_column(tmp_path: Path) -> None:
    no_status = _loans(Pair().drop("RREL69"), tmp_path)
    assert set(no_status["population"].to_list()) == {"Eligibility unknown"}
    no_balance = _loans(Pair().drop("RREL30"), tmp_path)
    assert set(no_balance["population"].to_list()) == {"Active status, balance missing"}


@pytest.mark.parametrize(("fixture", "retained", "records"), [("synthetic", 8, 8), ("real_anonymised", 1126, 1295)])
def test_agreement_with_stage2_on_the_fixtures(fixture: str, retained: int, records: int) -> None:
    folder = FIXTURES / fixture
    result = run_pipeline(
        loans_file_path=folder / "loans.csv", collaterals_file_path=folder / "collaterals.csv",
        deal_name="T", dry_run=True, verbose=False,
    )
    raw = read_file(folder / "loans.csv", "loans")
    frame = loan_records(raw, pool_facts(raw))
    comparison = compare_with_stage2(frame, result.stage2)
    assert comparison.evaluated and comparison.agrees
    assert comparison.differing_records == ()
    assert (comparison.records, comparison.retained_by_audit, comparison.retained_by_stage) == (
        records, retained, retained,
    )
    assert result.stage2 is not None and result.stage2.loan_rows is not None
    kept = frame.filter(pl.col("population") == "Active, retained")["record"].to_list()
    assert kept == [i + 1 for i in result.stage2.loan_rows.to_list()]


def test_a_difference_from_stage2_is_reported_by_record(tmp_path: Path) -> None:
    pair = _pair([("PERF", "100"), ("RDMD", "0"), ("PERF", "50")])
    pair.set("RREC2", "L1", record=1).set("RREC2", "L3", record=2)
    loans_path, collateral_path = pair.write(tmp_path)
    result = run_pipeline(
        loans_file_path=loans_path, collaterals_file_path=collateral_path,
        deal_name="T", dry_run=True, verbose=False,
    )
    records = _loans(pair, tmp_path)
    assert compare_with_stage2(records, result.stage2).agrees
    assert result.stage2 is not None and result.stage2.loan_drop_reasons is not None
    tampered = dataclasses.replace(
        result.stage2,
        loan_drop_reasons=result.stage2.loan_drop_reasons.with_columns(
            pl.Series("status_not_active", [False, False, False])
        ),
    )
    comparison = compare_with_stage2(records, tampered)
    assert comparison.evaluated and not comparison.agrees
    assert comparison.differing_records == (2,)


def test_predicted_when_the_pipeline_did_not_return(tmp_path: Path) -> None:
    comparison = compare_with_stage2(_loans(Pair(), tmp_path), None)
    assert not comparison.evaluated and not comparison.agrees
    assert comparison.reason == "the pipeline did not return a Stage 2 result"
    assert (comparison.retained_by_audit, comparison.retained_by_stage) == (2, None)


def _ids(rows: list[tuple[str, str]]) -> Pair:
    pair = Pair()
    template = pair.loans[0]
    pair.loans = [{**template, "RREL2": a, "RREL3": b} for a, b in rows]
    return pair


def test_identity_confidence(tmp_path: Path) -> None:
    rows = [("A1", "N1"), ("A1", "N2"), ("A3", "N3"), ("A4", " N3"), ("N5x", "N5"), ("A6", "N5x")]
    records = _loans(_ids(rows), tmp_path)
    # RREL2: A1 twice; N5x equals another record's RREL3. RREL3: N3 and " N3" collide after trimming;
    # N5x equals another record's RREL2.
    assert identity_confidence(records, "RREL3").to_list() == [
        IDENTITY_OTHER_COLUMN, IDENTITY_OTHER_COLUMN, IDENTITY_AMBIGUOUS, IDENTITY_AMBIGUOUS,
        IDENTITY_OTHER_COLUMN, IDENTITY_AMBIGUOUS,
    ]
    assert identity_confidence(records, "RREL2").to_list() == [
        IDENTITY_AMBIGUOUS, IDENTITY_AMBIGUOUS, IDENTITY_OTHER_COLUMN, IDENTITY_OTHER_COLUMN,
        IDENTITY_AMBIGUOUS, IDENTITY_OTHER_COLUMN,
    ]
    assert set(identity_confidence(records, None).to_list()) == {IDENTITY_UNKNOWN_SELECTION}
    clean = _loans(Pair(), tmp_path)  # RREL2 equals RREL3 on the same record: not a collision
    assert identity_confidence(clean, "RREL3").to_list() == [IDENTITY_UNIQUE, IDENTITY_UNIQUE]
    assert identity_confidence(clean, None).to_list() == [IDENTITY_UNIQUE, IDENTITY_UNIQUE]
    assert records["RREL3_trimmed"].to_list()[3] == "N3" and records["RREL3"].to_list()[3] == " N3"


@pytest.mark.parametrize(
    ("status", "rrel9", "lifecycle", "evidence"),
    [
        ("PERF", "ND5", "reporting-active (presumed)", "status PERF"),
        ("RARR", "2020-01-01", "reporting-active (presumed)", "status RARR"),
        ("RDMD", "2024-01-31", "final-inactive (presumed)", "status RDMD; RREL9 2024-01-31 (not after the cut-off)"),
        # A conflict in the evidence is shown; the claim is not relabelled.
        ("RDMD", "2030-01-01", "final-inactive (presumed)", "status RDMD; RREL9 2030-01-01 (after the cut-off)"),
        ("RDMD", "ND5", "final-inactive (presumed)", "status RDMD; RREL9 no valid date"),
        ("REBR", "ND5", "final-inactive (presumed)", "status REBR"),
        ("DFLT", "2024-06-30", "final-inactive (presumed)", "status DFLT; RREL9 2024-06-30 (not after the cut-off)"),
        ("DFLT", "ND5", "default, recoveries not determinable", "status DFLT; RREL9 no valid date"),
        ("DADB", "2024-07-01", "default, recoveries not determinable", "status DADB; RREL9 2024-07-01 (after the cut-off)"),
        ("OTHR", "ND5", "not established", "status OTHR"),
        ("", "ND5", "not established", "status empty"),
        (" PERF", "ND5", "not established", "status  PERF"),
    ],
)
def test_claimed_lifecycle(tmp_path: Path, status: str, rrel9: str, lifecycle: str, evidence: str) -> None:
    records = _loans(Pair().set("RREL69", status, record=1).set("RREL9", rrel9, record=1), tmp_path)
    assert records["lifecycle"].to_list()[0] == lifecycle
    assert records["lifecycle_evidence"].to_list()[0] == evidence


def test_lifecycle_does_not_use_the_values_it_will_check(tmp_path: Path) -> None:
    # RDMD with a positive balance stays final-inactive (presumed).
    records = _loans(Pair().set("RREL69", "RDMD", record=1).set("RREL30", "5000", record=1), tmp_path)
    assert records["lifecycle"].to_list()[0] == "final-inactive (presumed)"


def test_default_with_a_date_but_no_cut_off(tmp_path: Path) -> None:
    pair = Pair().set("RREL69", "DFLT", record=1).set("RREL9", "2020-01-01", record=1)
    pair.set("RREL6", "2024-06-29", record=1)  # disagrees with Pool_Cutoff_Date
    records = _loans(pair, tmp_path)
    assert records["lifecycle"].to_list()[0] == "default, recoveries not determinable"
    assert records["lifecycle_evidence"].to_list()[0] == "status DFLT; RREL9 2020-01-01 (cut-off not established)"


def test_collateral_class(tmp_path: Path) -> None:
    pair = Pair()
    template = pair.collateral[0]
    pair.collateral = [{**template, "RREC5": code} for code in ("RBLD", "OTRE", "GUAR", "CARX", "OTHR", "MIXD", "", "rbld")]
    _, collateral_path = pair.write(tmp_path)
    records = collateral_records(read_file(collateral_path, "collateral"))
    assert records["collateral_class"].to_list() == [
        "property", "property", "established non-property", "established non-property",
        "undetermined", "undetermined", "not established", "not established",
    ]
