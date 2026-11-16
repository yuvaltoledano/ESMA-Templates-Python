"""The read-only stage interface on both fixtures.

The stages report their own decisions as data (which records were kept
and why the others were dropped, which identifier column was chosen, which
valuation was used). These tests check, on the synthetic and the real
fixture, that the reported data agrees with what the stages produced and
with the figures the stages already log.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from esma_milan.pipeline.filters import (
    _keep,
    collateral_drop_reasons,
    collect_active_loan_ids,
    loan_drop_reasons,
)
from esma_milan.runner import PipelineResult, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SYNTHETIC = FIXTURES / "synthetic"
REAL = FIXTURES / "real_anonymised"

FIXTURE_PARAMS = [
    pytest.param(SYNTHETIC, id="synthetic"),
    pytest.param(
        REAL,
        id="real_anonymised",
        marks=pytest.mark.skipif(
            not (REAL / "loans.csv").exists(),
            reason="Real-anonymised fixture not present in this checkout",
        ),
    ),
]

_RESULTS: dict[Path, PipelineResult] = {}


def _run(fixture_dir: Path) -> PipelineResult:
    """One dry run per fixture, shared by every test in the module."""
    if fixture_dir not in _RESULTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _RESULTS[fixture_dir] = run_pipeline(
                loans_file_path=fixture_dir / "loans.csv",
                collaterals_file_path=fixture_dir / "collaterals.csv",
                deal_name="INTERFACE",
                dry_run=True,
                verbose=True,
            )
    return _RESULTS[fixture_dir]


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage2_expressions_keep_exactly_the_stage2_output(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage1 is not None and result.stage2 is not None
    stage1, stage2 = result.stage1, result.stage2

    loans_kept = stage1.loans.filter(_keep(loan_drop_reasons()))
    assert loans_kept.equals(stage2.loans)
    active = collect_active_loan_ids(loans_kept)
    properties_kept = stage1.properties.filter(_keep(collateral_drop_reasons(active)))
    assert properties_kept.equals(stage2.properties)


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage2_rows_index_the_stage1_frames(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage1 is not None and result.stage2 is not None
    stage1, stage2 = result.stage1, result.stage2
    assert stage2.loan_rows is not None and stage2.property_rows is not None
    assert stage2.loan_drop_reasons is not None
    assert stage2.property_drop_reasons is not None

    assert stage1.loans[stage2.loan_rows].equals(stage2.loans)
    assert stage1.properties[stage2.property_rows].equals(stage2.properties)
    assert stage2.loan_rows.is_sorted() and stage2.property_rows.is_sorted()

    # One reasons row per Stage 1 record; kept exactly when none is true.
    assert stage2.loan_drop_reasons.height == stage1.loans.height
    assert stage2.property_drop_reasons.height == stage1.properties.height
    for reasons, rows in (
        (stage2.loan_drop_reasons, stage2.loan_rows),
        (stage2.property_drop_reasons, stage2.property_rows),
    ):
        no_reason = reasons.select(~pl.any_horizontal(pl.all())).to_series()
        assert no_reason.arg_true().to_list() == rows.to_list()


def test_real_fixture_drop_reason_counts() -> None:
    """The real fixture's Stage 2 losses: 169 redeemed loans with a zero
    balance, and the 288 collateral rows that reference them."""
    if not (REAL / "loans.csv").exists():
        pytest.skip("Real-anonymised fixture not present in this checkout")
    stage2 = _run(REAL).stage2
    assert stage2 is not None
    assert stage2.loan_drop_reasons is not None
    assert stage2.property_drop_reasons is not None
    assert {c: int(stage2.loan_drop_reasons[c].sum()) for c in stage2.loan_drop_reasons.columns} == {
        "status_not_active": 169,
        "balance_not_positive": 169,
    }
    assert {
        c: int(stage2.property_drop_reasons[c].sum())
        for c in stage2.property_drop_reasons.columns
    } == {
        "property_type_missing": 0,
        "collateral_type_missing": 0,
        "collateral_type_excluded": 0,
        "no_active_loan": 288,
        "no_positive_valuation": 0,
    }


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------


def _stage3_logs(fixture_dir: Path) -> list[dict[str, object]]:
    """Run Stages 1-3 on the fixture and return the Stage 3 log events."""
    import structlog

    from esma_milan.pipeline.filters import run_stage2
    from esma_milan.pipeline.identifiers import run_stage3
    from esma_milan.pipeline.stage1 import run_stage1

    stage1 = run_stage1(
        loans_path=fixture_dir / "loans.csv",
        collaterals_path=fixture_dir / "collaterals.csv",
    )
    stage2 = run_stage2(stage1.loans, stage1.properties)
    with structlog.testing.capture_logs() as logs, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        run_stage3(stage2.loans, stage2.properties)
    assert caught == [], "neither fixture warns in Stage 3"
    return [dict(event) for event in logs]


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage3_figures_equal_the_select_calc_loan_id_log_event(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage3 is not None and result.stage3.decisions is not None
    selection = result.stage3.decisions.loan_id
    (event,) = [e for e in _stage3_logs(fixture_dir) if e["event"] == "select_calc_loan_id"]
    original, new = selection.candidates
    assert selection.chosen_column == event["picked"]
    assert selection.reason == event["reason"]
    assert original.has_duplicates == event["original_duplicates"]
    assert new.has_duplicates == event["new_duplicates"]
    assert f"{original.matched}/{original.total}" == event["original_coverage"]
    assert f"{new.matched}/{new.total}" == event["new_coverage"]
    assert selection.min_coverage == 0.85
    for candidate in (original, new):
        assert len(candidate.unmatched) == candidate.total - candidate.matched


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage3_id_choices_equal_the_generate_id_column_log_events(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage3 is not None and result.stage3.decisions is not None
    decisions = result.stage3.decisions
    events = {
        e["output_col"]: e
        for e in _stage3_logs(fixture_dir)
        if e["event"] == "generate_id_column"
    }
    assert set(events) == {"calc_borrower_id", "calc_property_id"}
    for choice, frame in (
        (decisions.borrower_id, result.stage3.loans),
        (decisions.property_id, result.stage3.properties),
    ):
        event = events[choice.output_column]
        assert choice.original_column == event["original"]
        assert choice.new_column == event["new"]
        assert choice.n_unique_original == event["n_unique_original"]
        assert choice.n_unique_new == event["n_unique_new"]
        assert choice.decision == event["decision"]
        assert choice.identical is (choice.decision == "identical, using original")
        # The output column really is the source column named.
        assert frame[choice.output_column].to_list() == frame[choice.chosen_column].to_list()


def test_fixture_stage3_figures() -> None:
    """Pinned figures: synthetic picks the new column at 8/8 against 0/8;
    the real fixture has both fully covered and defaults to original."""
    synthetic = _run(SYNTHETIC).stage3
    assert synthetic is not None and synthetic.decisions is not None
    selection = synthetic.decisions.loan_id
    assert selection.chosen_column == "new_underlying_exposure_identifier"
    assert selection.reason == "full coverage"
    assert [(c.matched, c.total) for c in selection.candidates] == [(0, 8), (8, 8)]
    assert len(selection.candidates[0].unmatched) == 8
    if (REAL / "loans.csv").exists():
        real = _run(REAL).stage3
        assert real is not None and real.decisions is not None
        selection = real.decisions.loan_id
        assert selection.chosen_column == "original_underlying_exposure_identifier"
        assert selection.reason == "both fully covered; default to original"
        assert [(c.matched, c.total) for c in selection.candidates] == [(1126, 1126)] * 2
        assert real.decisions.borrower_id.chosen_column == "new_obligor_identifier"
        assert real.decisions.property_id.identical


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage3_removed_nothing_on_the_fixtures(fixture_dir: Path) -> None:
    result = _run(fixture_dir)
    assert result.stage2 is not None and result.stage3 is not None
    decisions = result.stage3.decisions
    assert decisions is not None
    assert decisions.loans_missing_calc_loan_id.len() == 0
    assert decisions.loans_without_collateral.len() == 0
    assert decisions.properties_removed.len() == 0
    assert decisions.unmatched_loan_ids == ()
    assert decisions.unmatched_collateral_ids == ()
    assert decisions.loan_rows.to_list() == list(range(result.stage2.loans.height))
    assert decisions.property_rows.to_list() == list(range(result.stage2.properties.height))


# ---------------------------------------------------------------------------
# Record index
# ---------------------------------------------------------------------------


def _assert_record_index_and_alignment(result: PipelineResult) -> None:
    from esma_milan.pipeline.enriched import (
        compose_loans_enriched,
        compose_properties_enriched,
    )

    assert result.stage1 is not None and result.stage3 is not None
    assert result.stage4 is not None and result.stage5 is not None
    assert result.stage7 is not None
    assert result.loan_record_index is not None
    assert result.collateral_record_index is not None
    stage1, stage3 = result.stage1, result.stage3

    # The index points each kept row at its record in the Stage 1 frame,
    # which is the file's record order.
    loan_columns = stage1.loans.columns
    assert stage1.loans[result.loan_record_index].equals(stage3.loans.select(loan_columns))
    property_columns = stage1.properties.columns
    assert stage1.properties[result.collateral_record_index].equals(
        stage3.properties.select(property_columns)
    )
    assert result.loan_record_index.dtype == pl.UInt32
    assert result.loan_record_index.n_unique() == stage3.loans.height
    assert result.collateral_record_index.n_unique() == stage3.properties.height

    # The enriched frames (the two "Cleaned ESMA" sheets) and the flattened
    # pool keep Stage 3's row order, so the same index serves them.
    loans_enriched = compose_loans_enriched(
        stage3.loans, result.stage4.loan_groups, result.stage5.classifications
    )
    properties_enriched = compose_properties_enriched(
        stage3.properties, result.stage4.collateral_groups, result.stage5.classifications
    )
    assert loans_enriched.select(stage3.loans.columns).equals(stage3.loans)
    assert properties_enriched.select(stage3.properties.columns).equals(stage3.properties)
    assert (
        result.stage7.combined_flattened["calc_loan_id"].to_list()
        == stage3.loans["calc_loan_id"].to_list()
    )


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_record_index_and_row_alignment_on_the_fixtures(fixture_dir: Path) -> None:
    _assert_record_index_and_alignment(_run(fixture_dir))


def test_record_index_with_a_fully_duplicated_collateral_row(tmp_path: Path) -> None:
    """Nothing in the data tells two identical collateral rows apart; the
    record index does. The duplicate also shifts every later record."""
    lines = (SYNTHETIC / "collaterals.csv").read_text().splitlines()
    duplicated = [*lines[:3], lines[2], *lines[3:]]  # data record 1 repeated as record 2
    collaterals = tmp_path / "collaterals.csv"
    collaterals.write_text("\n".join(duplicated) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_pipeline(
            loans_file_path=SYNTHETIC / "loans.csv",
            collaterals_file_path=collaterals,
            deal_name="DUP",
            dry_run=True,
            verbose=False,
        )
    _assert_record_index_and_alignment(result)
    assert result.stage3 is not None and result.collateral_record_index is not None
    properties = result.stage3.properties
    assert properties.height == 13
    assert properties[1].equals(properties[2])  # identical in every column
    assert result.collateral_record_index.to_list() == list(range(13))
    assert result.collateral_record_index[1] != result.collateral_record_index[2]


def test_record_index_skips_records_stage2_dropped(tmp_path: Path) -> None:
    """Make the third loan inactive: its record index (2) is absent and the
    later loans keep their file positions."""
    import csv

    with (SYNTHETIC / "loans.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        header, rows = list(reader.fieldnames), list(reader)
    rows[2]["account_status"] = "RDMD"
    dropped_loan = rows[2]["new_underlying_exposure_identifier"]
    loans = tmp_path / "loans.csv"
    with loans.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run_pipeline(
            loans_file_path=loans,
            collaterals_file_path=SYNTHETIC / "collaterals.csv",
            deal_name="DROP",
            dry_run=True,
            verbose=False,
        )
    _assert_record_index_and_alignment(result)
    assert result.loan_record_index is not None
    assert result.loan_record_index.to_list() == [0, 1, 3, 4, 5, 6, 7]
    assert result.stage1 is not None and result.collateral_record_index is not None
    collateral_loans = result.stage1.properties["underlying_exposure_identifier"].to_list()
    assert result.collateral_record_index.to_list() == [
        i for i, loan in enumerate(collateral_loans) if loan != dropped_loan
    ]


# ---------------------------------------------------------------------------
# Stage 6 evidence and the aggregation route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_stage6_counts_equal_the_log_event(fixture_dir: Path) -> None:
    import structlog

    from esma_milan.pipeline.enriched import (
        compose_loans_enriched,
        compose_properties_enriched,
    )
    from esma_milan.pipeline.valuation import run_stage6

    result = _run(fixture_dir)
    assert result.stage3 is not None and result.stage4 is not None
    assert result.stage5 is not None and result.stage6 is not None
    loans_enriched = compose_loans_enriched(
        result.stage3.loans, result.stage4.loan_groups, result.stage5.classifications
    )
    properties_enriched = compose_properties_enriched(
        result.stage3.properties, result.stage4.collateral_groups, result.stage5.classifications
    )
    with structlog.testing.capture_logs() as logs:
        again = run_stage6(loans_enriched, properties_enriched)
    (event,) = [e for e in logs if e["event"] == "detect_aggregation_method"]
    stage6 = result.stage6
    assert (stage6.n_by_loan, stage6.n_by_group, stage6.total_evidence) == (
        event["n_by_loan"], event["n_by_group"], event["total_evidence"],
    )
    assert stage6.branch == "by_loan" == again.branch
    assert stage6.evidence is not None and again.evidence is not None
    assert stage6.evidence.equals(again.evidence)
    counted = stage6.evidence.filter(pl.col("implied_method").is_not_null())
    assert counted.height == stage6.total_evidence
    assert stage6.message == (
        f"Strong evidence ({stage6.n_by_loan}/{stage6.total_evidence} "
        "properties) suggests 'by_loan' duplication."
    )


def _subset_pool(tmp_path: Path, loan_ids: set[str]) -> tuple[Path, Path]:
    """The synthetic pool restricted to the given new loan identifiers."""
    import csv

    paths = []
    for name, column in (
        ("loans.csv", "new_underlying_exposure_identifier"),
        ("collaterals.csv", "underlying_exposure_identifier"),
    ):
        with (SYNTHETIC / name).open(newline="") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames is not None
            header = list(reader.fieldnames)
            rows = [r for r in reader if r[column] in loan_ids]
        out = tmp_path / name
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        paths.append(out)
    return paths[0], paths[1]


def test_aggregation_source_detected_argument_and_fallback(tmp_path: Path) -> None:
    detected = _run(SYNTHETIC)
    assert detected.aggregation_source == "detected"
    assert detected.chosen_aggregation_method == "by_loan"

    explicit = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        deal_name="INTERFACE",
        aggregation_method="by_group",
        dry_run=True,
        verbose=False,
    )
    assert explicit.aggregation_source == "argument"
    assert explicit.chosen_aggregation_method == "by_group"
    assert explicit.stage6 is not None and explicit.stage6.branch == "by_loan"
    assert explicit.stage7 is not None and explicit.stage7.provenance is not None
    assert explicit.stage7.provenance.aggregation_method == "by_group"

    # No group with more than one loan: nothing to detect, by_loan is used.
    loans, collaterals = _subset_pool(tmp_path, {"NEW_001", "NEW_002"})
    fallback = run_pipeline(
        loans_file_path=loans,
        collaterals_file_path=collaterals,
        deal_name="INTERFACE",
        dry_run=True,
        verbose=False,
    )
    assert fallback.aggregation_source == "ambiguous_fallback"
    assert fallback.chosen_aggregation_method == "by_loan"
    assert fallback.stage6 is not None
    assert fallback.stage6.branch == "no_multi_loan_groups"


# ---------------------------------------------------------------------------
# Valuation provenance
# ---------------------------------------------------------------------------


def _properties_enriched(result: PipelineResult) -> pl.DataFrame:
    from esma_milan.pipeline.enriched import compose_properties_enriched

    assert result.stage3 is not None and result.stage4 is not None
    assert result.stage5 is not None
    return compose_properties_enriched(
        result.stage3.properties, result.stage4.collateral_groups, result.stage5.classifications
    )


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_provenance_records_agree_with_the_selection_functions(fixture_dir: Path) -> None:
    from esma_milan.pipeline.valuation import (
        select_valuation_amount,
        select_valuation_fields,
    )

    result = _run(fixture_dir)
    assert result.stage3 is not None and result.stage7 is not None
    provenance = result.stage7.provenance
    assert provenance is not None
    enriched = _properties_enriched(result)
    records = provenance.records

    assert records.height == result.stage3.properties.height
    assert records["record"].to_list() == list(range(records.height))
    # A record is a row of stage3.properties.
    for column in ("calc_property_id", "underlying_exposure_identifier"):
        assert records[column].to_list() == result.stage3.properties[column].to_list()
    # Stage 7 (method, date, amount) and Stage 6 (amount) made the same
    # choice per record: both use this selection on this frame.
    fields = select_valuation_fields(enriched)
    for column in ("valuation_method", "valuation_date", "valuation_amount"):
        assert records[column].to_list() == fields[column].to_list()
    assert (
        records["valuation_amount"].to_list()
        == select_valuation_amount(enriched)["valuation_amount"].to_list()
    )


def test_fixture_selection_counts() -> None:
    """Synthetic: one of twelve records flips (a full inspection with an
    amount of 5). Real: all 1,664 prefer current and none flips."""
    synthetic = _run(SYNTHETIC).stage7
    assert synthetic is not None and synthetic.provenance is not None
    records = synthetic.provenance.records
    assert records.height == 12
    assert records["flipped"].sum() == 1
    assert records.filter(pl.col("flipped")).select("preferred", "source").row(0) == (
        "current", "original",
    )
    assert synthetic.provenance.properties.filter(pl.col("records").list.len() > 1).height == 4
    if (REAL / "loans.csv").exists():
        real = _run(REAL).stage7
        assert real is not None and real.provenance is not None
        records = real.provenance.records
        assert records.height == 1664
        assert records["preferred"].unique().to_list() == ["current"]
        assert records["flipped"].sum() == 0
        assert real.provenance.properties.filter(pl.col("records").list.len() > 1).height == 472


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_milan_property_value_is_reproduced_from_the_provenance(fixture_dir: Path) -> None:
    """Record -> property -> group -> loan, using only the provenance
    frames, gives every cell of MILAN "Property Value"."""
    import math

    from esma_milan.pipeline.milan_map import compose_milan_pool

    result = _run(fixture_dir)
    assert result.stage7 is not None and result.stage7.provenance is not None
    provenance = result.stage7.provenance
    assert provenance.aggregation_method == "by_loan"
    amounts = provenance.records["valuation_amount"].to_list()

    # Property value: the amount on the record named.
    property_values: dict[int, list[float]] = {}
    seen_records: list[int] = []
    for prop in provenance.properties.to_dicts():
        seen_records.extend(prop["records"])
        if prop["value_record"] is None:
            assert prop["property_value"] is None
            continue
        assert prop["value_record"] in prop["records"]
        assert prop["value_record"] == min(
            r for r in prop["records"] if amounts[r] is not None
        )
        assert amounts[prop["value_record"]] == prop["property_value"]
        property_values.setdefault(prop["collateral_group_id"], []).append(
            prop["property_value"]
        )
    assert sorted(seen_records) == list(range(len(amounts)))  # every record, once

    # Group total: the sum of its property values. The stored total is
    # used below; a Python sum and a Polars sum may differ in the last bit.
    groups = {g["collateral_group_id"]: g for g in provenance.groups.to_dicts()}
    for gid, group in groups.items():
        assert math.isclose(
            group["total_property_value"], sum(property_values[gid]), rel_tol=1e-12
        )
        assert group["metadata_record"] is not None
        meta = provenance.records.row(group["metadata_record"], named=True)
        assert meta["calc_property_id"] == group["main_property_id"]
        assert meta["collateral_group_id"] == gid

    # Allocation, formatted as the MILAN mapping formats a number.
    expected: list[str] = []
    for loan in provenance.loans.to_dicts():
        group = groups[loan["collateral_group_id"]]
        if group["allocation"] == "equal_split":
            value = group["total_property_value"] / group["n_loans"]
        else:
            value = (
                loan["current_principal_balance"] / group["total_loan_balance"]
            ) * group["total_property_value"]
        assert value == loan["aggregated_property_value"]
        expected.append(f"{value:.15g}")
    milan = compose_milan_pool(result.stage7.combined_flattened)
    assert milan["Property Value"].to_list() == expected
    assert milan["Loan Identifier"].to_list() == provenance.loans["calc_loan_id"].to_list()

    # The main property and its valuation metadata, as the flattened pool has them.
    flattened = result.stage7.combined_flattened
    for row in flattened.select(
        "calc_collateral_group_id",
        "calc_main_property_id",
        "final_valuation_method",
        "final_valuation_date",
    ).to_dicts():
        group = groups[row["calc_collateral_group_id"]]
        meta = provenance.records.row(group["metadata_record"], named=True)
        assert row["calc_main_property_id"] == group["main_property_id"]
        assert row["final_valuation_method"] == meta["valuation_method"]
        assert row["final_valuation_date"] == meta["valuation_date"]
